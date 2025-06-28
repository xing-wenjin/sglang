import dataclasses
import logging
import os
import re
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Set

import llm_datadist
import numpy as np
import torch
import zmq
from llm_datadist import CacheDesc, BlocksCacheKey, Cache
from llm_datadist import LLMDataDist, LLMRole, LLMConfig
from numpy import typing as npt

from sglang.srt.disaggregation.base import KVArgs, KVPoll, BaseKVSender
from sglang.srt.disaggregation.common import CommonKVManager, CommonKVReceiver, CommonKVBootstrapServer
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_by_remote

logger = logging.getLogger(__name__)

GUARD = "DataDistMsgGuard".encode("ascii")


class DataDistKVArgs(KVArgs):
    dp_rank: int


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]) -> "TransferInfo":
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            dst_kv_indices=np.frombuffer(msg[3], dtype=np.int32),
            dst_aux_index=int(msg[4].decode("ascii")),
            required_dst_info_num=int(msg[5].decode("ascii")),
        )


def get_device_ips():
    world_size = 8
    npu_info = subprocess.run(['npu-smi', 'info', '-m'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
    hccn_path = '/usr/local/Ascend/driver/tools/hccn_tool'
    if npu_info.returncode != 0 or not os.path.exists(hccn_path):
        raise RuntimeError("no npu-smi/hccn tools provided for NPU.")
    re_result = re.match(r'.*\n\t([0-9]+).*', npu_info.stdout)
    if re_result is None:
        raise RuntimeError("Can't find npu start index")
    npu_start_idx = int(re_result.group(1))
    device_ip_list = []
    for ip_offset in range(world_size):
        cmd = [
            hccn_path, '-i', f'{npu_start_idx + ip_offset}', '-ip', '-g'
        ]
        device_ip_info = subprocess.run(cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
        re_result = re.match(r'ipaddr:(.*)\n', device_ip_info.stdout)
        if re_result is None:
            raise RuntimeError("Can't find npu ip")
        device_ip = re_result.group(1)
        device_ip_list.append(device_ip)
    return device_ip_list


TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32
}


class DataDistKVManager(CommonKVManager):
    def __init__(
        self,
        args: DataDistKVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.registered_kv_caches: List[Cache] = []
        self.cluster_id = args.dp_rank  # kv_manager initial stage set dp_rank from scheduler
        self.device_ip_list = get_device_ips()
        self.device_id = self.kv_args.gpu_id + self.kv_args.engine_rank
        self.local_device_ip = self.device_ip_list[self.device_id]
        # bootstrap_room到状态的映射
        # todo考虑request_status的线程安全问题
        self.request_status: Dict[int, KVPoll] = {}
        # 初始化datadist
        llm_config = LLMConfig()
        llm_config.device_id = self.device_id
        llm_config.sync_kv_timeout = 20000
        llm_config.enable_cache_manager = True
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = LLMRole.PROMPT
            llm_config.listen_ip_info = f"{self.local_device_ip}:26000"  # todo cacheManager场景下不需要
            self.transfer_infos: Dict[int, Dict[int, TransferInfo]] = {}  # room到D侧传输信息的映射
            # 启动线程池异步传输kv_cache
            # todo 是否需要考虑多核cpu并行
            self.executor = ThreadPoolExecutor(max_workers=12)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.role = LLMRole.DECODER
            # 用来接受prefill发送的完成状态
            self.need_response_num: Dict[int, int] = {}
            self.response_tracker: Dict[int, Set[int]] = defaultdict(set)
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )
        self.llm_datadist = LLMDataDist(self.role, self.cluster_id)
        engine_options = llm_config.generate_options()
        self.llm_datadist.init(engine_options)

        # 注册内存
        self.cache_manager = self.llm_datadist.cache_manager
        self.register_buffer_to_engine()

        self.server_socket = zmq.Context().socket(zmq.PULL)
        # P侧创建zmq监听，接受D侧的握手信息
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.start_prefill_thread()

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            # D侧创建zmq监听接收P侧返回的传输完成消息
            self.start_decode_thread()
            self.register_link_lock = threading.Lock()
            self.link_registered = False

    def register_buffer_to_engine(self):
        # todo 通过参数获取到shape和dtype
        cache_desc = CacheDesc(num_tensors=len(self.kv_args.kv_data_ptrs),
                               shape=tuple(self.kv_args.kv_data_first.shape),
                               data_type=TORCH_DTYPE_TO_NPU_DTYPE[self.kv_args.kv_data_first.dtype])
        cache_addrs = self.kv_args.kv_data_ptrs
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            cache_key = BlocksCacheKey(self.cluster_id, 0)
        else:
            cache_key = None
        self.registered_kv_caches.append(self.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key))

        # metadata aux只注册output_ids
        output_ids = self.kv_args.aux_datas[0]
        cache_desc = CacheDesc(num_tensors=1, shape=tuple(output_ids.shape),
                               data_type=TORCH_DTYPE_TO_NPU_DTYPE[output_ids.dtype])
        cache_addrs = [output_ids.data_ptr()]
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            cache_key = BlocksCacheKey(self.cluster_id, 0)
        else:
            cache_key = None
        self.registered_kv_caches.append(self.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key))

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def register_link(self, bootstrap_infos):
        # receiver触发建链，只用建一次
        with self.register_link_lock:
            if self.link_registered:
                return
            cluster = llm_datadist.LLMClusterInfo()
            cluster.remote_cluster_id = self.cluster_id
            cluster.append_local_ip_info(self.local_device_ip, 0)
            for bootstrap_info in bootstrap_infos:
                remote_device_id = bootstrap_info["gpu_id"] + bootstrap_info["engine_rank"]
                remote_ip = self.device_ip_list[remote_device_id]
                cluster.append_remote_ip_info(remote_ip, 26000)
            self.llm_datadist.link_clusters([cluster], 20000)
            self.link_registered = True

    def start_prefill_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            while True:
                # receive and process handshake msg from KVReceiver bootstrap thread
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}, Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")

                required_dst_info_num = int(waiting_req_bytes[5].decode("ascii"))
                room = int(room)
                engine_rank = int(waiting_req_bytes[6].decode("ascii"))
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][engine_rank] = TransferInfo.from_zmq(waiting_req_bytes)
                # 多个D对应一个P场景，等待D全部握手，开始传输
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        if ":" in remote:
            remote = remote.split(":")[0]
        self._connect("tcp://" + remote + ":" + str(dst_port)).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ]
        )

    def start_decode_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def decode_thread():
            while True:
                (bootstrap_room, status, prefill_rank) = (
                    self.server_socket.recv_multipart()
                )
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        self.response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = self.need_response_num
                        arrived_response_num = len(self.response_tracker[bootstrap_room])
                        if (
                            self.is_mla_backend
                            or arrived_response_num == expected_response_num
                        ):
                            self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.update_status(bootstrap_room, KVPoll.Failed)

        threading.Thread(target=decode_thread).start()

    def sync_transfer_request(self, bootstrap_room: int,
                              kv_indices: npt.NDArray[np.int32],
                              index_slice: slice,
                              is_last: bool,
                              aux_index: Optional[int] = None):
        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy():
                continue
            chunked_dst_kv_indices = req.dst_kv_indices[index_slice]
            assert len(chunked_dst_kv_indices) == len(kv_indices)
            decode_cache_key = BlocksCacheKey(self.cluster_id, 0)
            # todo push_blocks接口没有参数填充D侧cacheId
            self.cache_manager.push_blocks(decode_cache_key,
                                           self.registered_kv_caches[0],
                                           kv_indices.tolist(),
                                           chunked_dst_kv_indices.tolist()
                                           )

            # Only the last chunk we need to send the aux data.
            if is_last:
                assert aux_index is not None
                # todo 发送失败的异常处理
                self.cache_manager.push_blocks(decode_cache_key,
                                               self.registered_kv_caches[1],
                                               [aux_index],
                                               [req.dst_aux_index]
                                               )
        if is_last:
            # 全部发送完成，同步状态到decoder
            self.update_status(bootstrap_room, KVPoll.Success)
            for req in reqs_to_be_processed:
                self.sync_status_to_decode_endpoint(
                    req.endpoint, req.dst_port, req.room, KVPoll.Success, self.kv_args.engine_rank
                )
            del self.transfer_infos[bootstrap_room]

    def add_transfer_request(self,
                             bootstrap_room: int,
                             kv_indices: npt.NDArray[np.int32],
                             index_slice: slice,
                             is_last: bool,
                             aux_index: Optional[int] = None):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)
        # 异步发送kv_cache
        self.executor.submit(
            lambda: self.sync_transfer_request(bootstrap_room, kv_indices, index_slice, is_last, aux_index))


class DataDistKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: DataDistKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, data_parallel_rank)
        # 触发llm-datadist建链
        mgr.register_link(self.bootstrap_infos)
        self.engine_rank = mgr.kv_args.engine_rank

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        kv_mgr: DataDistKVManager = self.kv_mgr
        kv_mgr.need_response_num[self.bootstrap_room] = len(self.bootstrap_infos)  # 记录D侧需要P侧响应的个数
        for bootstrap_info in self.bootstrap_infos:
            prefill_server_url = (
                f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            is_dummy = bootstrap_info["is_dummy"]
            sock, lock = self._connect("tcp://" + prefill_server_url)

            with lock:
                sock.send_multipart([
                    GUARD,
                    str(self.bootstrap_room).encode("ascii"),
                    get_local_ip_by_remote().encode("ascii"),
                    str(kv_mgr.rank_port).encode("ascii"),
                    kv_indices.tobytes() if not is_dummy else b"",
                    str(aux_index).encode("ascii"),
                    str(self.required_dst_info_num).encode("ascii"),
                    str(self.engine_rank).encode("ascii"),
                ])

        kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def clear(self) -> None:
        kv_mgr: DataDistKVManager = self.kv_mgr
        if self.bootstrap_room in kv_mgr.request_status:
            kv_mgr.request_status.pop(self.bootstrap_room)
        if self.bootstrap_room in kv_mgr.need_response_num:
            kv_mgr.need_response_num.pop(self.bootstrap_room)
        if self.bootstrap_room in kv_mgr.response_tracker:
            kv_mgr.response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()
        # todo 异常处理
        raise Exception("KVReceiver failure")


class DataDistKVSender(BaseKVSender):
    def __init__(self, mgr: DataDistKVManager, bootstrap_addr: str, bootstrap_room: int, dest_tp_ranks: List[int],
                 pp_rank: int):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.curr_idx = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(self, kv_indices: npt.NDArray[np.int32]):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.aux_index
        )

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()
        # todo 异常处理
        raise Exception("KVSender failure")


class DataDistKVBootstrapServer(CommonKVBootstrapServer):
    pass
