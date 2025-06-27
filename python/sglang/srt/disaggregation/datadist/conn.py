import dataclasses
import os
import re
import subprocess
import threading
from typing import Optional, Dict, List

import llm_datadist
import numpy as np
import torch
import zmq
from llm_datadist import CacheDesc, BlocksCacheKey
from llm_datadist import LLMDataDist, LLMRole, LLMConfig
from numpy import typing as npt

from sglang.srt.disaggregation.base import KVArgs, KVPoll, BaseKVSender, BaseKVManager
from sglang.srt.disaggregation.common import CommonKVManager, CommonKVReceiver, CommonKVBootstrapServer
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_by_remote


@dataclasses.dataclass
class TransferInfo:
    # todo 握手信息定义
    pass


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
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.registered_kv_caches = []
        self.cluster_id = 0  # todo 根据dp_rank确定
        self.device_ip_list = get_device_ips()
        self.local_device_ip = self.device_ip_list[self.kv_args.gpu_id]

        # 初始化datadist
        llm_config = LLMConfig()
        llm_config.device_id = self.kv_args.gpu_id
        llm_config.sync_kv_timeout = 20000
        llm_config.enable_cache_manager = True
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = LLMRole.PROMPT
            llm_config.listen_ip_info = f"{self.local_device_ip}:26000"  # todo cacheManager场景下不需要

            self.transfer_infos: Dict[int, Dict[str, TransferInfo]] = {}
            self.request_status: Dict[int, KVPoll] = {}  # todo 状态更新
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.role = LLMRole.DECODER
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

        # P侧创建zmq监听，接受D侧的握手信息
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.server_socket = zmq.Context().socket(zmq.PULL)
            self.start_prefill_thread()

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            # D侧创建zmq监听接收P侧返回的传输完成消息
            self.start_decode_thread()
            # D侧datadist建链
            self.register_link()

    def register_buffer_to_engine(self):
        # todo 通过参数获取到shape和dtype
        cache_desc = CacheDesc(num_tensors=len(self.kv_args.kv_data_ptrs),
                               shape=tuple(self.kv_args.kv_data_first.shape),
                               data_type=TORCH_DTYPE_TO_NPU_DTYPE[self.kv_args.kv_data_first.dtype])
        cache_addrs = self.kv_args.kv_data_ptrs
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            cache_key = BlocksCacheKey(self.cluster_id, 0)
        else:
            cache_key = None
        self.registered_kv_caches.append(self.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key))

        # metadata aux只注册output_ids
        output_ids = self.kv_args.aux_datas[0]
        cache_desc = CacheDesc(num_tensors=1, shape=tuple(output_ids.shape),
                               data_type=TORCH_DTYPE_TO_NPU_DTYPE[output_ids.dtype])
        cache_addrs = [output_ids.data_ptr()]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            cache_key = BlocksCacheKey(self.cluster_id, 0)
        else:
            cache_key = None
        self.registered_kv_caches.append(self.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key))

    def register_link(self):
        # todo 考虑tp、dp不等于1的场景
        # 先实现一个demo版本，tp=1、dp=1，PD一对一
        cluster = llm_datadist.LLMClusterInfo()
        cluster.remote_cluster_id = self.cluster_id
        remote_ip = self.device_ip_list[0]  # todo 假设prefill的deviceId为0
        cluster.append_local_ip_info(self.local_device_ip, 0)
        cluster.append_remote_ip_info(remote_ip, 26000)
        self.llm_datadist.link_clusters([cluster], 20000)

    def start_prefill_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def bootstrap_thread():
            while True:
                # todo 接收receiver发送的握手消息
                pass

        threading.Thread(target=bootstrap_thread).start()

    def start_decode_thread(self):
        self.server_socket.bind(f"tcp://{get_local_ip_by_remote()}:{self.rank_port}")

        def decode_thread():
            while True:
                # todo 接收sender发送状态通知
                pass

        threading.Thread(target=decode_thread).start()


class DataDistKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: DataDistKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.conclude_state = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, data_parallel_rank)

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        pass

    def poll(self) -> KVPoll:
        pass

    def failure_exception(self):
        pass


class DataDistKVSender(BaseKVSender):
    def __init__(self, mgr: BaseKVManager, bootstrap_addr: str, bootstrap_room: int, dest_tp_ranks: List[int],
                 pp_rank: int):
        pass

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        pass

    def send(self, kv_indices: npt.NDArray[np.int32]):
        pass

    def poll(self) -> KVPoll:
        pass

    def failure_exception(self):
        pass


class DataDistKVBootstrapServer(CommonKVBootstrapServer):
    pass
