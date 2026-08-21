from unittest.mock import MagicMock, patch

from vllm_ascend.cpu_binding import CpuAlloc, bind_process_fraction_from_current_affinity


def test_bind_process_fraction_uses_relative_main_pool():
    allocator = object.__new__(CpuAlloc)
    allocator.rank_id = 0
    allocator.device_info = MagicMock(running_npu_list=[14])
    allocator.assign_main = {14: list(range(282, 298))}
    process = MagicMock()
    process.threads.return_value = [MagicMock(id=100), MagicMock(id=101)]

    with patch("vllm_ascend.cpu_binding.psutil.Process", return_value=process), patch(
        "vllm_ascend.cpu_binding.os.sched_setaffinity"
    ) as set_affinity:
        allocator.bind_process_fraction((0.0, 0.625))

    assert set_affinity.call_count == 2
    for call in set_affinity.call_args_list:
        assert call.args[1] == list(range(282, 292))


def test_bind_process_fraction_uses_tail_slice():
    allocator = object.__new__(CpuAlloc)
    allocator.rank_id = 0
    allocator.device_info = MagicMock(running_npu_list=[14])
    allocator.assign_main = {14: list(range(282, 298))}
    process = MagicMock()
    process.threads.return_value = [MagicMock(id=100)]

    with patch("vllm_ascend.cpu_binding.psutil.Process", return_value=process), patch(
        "vllm_ascend.cpu_binding.os.sched_setaffinity"
    ) as set_affinity:
        allocator.bind_process_fraction((0.625, 1.0))

    assert set_affinity.call_args.args[1] == list(range(292, 298))


def test_bind_process_fraction_falls_back_to_inherited_cpuset():
    process = MagicMock()
    process.threads.return_value = [MagicMock(id=100), MagicMock(id=101)]

    with patch("vllm_ascend.cpu_binding.os.sched_getaffinity", return_value={2, 4, 8, 10}), patch(
        "vllm_ascend.cpu_binding.psutil.Process", return_value=process
    ), patch("vllm_ascend.cpu_binding.os.sched_setaffinity") as set_affinity:
        selected = bind_process_fraction_from_current_affinity((0.5, 1.0))

    assert selected == [8, 10]
    assert set_affinity.call_count == 2
    for call in set_affinity.call_args_list:
        assert call.args[1] == [8, 10]
