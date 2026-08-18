import logging

from vllm_ascend.worker import worker as worker_module


def test_profile_memory_skips_allocator_queries_without_debug(monkeypatch):
    worker = object.__new__(worker_module.NPUWorker)

    monkeypatch.setattr(worker_module.logger, "isEnabledFor", lambda level: False)

    def unexpected_query():
        raise AssertionError("allocator query must not run outside DEBUG logging")

    monkeypatch.setattr(worker_module.torch.npu, "memory_reserved", unexpected_query)
    monkeypatch.setattr(worker_module.torch.npu, "memory_allocated", unexpected_query)

    worker.profile_memory()


def test_profile_memory_collects_allocator_values_in_debug(monkeypatch):
    worker = object.__new__(worker_module.NPUWorker)
    worker.torch_reserved = 0
    worker.torch_allocated = 0

    monkeypatch.setattr(worker_module.logger, "isEnabledFor", lambda level: level == logging.DEBUG)
    monkeypatch.setattr(worker_module.logger, "debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module.torch.npu, "memory_reserved", lambda: 123)
    monkeypatch.setattr(worker_module.torch.npu, "memory_allocated", lambda: 45)

    worker.profile_memory()

    assert worker.torch_reserved == 123
    assert worker.torch_allocated == 45
