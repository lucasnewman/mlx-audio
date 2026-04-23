import threading
import time

from mlx_audio.server_inference import (
    BaseModelExecutionAdapter,
    InferenceBroker,
    InferenceRequest,
)


def _collect(handle, timeout=2.0):
    deadline = time.time() + timeout
    chunks = []
    while time.time() < deadline:
        chunk = handle.result_queue.get(timeout=timeout)
        chunks.append(chunk)
        if chunk.kind == "done":
            return chunks
    raise TimeoutError("timed out waiting for broker results")


class SerializedAdapter(BaseModelExecutionAdapter):
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def run_serial(self, request: InferenceRequest) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        time.sleep(0.05)
        request.emit_data(request.payload)

        with self.lock:
            self.active -= 1

        request.emit_done()


class BatchingAdapter(BaseModelExecutionAdapter):
    max_batch_size = 4

    def __init__(self):
        self.batch_sizes = []

    def supports_batch(self, request: InferenceRequest) -> bool:
        return request.payload != "serial"

    def batch_key(self, request: InferenceRequest):
        return "same"

    def run_serial(self, request: InferenceRequest) -> None:
        request.emit_data(("serial", request.payload))
        request.emit_done()

    def run_batch(self, requests: list[InferenceRequest]) -> None:
        self.batch_sizes.append(len(requests))
        for request in requests:
            request.emit_data(("batch", request.payload))
            request.emit_done()


class ContinuousAdapter(BatchingAdapter):
    def __init__(self):
        super().__init__()
        self.continuous_payloads = []

    def supports_continuous_batch(self, request: InferenceRequest) -> bool:
        return self.supports_batch(request)

    def run_continuous(self, request: InferenceRequest) -> None:
        self.continuous_payloads.append(request.payload)
        request.emit_data(("continuous", request.payload))
        request.emit_done()


def test_inference_broker_serializes_requests():
    broker = InferenceBroker()
    adapter = SerializedAdapter()
    broker.register_adapter("serial", adapter)

    try:
        first = broker.submit(
            endpoint_kind="serial",
            model_name="model-a",
            payload="first",
        )
        second = broker.submit(
            endpoint_kind="serial",
            model_name="model-a",
            payload="second",
        )

        assert _collect(first)[0].payload == "first"
        assert _collect(second)[0].payload == "second"
        assert adapter.max_active == 1
    finally:
        broker.stop_and_join()


def test_inference_broker_routes_continuous_batch_without_collect_window():
    broker = InferenceBroker()
    adapter = ContinuousAdapter()
    broker.register_adapter("batch", adapter)

    try:
        handle = broker.submit(
            endpoint_kind="batch",
            model_name="model-a",
            payload="first",
        )

        assert _collect(handle)[0].payload == ("continuous", "first")
        assert adapter.continuous_payloads == ["first"]
        assert adapter.batch_sizes == []
    finally:
        broker.stop_and_join()
