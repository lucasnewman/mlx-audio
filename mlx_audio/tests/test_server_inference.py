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
