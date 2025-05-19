from collections import deque
from queue import Queue
from threading import Event, Lock, Thread

import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self, sample_rate=24_000, buffer_size=2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.playing = False
        self.drain_event = Event()
        self._enqueue_queue = Queue()
        self._enqueue_thread = Thread(target=self._enqueue_worker, daemon=True)
        self._enqueue_thread.start()

    def callback(self, outdata, frames, time, status):
        with self.buffer_lock:
            if len(self.audio_buffer) > 0:
                available = min(frames, len(self.audio_buffer[0]))
                chunk = self.audio_buffer[0][:available].copy()
                self.audio_buffer[0] = self.audio_buffer[0][available:]

                if len(self.audio_buffer[0]) == 0:
                    self.audio_buffer.popleft()
                    if len(self.audio_buffer) == 0:
                        self.drain_event.set()

                outdata[:, 0] = np.zeros(frames)
                outdata[:available, 0] = chunk
            else:
                outdata[:, 0] = np.zeros(frames)
                self.drain_event.set()

    def play(self):
        if not self.playing:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.callback,
                blocksize=self.buffer_size,
            )
            self.stream.start()
            self.playing = True
            self.drain_event.clear()

    def queue_audio(self, samples):
        self.drain_event.clear()

        with self.buffer_lock:
            self.audio_buffer.append(np.array(samples))
        if not self.playing:
            self.play()

    def queue_audio_async(self, samples):
        """Enqueue audio samples from any thread without blocking."""
        if not self._enqueue_thread.is_alive():
            self._enqueue_thread = Thread(target=self._enqueue_worker, daemon=True)
            self._enqueue_thread.start()
        self._enqueue_queue.put(np.array(samples))

    def _enqueue_worker(self):
        while True:
            samples = self._enqueue_queue.get()
            if samples is None:
                break
            self.queue_audio(samples)
            self._enqueue_queue.task_done()

    def wait_for_drain(self):
        return self.drain_event.wait()

    def stop(self):
        if self.playing:
            self.wait_for_drain()
            sd.sleep(100)

            self.stream.stop()
            self.stream.close()
            self.playing = False
        # signal enqueue worker to exit
        if self._enqueue_thread.is_alive():
            self._enqueue_queue.put(None)
            self._enqueue_thread.join(timeout=0.1)

    def flush(self):
        """Discard everything and stop playback immediately."""
        if not self.playing:
            return

        with self.buffer_lock:
            self.audio_buffer.clear()

        #  abort() is instantaneous; stop() waits for drain
        try:
            self.stream.abort()
        except AttributeError:  # older sounddevice
            self.stream.stop(ignore_errors=True)

        self.stream.stop()
        self.stream.close()
        self.playing = False
        self.drain_event.set()
