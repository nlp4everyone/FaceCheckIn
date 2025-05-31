import time
import logging

logging.basicConfig(level=logging.INFO)

class Timer:
    def __init__(self, name="Total"):
        self.name = name
        self.steps = []
        self._start = None

    def start(self):
        self._start = time.perf_counter()
        self.steps = [("START", self._start)]

    def mark(self, label):
        now = time.perf_counter()
        if self._start is None:
            raise RuntimeError("Timer not started. Call `.start()` first.")
        self.steps.append((label, now))

    def stop(self):
        self.mark("END")

    def report(self):
        statements = []
        for i in range(1, len(self.steps)):
            prev_label, prev_time = self.steps[i - 1]
            curr_label, curr_time = self.steps[i]
            duration = curr_time - prev_time
            statements.append(f"{curr_label}: {duration:.4f}s")
        return statements

