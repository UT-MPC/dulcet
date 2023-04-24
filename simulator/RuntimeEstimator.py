from timeit import default_timer as timer
from datetime import timedelta
from statistics import mean

class RuntimeEstimator():
    def __init__(self, stop_time, queue_size=1000):
        self.queue_size = queue_size
        self.queue = []
        self.stop_time = stop_time
        self.last_estimate = 'estimating runtime...'

    def start_block(self, step):
        self.cur_step = step
        self.t0 = timer()

    def end_block(self):
        self.queue.append(timer() - self.t0)

        if len(self.queue) == self.queue_size:
            steps_left = self.stop_time - self.cur_step
            self.last_estimate = timedelta(seconds=mean(self.queue) * steps_left)
            self.queue = []

        return self.last_estimate