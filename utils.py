import time


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Time taken for %s: %s" % (func.__name__, end - start))
        return result
    return wrapper