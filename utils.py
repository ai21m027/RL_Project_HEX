import time
import logging


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def measure_time(func, message=None):
    message = message if message is not None else func.__name__
    def wrapper(*args, **kwargs):
        logging.debug("Starting %s" % message)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.debug("%s took: %s" % (message, end - start))
        return result
    return wrapper