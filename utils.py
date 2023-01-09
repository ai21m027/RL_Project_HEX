import os
import time
import logging

def measure_time(func, message=None):
    message = message if message is not None else func.__name__
    def wrapper(*args, **kwargs):
        logging.debug("Starting %s" % message)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info("%s took: %s" % (message, end - start))
        return result
    return wrapper

def file_write_atomic(path, callback, worker_id=None):    
    temp_path = "tmp" if worker_id is None else f"tmp/worker_{worker_id}"

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    file_name = os.path.basename(path)
    temp_file = f"{temp_path}/{file_name}"

    callback(temp_file)
    os.rename(temp_file, path)
    if worker_id is not None:
        os.rmdir(temp_path)