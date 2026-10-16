import time
import logging
import functools

logger = logging.getLogger(__name__)

def log_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s takes %.3f s", func.__name__, elapsed)
        return result
    return wrapper