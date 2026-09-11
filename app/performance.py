"""Small, bounded caches for data shared by the dashboard pages."""
from functools import wraps
import hashlib
import pickle
from django.core.cache import cache


def cached_result(seconds, empty_seconds=15):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            digest = hashlib.sha256(pickle.dumps((args, sorted(kwargs.items())))).hexdigest()
            key = f'gold:v1:{function.__module__}.{function.__name__}:{digest}'
            result = cache.get(key)
            if result is not None:
                return result
            result = function(*args, **kwargs)
            empty = result.empty if hasattr(result, 'empty') else not bool(result)
            cache.set(key, result, empty_seconds if empty else seconds)
            return result
        return wrapped
    return decorate
