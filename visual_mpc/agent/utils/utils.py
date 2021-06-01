import time
from contextlib import contextmanager

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d


@contextmanager
def timing(text):
    start = time.time()
    yield
    elapsed = time.time() - start
    print("{} {}".format(text, elapsed))


class timed:
    """ A function decorator that prints the elapsed time """

    def __init__(self, text):
        """ Decorator parameters """
        self.text = text

    def __call__(self, func):
        """ Wrapping """

        def wrapper(*args, **kwargs):
            with timing(self.text):
                result = func(*args, **kwargs)
            return result

        return wrapper




