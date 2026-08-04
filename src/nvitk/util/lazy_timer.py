"""Console loading-bar and function-timing helpers."""

import sys
import time
from nvitk.core.logger import Logger

log = Logger()


class LoadingBar:
    """
    Class to create a loading bar in the console.
    :param n_iters: number of iterations
    :param length: length of the loading bar
    """
    def __init__(self, n_iters, length=50):
        """Initialize the bar for *n_iters* expected updates, drawn *length* characters wide."""
        self.iter = 0
        self.n_iters = n_iters
        self.length = length

    def update(self):
        """
        Update the loading bar.
        """
        self.iter += 1
        progress = int((self.iter / self.n_iters) * self.length)
        sys.stdout.write(
            f'\r[{progress * "=" + (self.length - progress) * " "}] '
            f'{self.iter / self.n_iters:.2%}'
        )
        sys.stdout.write('\n')
        sys.stdout.flush()

    def end(self):
        """
        End the loading bar.
        """
        self.iter = 0
        sys.stdout.write('\n')
        sys.stdout.flush()


def timed(fun):
    """
    Decorator to measure the time it takes for a function to execute.
    :param fun: function to be measured
    :return: wrapper function
    """
    def wrapper(*args, **kwargs):
        """Call the wrapped function, timing it and logging the elapsed seconds."""
        start = time.time()
        res = fun(*args, **kwargs)
        end = time.time()

        f_name = fun.__name__
        log.info(f"{f_name} took {end - start:.2f} seconds.")
        return res
    return wrapper