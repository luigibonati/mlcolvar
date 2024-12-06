import sys
import time
import typing

"""
A simple progress bar.
"""

__all__ = ['pbar']


def pbar(
    item: typing.List[int],
    prefix: str = '',
    size: int = 25,
    frequency: int = 0.05,
    use_unicode: bool = True,
    file: typing.TextIO = sys.stdout
):
    """
    A simple progress bar. Taken from stackoverflow:
    https://stackoverflow.com/questions/3160699

    Parameters
    ----------
    it : List[int]
        The looped item.
    prefix : str
        Prefix of the bar.
    size : int
        Size of the bar.
    frequency : float
        Flush frequency of the bar.
    use_unicode : bool
        If use unicode char to draw the bar.
    file : TextIO
        The output file.
    """
    if (use_unicode):
        c_1 = ''
        c_2 = '█'
        c_3 = '━'
        c_4 = ''
    else:
        c_1 = '|'
        c_2 = '|'
        c_3 = '-'
        c_4 = '|'
    count = len(item)
    start = time.time()
    interval = max(int(count * frequency), 1)

    def show(j) -> None:
        x = int(size * j / count)
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60)
        time_string = f'{int(mins):02}:{sec:02.1f}'
        output = f' {prefix} {c_1}{c_2 * (x - 1) + c_4}{c_3 * (size - x)} ' + \
                 f'{j}/{count} Est. {time_string}'
        print('\x1b[1A\x1b[2K' + output, file=file, flush=True)

    for i, it in enumerate(item):
        yield it
        if ((i % interval) == 0 or i in [0, (count - 1)]):
            show(i + 1)
    print(flush=True, file=file)
