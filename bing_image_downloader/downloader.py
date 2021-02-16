import os
import shutil

try:
    from bing import Bing
except ImportError:  # Python 3
    from .bing import Bing


def download(query, limit=100, output_dir='dataset', adult_filter_off=True, timeout=60,
              dedup=True):

    # engine = 'bing'
    if adult_filter_off:
        adult = 'off'
    else:
        adult = 'on'

    bing = Bing(query, limit, output_dir, adult, timeout)
    bing.run()


if __name__ == '__main__':
    download('cat', limit=10, timeout='1')
