import bz2
import lz4.frame as lzframe
from typing import Literal

import zlib


class ResCoder(object):
    def __init__(self, coder: Literal['bzip', 'zlib', 'lz'] = 'zlib'):
        if coder not in {'bzip', 'zlib', 'lz'}:
            raise ValueError("rec_coder must be 'bzip', 'zlib', or 'lz'")
        self.res_coder = coder

    def compress(self, x):
        try:
            if self.res_coder == 'bzip':
                return bz2.compress(x, compresslevel=9)
            elif self.res_coder == 'zlib':
                return zlib.compress(x, level=9)
            elif self.res_coder == 'lz':
                return lzframe.compress(x, compression_level=16)
            else:
                raise ValueError("rec_coder must be 'bzip', 'zlib', or 'lz'")
        except Exception as err:
            print(f"RES编码过程中发生错误:{err}")

    def decompress(self, x):
        try:
            if self.res_coder == 'bzip':
                return bz2.decompress(x)
            elif self.res_coder == 'zlib':
                return zlib.decompress(x)
            elif self.res_coder == 'lz':
                return lzframe.decompress(x)
            else:
                raise ValueError("rec_coder must be 'bzip', 'zlib', or 'lz'")
        except Exception as err:
            print(f"RES编码过程中发生错误:{err}")
