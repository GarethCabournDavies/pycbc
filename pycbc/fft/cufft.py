# Copyright (C) 2012-2024  Josh Willis, Andrew Miller, Gareth Cabourn Davies
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
This module provides the cupy scipy fft backend of the fast Fourier transform
for the PyCBC package.
"""

import pycbc.scheme
# The following is a hack, to ensure that any error in importing
# cupy is treated as the module being unavailable at runtime.

try:
    import cupy
except ImportError as e:
    print("Unable to import cupy")
    raise e

from .core import _check_fft_args
from .core import _BaseFFT, _BaseIFFT

def fft(invec, outvec, _, itype, otype):
    if invec.ptr == outvec.ptr:
        raise NotImplementedError("cupy backend of pycbc.fft does not "
                                  "support in-place transforms")
    if itype == 'complex' and otype == 'complex':
        outvec.data[:] = cupy.asarray(cupy.fft.fft(invec.data),
                                       dtype=outvec.dtype)
    elif itype == 'real' and otype == 'complex':
        outvec.data[:] = cupy.asarray(cupy.fft.rfft(invec.data),
                                       dtype=outvec.dtype)
    else:
        raise ValueError(_INV_FFT_MSG.format("FFT", itype, otype))


def ifft(invec, outvec, _, itype, otype):
    if invec.ptr == outvec.ptr:
        raise NotImplementedError("cupy backend of pycbc.fft does not "
                                  "support in-place transforms")
    if itype == 'complex' and otype == 'complex':
        outvec.data[:] = cupy.asarray(cupy.fft.ifft(invec.data),
                                       dtype=outvec.dtype)
        outvec *= len(outvec)
    elif itype == 'complex' and otype == 'real':
        outvec.data[:] = cupy.asarray(cupy.fft.irfft(invec.data,len(outvec)),
                                       dtype=outvec.dtype)
        outvec *= len(outvec)
    else:
        raise ValueError(_INV_FFT_MSG.format("IFFT", itype, otype))

class FFT(_BaseFFT):
    """
    Class for performing FFTs via the cupy interface.
    """
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(FFT, self).__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)

    def execute(self):
        fft(self.invec, self.outvec, self.prec, self.itype, self.otype)


class IFFT(_BaseIFFT):
    """
    Class for performing IFFTs via the cupy interface.
    """
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(IFFT, self).__init__(invec, outvec, nbatch, size)
        self.prec, self.itype, self.otype = _check_fft_args(invec, outvec)

    def execute(self):
        ifft(self.invec, self.outvec, self.prec, self.itype, self.otype)

