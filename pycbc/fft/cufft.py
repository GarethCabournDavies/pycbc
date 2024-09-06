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
# cu_fft is treated as the module being unavailable at runtime.

try:
    import cupy
    import cupyx.scipy.fft as cu_fft
    from cupyx.scipy.fft import get_fft_plan
except ImportError as e:
    print("Unable to import cupy's scipy fft backend")
    raise e

from .core import _BaseFFT, _BaseIFFT

#_forward_plans = {}
#_reverse_plans = {}

#These dicts need to be cleared before the cuda context is destroyed
#def _clear_plan_dicts():
#    _forward_plans.clear()
#    _reverse_plans.clear()

#pycbc.scheme.register_clean_cuda(_clear_plan_dicts)

# No need for separate forward and inverse plan functions.
# Use get_fft_plan for both forward and inverse FFTs when necessary.

#def fft(invec, outvec, prec, itype, otype):
#    # Create a plan for FFT
#    with get_fft_plan(invec.data) as plan:
#        print(cu_fft.fft(invec.data))
#        outvec.data[:] = cupy.asarray(cu_fft.fft(invec.data), dtype=otype)
#
#def ifft(invec, outvec, prec, itype, otype):
#    # Create a plan for IFFT
#    with get_fft_plan(invec.data) as plan:
#        outvec[:] = cupy.asarray(cu_fft.ifft(invec.data), dtype=otype)

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
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(FFT, self).__init__(invec, outvec, nbatch, size)
        self.invec = invec
        self.outvec = outvec

    def execute(self):
        # Create and use an FFT plan for the execution
        with get_fft_plan(self.invec) as plan:
            self.outvec[:] = cu_fft.fft(self.invec, plan=plan)

class IFFT(_BaseIFFT):
    def __init__(self, invec, outvec, nbatch=1, size=None):
        super(IFFT, self).__init__(invec, outvec, nbatch, size)
        self.invec = invec
        self.outvec = outvec

    def execute(self):
        # Create and use an FFT plan for the execution
        with get_fft_plan(self.invec) as plan:
            self.outvec[:] = cu_fft.fft(self.invec, plan=plan)

