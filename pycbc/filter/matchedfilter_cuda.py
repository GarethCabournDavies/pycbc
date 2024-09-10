# Copyright (C) 2012-2024  Alex Nitz, Gareth Cabourn Davies
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


#from pycuda.elementwise import ElementwiseKernel
#from pycuda.tools import context_dependent_memoize
#from pycuda.tools import dtype_to_ctype
#from pycuda.gpuarray import _get_common_dtype
import cupy as cp
from .matchedfilter import _BaseCorrelator

kernel_code = """
extern "C" __global__
void correlate(const float2 *x, const float2 *y, float2 *z) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    float2 conj_x;
    conj_x.x = x[i].x; // Real part of x
    conj_x.y = -x[i].y; // Negative imaginary part of x

    // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    float real = conj_x.x * y[i].x - conj_x.y * y[i].y;
    float imag = conj_x.x * y[i].y + conj_x.y * y[i].x;

    z[i].x = real;
    z[i].y = imag;
}
"""

correlate_kernel = cp.RawKernel(kernel_code, 'correlate')

def correlate(a, b, out):
    # Launch the kernel
    block_dim = 256
    grid_dim = (a.size + block_dim - 1) // block_dim

    corrolate_kernel((grid_dim,), (block_dim,), (a.data, b.data, out.data))

class CUDACorrelator(_BaseCorrelator):
    def __init__(self, x, y, z):
        self.x = x.data
        self.y = y.data
        self.z = z.data
        self.krnl = correlate_kernel

    def correlate(self):
        # Launch the kernel
        block_dim = 256
        grid_dim = (self.x.size + block_dim - 1) // block_dim

        self.krnl((grid_dim,), (block_dim,), (self.x, self.y, self.z))

def _correlate_factory(x,y,z):
    return CUDACorrelator


