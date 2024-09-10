# Copyright (C) 2012  Alex Nitz
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
import logging
import numpy, mako.template
import cupy as cp
#from pycuda.tools import dtype_to_ctype
#from pycuda.elementwise import ElementwiseKernel
#from pycuda.compiler import SourceModule
from .eventmgr import _BaseThresholdCluster
import pycbc.scheme

logger = logging.getLogger('pycbc.events.threshold_cuda')

kernel_code = """
extern "C" __global__
void getstuff(const float2 *in, const float2 *outv, unsigned int *outl, float threshold, unsigned int *bn) {
    if (i == 0)
        bn[0] = 0;

    cuComplex val = in[i];
    if ( cuCabsf(val) > threshold){
        int n_w = atomicAdd(bn, 1);
        outv[n_w] = val;
        outl[n_w] = i;
    }
}
"""

threshold_kernel = cp.RawKernel(kernel_code, 'getstuff')

# Allocate device memory
#n = cp.cuda.alloc_pinned_memory(1 * cp.uint32().itemsize)
#nptr = cp.cuda.MemoryPointer(n, 0)
# Allocate device memory
n = cp.zeros((1), dtype=cp.uint32)
nptr = n.data.ptr

#val = cp.cuda.alloc_pinned_memory(4096 * 256 * cp.complex64().itemsize)
#vptr = cp.cuda.MemoryPointer(val, 0)
val = cp.zeros((4096 * 256), dtype=cp.complex64)
vptr = val.data.ptr

#loc = cp.cuda.alloc_pinned_memory(4096 * 256 * cp.int32().itemsize)
#lptr = cp.cuda.MemoryPointer(loc, 0)
loc = cp.zeros((4096 * 256), dtype=cp.int32)
lptr = loc.data.ptr

class T():
    pass

tn = T()
tv = T()
tl = T()
tn.gpudata = nptr
tv.gpudata = vptr
tl.gpudata = lptr
tn.flags = tv.flags = tl.flags = n.flags

import cupy as cp
import numpy as np
from cupy import RawKernel

# Define the CUDA kernel source code
threshold_kernel_source = """
extern "C" __global__ void threshold_and_cluster(float2* in, float2* outv, int* outl, int window, float threshold){
    int s = window * blockIdx.x;
    int e = s + window;

    // shared memory for chunk size candidates
    __shared__ float svr[128]; // Adjust size as needed
    __shared__ float svi[128];
    __shared__ int sl[128];

    // shared memory for the warp size candidates
    __shared__ float svv[32];
    __shared__ int idx[32];

    int ml = -1;
    float mvr = 0;
    float mvi = 0;
    float re;
    float im;

    // Iterate through the entire window size chunk and find blockDim.x number
    // of candidates
    for (int i = s + threadIdx.x; i < e; i += blockDim.x){
        re = in[i].x;
        im = in[i].y;
        if ((re * re + im * im) > (mvr * mvr + mvi * mvi)){
            mvr = re;
            mvi = im;
            ml = i;
        }
    }

    // Save the candidate from this thread to shared memory
    svr[threadIdx.x] = mvr;
    svi[threadIdx.x] = mvi;
    sl[threadIdx.x] = ml;

    __syncthreads();

    if (threadIdx.x < 32){
        int tl = threadIdx.x;

        // Now that we have all the candidates for this chunk in shared memory
        // Iterate through in the warp size to reduce to 32 candidates
        for (int i = threadIdx.x; i < 128; i += 32){
            re = svr[i];
            im = svi[i];
            if ((re * re + im * im) > (mvr * mvr + mvi * mvi)){
                tl = i;
                mvr = re;
                mvi = im;
            }
        }

        // Store the 32 candidates into shared memory
        svv[threadIdx.x] = svr[tl] * svr[tl] + svi[tl] * svi[tl];
        idx[threadIdx.x] = tl;

        // Find the 1 candidate we are looking for using a manual log algorithm
        if ((threadIdx.x < 16) && (svv[threadIdx.x] < svv[threadIdx.x + 16])){
            svv[threadIdx.x] = svv[threadIdx.x + 16];
            idx[threadIdx.x] = idx[threadIdx.x + 16];
        }

        if ((threadIdx.x < 8) && (svv[threadIdx.x] < svv[threadIdx.x + 8])){
            svv[threadIdx.x] = svv[threadIdx.x + 8];
            idx[threadIdx.x] = idx[threadIdx.x + 8];
        }

        if ((threadIdx.x < 4) && (svv[threadIdx.x] < svv[threadIdx.x + 4])){
            svv[threadIdx.x] = svv[threadIdx.x + 4];
            idx[threadIdx.x] = idx[threadIdx.x + 4];
        }

        if ((threadIdx.x < 2) && (svv[threadIdx.x] < svv[threadIdx.x + 2])){
            svv[threadIdx.x] = svv[threadIdx.x + 2];
            idx[threadIdx.x] = idx[threadIdx.x + 2];
        }

        // Save the 1 candidate maximum and location to the output vectors
        if (threadIdx.x == 0){
            if (svv[threadIdx.x] < svv[threadIdx.x + 1]){
                idx[0] = idx[1];
                svv[0] = svv[1];
            }

            if (svv[0] > threshold){
                int tl = idx[0];
                outv[blockIdx.x].x = svr[tl];
                outv[blockIdx.x].y = svi[tl];
                outl[blockIdx.x] = sl[tl];
            } else{
                outl[blockIdx.x] = -1;
            }
        }
    }
}

extern "C" __global__ void threshold_and_cluster2(float2* outv, int* outl, float threshold, int window){
    __shared__ int loc[128]; // Adjust size as needed
    __shared__ float val[128];

    int i = threadIdx.x;

    int l = outl[i];
    loc[i] = l;

    if (l == -1)
        return;

    val[i] = outv[i].x * outv[i].x + outv[i].y * outv[i].y;

    // Check right
    if ((i < (128 - 1)) && (val[i + 1] > val[i])){
        outl[i] = -1;
        return;
    }

    // Check left
    if ((i > 0) && (val[i - 1] > val[i])){
        outl[i] = -1;
        return;
    }
}
"""

# Compile the CUDA kernels
raw_kernel = RawKernel(threshold_kernel_source, 'threshold_and_cluster')
raw_kernel2 = RawKernel(threshold_kernel_source, 'threshold_and_cluster2')

def threshold_and_cluster(series, threshold, window):
    slen = len(series)
    threshold = np.float32(threshold * threshold)
    window = np.int32(window)
    
    # Determine block and grid sizes based on window size
    if window <= 4096:
        nt = 128
    elif window <= 16384:
        nt = 256
    elif window <= 32768:
        nt = 512
    else:
        nt = 1024

    nb = int(np.ceil(slen / float(window)))

    if nb > 1024:
        raise ValueError("More than 1024 blocks not supported yet")

    # Prepare output arrays
    outl = cp.zeros(nb, dtype=cp.int32)
    outv = cp.zeros((nb,), dtype=cp.complex64)
    series_gpu = cp.asarray(series, dtype=cp.complex64)

    # Launch the first kernel
    raw_kernel((nb,), (nt,), (series_gpu, outv, outl, window, threshold))
    
    # Launch the second kernel
    raw_kernel2((1,), (nb,), (outv, outl, threshold, window))

    # Synchronize
    cp.cuda.Device().synchronize()

    # Filter valid results
    cl = cp.asnumpy(outl)
    cv = cp.asnumpy(outv)
    valid_indices = cl != -1
    return cv[valid_indices], cl[valid_indices]

class CUDAThresholdCluster:
    def __init__(self, series):
        self.series = cp.asarray(series, dtype=cp.complex64)
        self.outl = cp.zeros(len(series), dtype=cp.int32)
        self.outv = cp.zeros((len(series),), dtype=cp.complex64)
        self.slen = len(series)

    def threshold_and_cluster(self, threshold, window):
        threshold = np.float32(threshold * threshold)
        window = np.int32(window)
        
        # Determine block and grid sizes based on window size
        if window <= 4096:
            nt = 128
        elif window <= 16384:
            nt = 256
        elif window <= 32768:
            nt = 512
        else:
            nt = 1024

        nb = int(np.ceil(self.slen / float(window)))

        if nb > 1024:
            raise ValueError("More than 1024 blocks not supported yet")

        # Launch the first kernel
        raw_kernel((nb,), (nt,), (self.series, self.outv, self.outl, window, threshold))
        
        # Launch the second kernel
        raw_kernel2((1,), (nb,), (self.outv, self.outl, threshold, window))

        # Synchronize
        cp.cuda.Device().synchronize()

        # Filter valid results
        cl = cp.asnumpy(self.outl)
        cv = cp.asnumpy(self.outv)
        valid_indices = cl != -1
        return cv[valid_indices], cl[valid_indices]

def _threshold_cluster_factory(series):
    return CUDAThresholdCluster
