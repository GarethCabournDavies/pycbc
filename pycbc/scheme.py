# Copyright (C) 2014  Alex Nitz, Andrew Miller
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This modules provides python contexts that set the default behavior for PyCBC
objects.
"""
import os
import pycbc
from functools import wraps
import logging
from .libutils import get_ctypes_library
from .pool import use_mpi

logger = logging.getLogger('pycbc.scheme')


class _SchemeManager(object):
    _single = None

    def __init__(self):

        if _SchemeManager._single is not None:
            raise RuntimeError("SchemeManager is a private class")
        _SchemeManager._single= self

        self.state= None
        self._lock= False

    def lock(self):
        self._lock= True

    def unlock(self):
        self._lock= False

    def shift_to(self, state):
        if self._lock is False:
            self.state = state
        else:
            raise RuntimeError("The state is locked, cannot shift schemes")

# Create the global processing scheme manager
mgr = _SchemeManager()
DefaultScheme = None
default_context = None


class Scheme(object):
    """Context that sets PyCBC objects to use CPU processing. """
    _single = None
    def __init__(self):
        if DefaultScheme is type(self):
            return
        if Scheme._single is not None:
            raise RuntimeError("Only one processing scheme can be used")
        Scheme._single = True
    def __enter__(self):
        mgr.shift_to(self)
        mgr.lock()
    def __exit__(self, type, value, traceback):
        mgr.unlock()
        mgr.shift_to(default_context)
    def __del__(self):
        if Scheme is not None:
            Scheme._single = None

_cuda_cleanup_list=[]

def register_clean_cuda(function):
    _cuda_cleanup_list.append(function)

def clean_cuda(context):
    #Before cuda context is destroyed, all item destructions dependent on cuda
    # must take place. This calls all functions that have been registered
    # with _register_clean_cuda() in reverse order
    #So the last one registered, is the first one cleaned
    _cuda_cleanup_list.reverse()
    for func in _cuda_cleanup_list:
        func()

    context.pop()
    from pycuda.tools import clear_context_caches
    clear_context_caches()

class CUDAScheme(Scheme):
    """Context that sets PyCBC objects to use a CUDA processing scheme. """
    def __init__(self, device_num=0):
        Scheme.__init__(self)
        if not pycbc.HAVE_CUDA:
            raise RuntimeError("Install PyCUDA to use CUDA processing")
        import pycuda.driver
        pycuda.driver.init()
        self.device = pycuda.driver.Device(device_num)
        self.context = self.device.make_context(flags=pycuda.driver.ctx_flags.SCHED_BLOCKING_SYNC)
        import atexit
        atexit.register(clean_cuda,self.context)


class CUPYScheme(Scheme):
    """Scheme for using CUPY.

    Supports using CUPY with MPI. If MPI is enabled, will use all available
    devices. The environment variable `CUDA_VISIBLE_DEVICES` can be used to
    restrict the devices used.

    Parameters
    ----------
    device_num : int, optional
        The device number to use. If not provided, will use the default, 0.
        Should not be provided when using MPI to parallelize across devices.
    """
    def __init__(self, device_num=None):
        import cupy # Fail now if cupy is not there.
        import cupy.cuda

        do_mpi, _, rank = use_mpi(require_mpi=False, log=False)

        if device_num is not None and do_mpi:
            logger.warning("MPI is enabled, but a device number was provided.")

        if device_num is None and do_mpi:
            # Logical device numbers will always be 0, 1, 2, ... etc. irrespective
            # of the physical device numbers.
            device_num = rank % cupy.cuda.runtime.getDeviceCount()
            logger.debug("MPI enabled, using CUDA device %s", device_num)

        self.device_num = device_num
        self.cuda_device = cupy.cuda.Device(self.device_num)

    def __enter__(self):
        super().__enter__()
        self.cuda_device.__enter__()
        logger.warning(
            "You are using the CUPY GPU backend for PyCBC. This backend is "
            "still only a prototype. It may be useful for your application "
            "but it may fail unexpectedly, run slowly, or not give correct "
            "output. Please do contribute to the effort to develop this "
            "further."
        )

    def __exit__(self, *args):
        super().__exit__(*args)
        self.cuda_device.__exit__(*args)


class CPUScheme(Scheme):
    def __init__(self, num_threads=1):
        if isinstance(num_threads, int):
            self.num_threads=num_threads
        elif num_threads == 'env' and "PYCBC_NUM_THREADS" in os.environ:
            self.num_threads = int(os.environ["PYCBC_NUM_THREADS"])
        else:
            import multiprocessing
            self.num_threads = multiprocessing.cpu_count()
        self._libgomp = None

    def __enter__(self):
        Scheme.__enter__(self)
        try:
            self._libgomp = get_ctypes_library("gomp", ['gomp'],
                                               mode=ctypes.RTLD_GLOBAL)
        except:
            # Should we fail or give a warning if we cannot import
            # libgomp? Seems to work even for MKL scheme, but
            # not entirely sure why...
            pass

        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        if self._libgomp is not None:
            self._libgomp.omp_set_num_threads( int(self.num_threads) )

    def __exit__(self, type, value, traceback):
        os.environ["OMP_NUM_THREADS"] = "1"
        if self._libgomp is not None:
            self._libgomp.omp_set_num_threads(1)
        Scheme.__exit__(self, type, value, traceback)

class MKLScheme(CPUScheme):
    def __init__(self, num_threads=1):
        CPUScheme.__init__(self, num_threads)
        if not pycbc.HAVE_MKL:
            raise RuntimeError("Can't find MKL libraries")

class NumpyScheme(CPUScheme):
    pass


scheme_prefix = {
    CUDAScheme: "cuda",
    CPUScheme: "cpu",
    CUPYScheme: "cupy",
    MKLScheme: "mkl",
    NumpyScheme: "numpy",
}
_scheme_map = {v: k for (k, v) in scheme_prefix.items()}

_default_scheme_prefix = os.getenv("PYCBC_SCHEME", "cpu")
try:
    _default_scheme_class = _scheme_map[_default_scheme_prefix]
except KeyError as exc:
    raise RuntimeError(
        "PYCBC_SCHEME={!r} not recognised, please select one of: {}".format(
            _default_scheme_prefix,
            ", ".join(map(repr, _scheme_map)),
        ),
    )

class DefaultScheme(_default_scheme_class):
    pass

default_context = DefaultScheme()
mgr.state = default_context
scheme_prefix[DefaultScheme] = _default_scheme_prefix

def current_prefix():
    return scheme_prefix[type(mgr.state)]

_import_cache = {}
def schemed(prefix):

    def scheming_function(func):
        @wraps(func)
        def _scheming_function(*args, **kwds):
            try:
                return _import_cache[mgr.state][func](*args, **kwds)
            except KeyError:
                exc_errors = []
                for sch in mgr.state.__class__.__mro__[0:-2]:
                    try:
                        backend = __import__(prefix + scheme_prefix[sch],
                                             fromlist=[func.__name__])
                        schemed_fn = getattr(backend, func.__name__)
                    except (ImportError, AttributeError) as e:
                        exc_errors += [e]
                        continue

                    if mgr.state not in _import_cache:
                        _import_cache[mgr.state] = {}

                    _import_cache[mgr.state][func] = schemed_fn

                    return schemed_fn(*args, **kwds)

                err = (f"Failed to find implementation of {func.__name__} "
                       f"for {current_prefix()} scheme. ")
                for emsg in exc_errors:
                    err += str(emsg) + " "
                raise RuntimeError(err)
        return _scheming_function

    return scheming_function

def cpuonly(func):
    @wraps(func)
    def _cpuonly(*args, **kwds):
        if not issubclass(type(mgr.state), CPUScheme):
            raise TypeError(fn.__name__ +
                            " can only be called from a CPU processing scheme.")
        else:
            return func(*args, **kwds)
    return _cpuonly

def insert_processing_option_group(parser):
    """
    Adds the options used to choose a processing scheme. This should be used
    if your program supports the ability to select the processing scheme.

    Parameters
    ----------
    parser : object
        OptionParser instance
    """
    processing_group = parser.add_argument_group("Options for selecting the"
                                   " processing scheme in this program.")
    processing_group.add_argument("--processing-scheme",
                      help="The choice of processing scheme. "
                           "Choices are " + str(list(set(scheme_prefix.values()))) +
                           ". (optional for CPU scheme) The number of "
                           "execution threads "
                           "can be indicated by cpu:NUM_THREADS, "
                           "where NUM_THREADS "
                           "is an integer. The default is a single thread. "
                           "If the scheme is provided as cpu:env, the number "
                           "of threads can be provided by the PYCBC_NUM_THREADS "
                           "environment variable. If the environment variable "
                           "is not set, the number of threads matches the number "
                           "of logical cores. ",
                      default="cpu")

    processing_group.add_argument("--processing-device-id",
                      help="(optional) ID of GPU to use for accelerated "
                           "processing",
                      default=0, type=int)

def from_cli(opt):
    """Parses the command line options and returns a processing scheme.

    Parameters
    ----------
    opt: object
        Result of parsing the CLI with OptionParser, or any object with
        the required attributes.

    Returns
    -------
    ctx: Scheme
        Returns the requested processing scheme.
    """
    scheme_str = opt.processing_scheme.split(':')
    name = scheme_str[0]

    if name == "cuda":
        logger.info("Running with CUDA support")
        ctx = CUDAScheme(opt.processing_device_id)
    elif name == "mkl":
        if len(scheme_str) > 1:
            numt = scheme_str[1]
            if numt.isdigit():
                numt = int(numt)
            ctx = MKLScheme(num_threads=numt)
        else:
            ctx = MKLScheme()
        logger.info("Running with MKL support: %s threads" % ctx.num_threads)
    elif name == 'cupy':
        logger.info("Running with CUPY support")
        ctx = CUPYScheme()
    else:
        if len(scheme_str) > 1:
            numt = scheme_str[1]
            if numt.isdigit():
                numt = int(numt)
            ctx = CPUScheme(num_threads=numt)
        else:
            ctx = CPUScheme()
        logger.info("Running with CPU support: %s threads" % ctx.num_threads)
    return ctx

def verify_processing_options(opt, parser):
    """Parses the  processing scheme options and verifies that they are
       reasonable.


    Parameters
    ----------
    opt : object
        Result of parsing the CLI with OptionParser, or any object with the
        required attributes.
    parser : object
        OptionParser instance.
    """
    scheme_types = scheme_prefix.values()
    if opt.processing_scheme.split(':')[0] not in scheme_types:
        parser.error("(%s) is not a valid scheme type.")

class ChooseBySchemeDict(dict):
    """ This class represents a dictionary whose purpose is to chose objects
    based on their processing scheme. The keys are intended to be processing
    schemes.
    """
    def __getitem__(self, scheme):
        for base in scheme.__mro__[0:-1]:
            try:
                return dict.__getitem__(self, base)
                break
            except:
                pass

