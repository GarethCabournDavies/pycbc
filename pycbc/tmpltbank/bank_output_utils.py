import logging
import numpy

from lal import PI, MTSUN_SI, TWOPI, GAMMA
from igwn_ligolw import ligolw, lsctables, utils as ligolw_utils

from pycbc import pnutils
from pycbc.tmpltbank.lambda_mapping import ethinca_order_from_string
from pycbc.io.ligolw import (
    return_empty_sngl, return_search_summary, create_process_table
)
from pycbc.io.hdf import HFile

from pycbc.waveform import get_waveform_filter_length_in_time as gwflit

logger = logging.getLogger('pycbc.tmpltbank.bank_output_utils')

def convert_to_sngl_inspiral_table(params, proc_id):
    '''
    Convert a list of m1,m2,spin1z,spin2z values into a basic sngl_inspiral
    table with mass and spin parameters populated and event IDs assigned

    Parameters
    -----------
    params : iterable
        Each entry in the params iterable should be a sequence of
        [mass1, mass2, spin1z, spin2z] in that order
    proc_id : int
        Process ID to add to each row of the sngl_inspiral table

    Returns
    ----------
    SnglInspiralTable
        Bank of templates in SnglInspiralTable format
    '''
    sngl_inspiral_table = lsctables.SnglInspiralTable.new()
    col_names = ['mass1','mass2','spin1z','spin2z']

    for values in params:
        tmplt = return_empty_sngl()

        tmplt.process_id = proc_id
        for colname, value in zip(col_names, values):
            setattr(tmplt, colname, value)
        tmplt.mtotal, tmplt.eta = pnutils.mass1_mass2_to_mtotal_eta(
            tmplt.mass1, tmplt.mass2)
        tmplt.mchirp, _ = pnutils.mass1_mass2_to_mchirp_eta(
            tmplt.mass1, tmplt.mass2)
        tmplt.template_duration = 0 # FIXME
        tmplt.event_id = sngl_inspiral_table.get_next_id()
        sngl_inspiral_table.append(tmplt)

    return sngl_inspiral_table

def calculate_ethinca_metric_comps(metricParams, ethincaParams, mass1, mass2,
                                   spin1z=0., spin2z=0., full_ethinca=True):
    """
    Calculate the Gamma components needed to use the ethinca metric.
    At present this outputs the standard TaylorF2 metric over the end time
    and chirp times \tau_0 and \tau_3.
    A desirable upgrade might be to use the \chi coordinates [defined WHERE?]
    for metric distance instead of \tau_0 and \tau_3.
    The lower frequency cutoff is currently hard-coded to be the same as the
    bank layout options fLow and f0 (which must be the same as each other).

    Parameters
    -----------
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    ethincaParams : ethincaParameters instance
        Structure holding options relevant to the ethinca metric computation.
    mass1 : float
        Mass of the heavier body in the considered template.
    mass2 : float
        Mass of the lighter body in the considered template.
    spin1z : float (optional, default=0)
        Spin of the heavier body in the considered template.
    spin2z : float (optional, default=0)
        Spin of the lighter body in the considered template.
    full_ethinca : boolean (optional, default=True)
        If True calculate the ethinca components in all 3 directions (mass1,
        mass2 and time). If False calculate only the time component (which is
        stored in Gamma0).
    Returns
    --------
    fMax_theor : float
        Value of the upper frequency cutoff given by the template parameters
        and the cutoff formula requested.

    gammaVals : numpy_array
        Array holding 6 independent metric components in
        (end_time, tau_0, tau_3) coordinates to be stored in the Gamma0-5
        slots of a SnglInspiral object.
    """
    if (float(spin1z) != 0. or float(spin2z) != 0.) and full_ethinca:
        raise NotImplementedError("Ethinca cannot at present be calculated "
                                  "for nonzero component spins!")
    f0 = metricParams.f0
    if f0 != metricParams.fLow:
        raise ValueError("If calculating ethinca the bank f0 value must be "
                         "equal to f-low!")
    if ethincaParams.fLow is not None and (
        ethincaParams.fLow != metricParams.fLow):
        raise NotImplementedError("An ethinca metric f-low different from the"
                                  " bank metric f-low is not supported!")

    twicePNOrder = ethinca_order_from_string(ethincaParams.pnOrder)

    piFl = PI * f0
    totalMass, eta = pnutils.mass1_mass2_to_mtotal_eta(mass1, mass2)
    totalMass = totalMass * MTSUN_SI
    v0cube = totalMass*piFl
    v0 = v0cube**(1./3.)

    # Get theoretical cutoff frequency and work out the closest
    # frequency for which moments were calculated
    fMax_theor = pnutils.frequency_cutoff_from_name(
        ethincaParams.cutoff, mass1, mass2, spin1z, spin2z)
    fMaxes = list(metricParams.moments['J4'].keys())
    fMaxIdx = abs(numpy.array(fMaxes,dtype=float) - fMax_theor).argmin()
    fMax = fMaxes[fMaxIdx]

    # Set the appropriate moments
    Js = numpy.zeros([18,3],dtype=float)
    for i in range(18):
        Js[i,0] = metricParams.moments['J%d'%(i)][fMax]
        Js[i,1] = metricParams.moments['log%d'%(i)][fMax]
        Js[i,2] = metricParams.moments['loglog%d'%(i)][fMax]

    # Compute the time-dependent metric term.
    two_pi_flower_sq = TWOPI * f0 * TWOPI * f0
    gammaVals = numpy.zeros([6],dtype=float)
    gammaVals[0] = 0.5 * two_pi_flower_sq * \
                    ( Js[(1,0)] - (Js[(4,0)]*Js[(4,0)]) )

    # If mass terms not required stop here
    if not full_ethinca:
        return fMax_theor, gammaVals

    # 3pN is a mess, so split it into pieces
    a0 = 11583231236531/200286535680 - 5*PI*PI - 107*GAMMA/14
    a1 = (-15737765635/130056192 + 2255*PI*PI/512)*eta
    a2 = (76055/73728)*eta*eta
    a3 = (-127825/55296)*eta*eta*eta
    alog = numpy.log(4*v0) # Log terms are tricky - be careful

    # Get the Psi coefficients
    Psi = [{},{}] #Psi = numpy.zeros([2,8,2],dtype=float)
    Psi[0][0,0] = 3/5
    Psi[0][2,0] = (743/756 + 11*eta/3)*v0*v0
    Psi[0][3,0] = 0.
    Psi[0][4,0] = (-3058673/508032 + 5429*eta/504 + 617*eta*eta/24)\
                    *v0cube*v0
    Psi[0][5,1] = (-7729*PI/126)*v0cube*v0*v0/3
    Psi[0][6,0] = (128/15)*(-3*a0 - a1 + a2 + 3*a3 + 107*(1+3*alog)/14)\
                    *v0cube*v0cube
    Psi[0][6,1] = (6848/35)*v0cube*v0cube/3
    Psi[0][7,0] = (-15419335/63504 - 75703*eta/756)*PI*v0cube*v0cube*v0

    Psi[1][0,0] = 0.
    Psi[1][2,0] = (3715/12096 - 55*eta/96)/PI/v0;
    Psi[1][3,0] = -3/2
    Psi[1][4,0] = (15293365/4064256 - 27145*eta/16128 - 3085*eta*eta/384)\
                    *v0/PI
    Psi[1][5,1] = (193225/8064)*v0*v0/3
    Psi[1][6,0] = (4/PI)*(2*a0 + a1/3 - 4*a2/3 - 3*a3 -107*(1+6*alog)/42)\
                    *v0cube
    Psi[1][6,1] = (-428/PI/7)*v0cube/3
    Psi[1][7,0] = (77096675/1161216 + 378515*eta/24192 + 74045*eta*eta/8064)\
                    *v0cube*v0

    # Set the appropriate moments
    Js = numpy.zeros([18,3],dtype=float)
    for i in range(18):
        Js[i,0] = metricParams.moments['J%d'%(i)][fMax]
        Js[i,1] = metricParams.moments['log%d'%(i)][fMax]
        Js[i,2] = metricParams.moments['loglog%d'%(i)][fMax]

    # Calculate the g matrix
    PNterms = [(0,0),(2,0),(3,0),(4,0),(5,1),(6,0),(6,1),(7,0)]
    PNterms = [term for term in PNterms if term[0] <= twicePNOrder]

    # Now can compute the mass-dependent gamma values
    for m in [0, 1]:
        for k in PNterms:
            gammaVals[1+m] += 0.5 * two_pi_flower_sq * Psi[m][k] * \
                                ( Js[(9-k[0],k[1])]
                                - Js[(12-k[0],k[1])] * Js[(4,0)] )

    g = numpy.zeros([2,2],dtype=float)
    for (m,n) in [(0,0),(0,1),(1,1)]:
        for k in PNterms:
            for l in PNterms:
                g[m,n] += Psi[m][k] * Psi[n][l] * \
                        ( Js[(17-k[0]-l[0], k[1]+l[1])]
                        - Js[(12-k[0],k[1])] * Js[(12-l[0],l[1])] )
        g[m,n] = 0.5 * two_pi_flower_sq * g[m,n]
        g[n,m] = g[m,n]

    gammaVals[3] = g[0,0]
    gammaVals[4] = g[0,1]
    gammaVals[5] = g[1,1]

    return fMax_theor, gammaVals

def output_sngl_inspiral_table(outputFile, tempBank, programName="",
                               optDict = None, outdoc=None,
                               **kwargs): # pylint:disable=unused-argument
    """
    Function that converts the information produced by the various PyCBC bank
    generation codes into a valid LIGOLW XML file containing a sngl_inspiral
    table and outputs to file.

    Parameters
    -----------
    outputFile : string
        Name of the file that the bank will be written to
    tempBank : iterable
        Each entry in the tempBank iterable should be a sequence of
        [mass1,mass2,spin1z,spin2z] in that order.
    programName (key-word-argument) : string
        Name of the executable that has been run
    optDict (key-word argument) : dictionary
        Dictionary of the command line arguments passed to the program
    outdoc (key-word argument) : ligolw xml document
        If given add template bank to this representation of a xml document and
        write to disk. If not given create a new document.
    kwargs : optional key-word arguments
        Allows unused options to be passed to this function (for modularity)
    """
    if optDict is None:
        optDict = {}
    if outdoc is None:
        outdoc = ligolw.Document()
        outdoc.appendChild(ligolw.LIGO_LW())

    # get IFO to put in search summary table
    ifos = []
    if 'channel_name' in optDict.keys():
        if optDict['channel_name'] is not None:
            ifos = [optDict['channel_name'][0:2]]

    proc = create_process_table(
        outdoc,
        program_name=programName,
        detectors=ifos,
        options=optDict
    )
    proc_id = proc.process_id
    sngl_inspiral_table = convert_to_sngl_inspiral_table(tempBank, proc_id)

    # set per-template low-frequency cutoff
    if 'f_low_column' in optDict and 'f_low' in optDict and \
            optDict['f_low_column'] is not None:
        for sngl in sngl_inspiral_table:
            setattr(sngl, optDict['f_low_column'], optDict['f_low'])

    outdoc.childNodes[0].appendChild(sngl_inspiral_table)

    # get times to put in search summary table
    start_time = 0
    end_time = 0
    if 'gps_start_time' in optDict.keys() and 'gps_end_time' in optDict.keys():
        start_time = optDict['gps_start_time']
        end_time = optDict['gps_end_time']

    # make search summary table
    search_summary_table = lsctables.SearchSummaryTable.new()
    search_summary = return_search_summary(
        start_time, end_time, len(sngl_inspiral_table), ifos
    )
    search_summary_table.append(search_summary)
    outdoc.childNodes[0].appendChild(search_summary_table)

    # write the xml doc to disk
    ligolw_utils.write_filename(outdoc, outputFile)


def output_bank_to_hdf(outputFile, tempBank, optDict=None, programName='',
                       approximant=None, output_duration=False,
                       **kwargs): # pylint:disable=unused-argument
    """
    Function that converts the information produced by the various PyCBC bank
    generation codes into a hdf5 file.

    Parameters
    -----------
    outputFile : string
        Name of the file that the bank will be written to
    tempBank : iterable
        Each entry in the tempBank iterable should be a sequence of
        [mass1,mass2,spin1z,spin2z] in that order.
    programName (key-word-argument) : string
        Name of the executable that has been run
    optDict (key-word argument) : dictionary
        Dictionary of the command line arguments passed to the program
    approximant : string
        The approximant to be outputted to the file,
        if output_duration is True, this is also used for that calculation.
    output_duration : boolean
        Output the duration of the template, calculated using
        get_waveform_filter_length_in_time, to the file.
    kwargs : optional key-word arguments
        Allows unused options to be passed to this function (for modularity)
    """
    bank_dict = {}
    mass1, mass2, spin1z, spin2z = list(zip(*tempBank))
    bank_dict['mass1'] = mass1
    bank_dict['mass2'] = mass2
    bank_dict['spin1z'] = spin1z
    bank_dict['spin2z'] = spin2z

    # Add other values to the bank dictionary as appropriate
    if optDict is not None:
        bank_dict['f_lower'] = numpy.ones_like(mass1) * \
            optDict['f_low']
        argument_string = [f'{k}:{v}' for k, v in optDict.items()]

    if optDict is not None and optDict['output_f_final']:
        bank_dict['f_final'] = numpy.ones_like(mass1) * \
            optDict['f_upper']

    if approximant:
        if not isinstance(approximant, bytes):
            appx = approximant.encode()
        bank_dict['approximant'] = numpy.repeat(appx, len(mass1))

    if output_duration:
        appx = approximant if approximant else 'SPAtmplt'
        tmplt_durations = numpy.zeros_like(mass1)
        for i in range(len(mass1)):
            wfrm_length = gwflit(appx,
                                 mass1=mass1[i],
                                 mass2=mass2[i],
                                 f_lower=optDict['f_low'],
                                 phase_order=7)
            tmplt_durations[i] = wfrm_length
        bank_dict['template_duration'] = tmplt_durations

    with HFile(outputFile, 'w') as bankf_out:
        bankf_out.attrs['program'] = programName
        if optDict is not None:
            bankf_out.attrs['arguments'] = argument_string
        for k, v in bank_dict.items():
            bankf_out[k] = v


def output_bank_to_file(outputFile, tempBank, **kwargs):
    if outputFile.endswith(('.xml','.xml.gz','.xmlgz')):
        output_sngl_inspiral_table(
            outputFile,
            tempBank,
            **kwargs
        )
    elif outputFile.endswith(('.h5','.hdf','.hdf5')):
        output_bank_to_hdf(
            outputFile,
            tempBank,
            **kwargs
        )
    else:
        err_msg = f"Unrecognized extension for file {outputFile}."
        raise ValueError(err_msg)
