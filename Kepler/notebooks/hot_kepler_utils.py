import numpy as np
import matplotlib.pyplot as plt
import glob
import fitsio

from astropy.timeseries import LombScargle
import astropy.units as u
import warnings

# general
import copy
from scipy.ndimage import gaussian_filter1d as gaussfilt
from matplotlib import rc
from time import time as clock

from lightkurve import KeplerLightCurveFile, KeplerLightCurve
import lightkurve
from numpy.core.records import array as rarr

# planet search stuff
from scipy.ndimage import binary_dilation
from acor import acor

from oxksc.cbvc import cbv

from astropy.timeseries import BoxLeastSquares

import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt


#plotting
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplots, subplot
colours = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

from IPython import get_ipython

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore") # living dangerously

def fold(time, period, origo=0.0, shift=0.0, normalize=True,  clip_range=None):
    """Folds the given data over a given period.
    Parameters
    ----------

      time
      period
      origo
      shift
      normalize
      clip_range
    Returns
    -------
      phase
    """
    tf = ((time - origo)/period + shift) % 1.

    if not normalize:
        tf *= period

    if clip_range is not None:
        mask = np.logical_and(clip_range[0]<tf, tf<clip_range[1])
        tf = tf[mask], mask
    return tf

def find_cbv(quarter):
    ## To Do: fit it to the individual module and output
    fname = glob.glob('../data/kplr_cbv/*q%02d*-d25_lcbv.fits' % quarter)[0]
    return fname

def get_num(quarter):
    '''Get the file number for each quarter'''
    if quarter == 0:
        return 2009131105131
    elif quarter == 1:
        return 2009166043257
    elif quarter == 2:
        return 2009259160929
    elif quarter == 3:
        return 2009350155506
    elif quarter == 4: # two sets of collateral - this is released 2014-12-17 16:15:42
        return 2010078095331
    # elif quarter == 4: # two sets of collateral - this is released 2012-04-25 13:41:56
    #   return 2010078170814
    elif quarter == 5:
        return 2010174085026
    elif quarter == 6:
        return 2010265121752
    elif quarter == 7:
        return 2010355172524
    elif quarter == 8:
        return 2011073133259
    elif quarter == 9:
        return 2011177032512
    elif quarter == 10:
        return 2011271113734
    elif quarter == 11:
        return 2012004120508
    elif quarter == 12:
        return 2012088054726
    elif quarter == 13:
        return 2012179063303
    elif quarter == 14:
        return 2012277125453
    elif quarter == 15:
        return 2013011073258
    elif quarter == 16:
        return 2013098041711
    elif quarter == 17:
        return 2013131215648

def get_mod_out(channel):
    tab = Table.read('mod_out.csv')
    index = np.where(tab['Channel']==channel)
    mod, out = tab['Mod'][index], tab['Out'][index]
    return int(mod), int(out)


def stitch_lc_list(lcs,flux_type='PDCSAP_FLUX'):
    quarters = np.array([])
    channels = np.array([])
    for j, lci in enumerate(lcs):
        lci = lci.get_lightcurve(flux_type).remove_nans()
        lci = lci[lci.quality==0]
        lcs[j] = lci.normalize()
        quarters = np.append(quarters,lci.quarter*np.ones_like(lci.flux))
        channels = np.append(channels,lci.channel*np.ones_like(lci.flux))

    lc = copy.copy(lcs[0])
    for lci in lcs[1:]:
        lc = lc.append(copy.copy(lci))

    args = np.argsort(lc.time)
    lc = lc[args]

    quarters = quarters[args]
    channels = channels[args]

    lc.quarter = quarters.astype('int')
    lc.channel = channels.astype('int')

    return lc

def censor_quarters(lc):
    noisy = np.ones_like(lci.flux)
    for quarter in np.unique(lc.quarter):
        m = lc.quarter==quarter
        noisy[m] = np.nanstd(lc.flux[m])

    bad = noisy>(5*np.nanmin(noisy))
    return lc[~bad]

def match_cadences(cbvcads,lccads):
    indices =np.array([1 if j in lccads else 0 for j in cbvcads])
    return np.where(indices==1)[0]

def correct_quarter(lc,quarter):
    fname = find_cbv(quarter)
    cbvfile = fitsio.FITS(fname)
    m = (lc.quarter== quarter)
    channel = lc.channel[0]

    cads = match_cadences(cbvfile[channel]['CADENCENO'][:],lc[m].cadenceno)
    basis = np.zeros((lc[m].flux.shape[0],16))

    for j in range(16):
        try:
            basis[:,j] = cbvfile[channel]['VECTOR_%d'% (j+1)][cads]
        except:
            print('Missing CBV',j)

    corrected_flux, weights = cbv.fixed_nb(lc[m].flux, basis, doPlot = False)

    return corrected_flux

def correct_all_quarters(lc):
    lc2 = copy.copy(lc)
    lc2.trposi = np.zeros_like(lc2.flux)
    for qq in np.unique(lc2.quarter.astype('int')):
        m = lc2.quarter==qq
        corrflux = correct_quarter(lc2,qq)
        lc2.trposi[m] = lc.flux[m] - corrflux + np.nanmedian(corrflux)
    return lc2

def correct_all(lc,ff):
    niter = lc.niter

    design = make_design_matrix(lc,ff)
    weights, residuals, rank, s = np.linalg.lstsq(design.T,lc.flux)

    model = np.dot(weights,design)
    model_time = np.dot(weights[:(niter*2)],design[:(niter*2),:])
    model_pos = np.dot(weights[(niter*2):],design[(niter*2):,:])

    model_time = model_time-np.nanmedian(model_time)+1
    model_pos = model_pos-np.nanmedian(model_pos)+1

    lc2 = lc.copy()
    lc2.trposi = model_pos
    lc2.trtime = model_time
    return lc2


def sine_renormalize(lc,min_period=4./24.,max_period=30.):
    powers = []

    # get an overall mean model
    bestfreq, power, falarm = get_best_freq(lc,min_period=0.5,max_period=10)
    ytest_mean = LombScargle(lc.time, lc.flux-1, lc.flux_err).model(lc.time, bestfreq)
    current = (np.max(ytest_mean)-np.min(ytest_mean))/2.

    dummy = copy.copy(lc)

    for j, q in enumerate(np.unique(lc.quarter)):
        m = (lc.quarter == q)

        ytest = LombScargle(lc.time[m], lc.flux[m]-1., lc.flux_err[m]).model(lc.time[m], bestfreq)
        power = (np.max(ytest)-np.min(ytest))/2.#/np.median(dummy.flux[m])
        dummy.flux[m] = 1.+(dummy.flux[m]-1.)/(power/current)

        powers.append(power)

    return dummy, powers

def get_best_freq(lc,min_period=4./24., max_period=30.):
    lc2 = copy.copy(lc)

    frequency, power = LombScargle(lc2.time, lc2.flux, lc2.flux_err).autopower(minimum_frequency=1./max_period,
                                                                                         maximum_frequency=1./min_period,
                                                                                         samples_per_peak=3)
    best_freq = frequency[np.argmax(power)]

    # refine
    for j in range(3):
        ls = LombScargle(lc2.time, lc2.flux, lc2.flux_err)
        frequency, power = ls.autopower(minimum_frequency=best_freq*(1-1e-3*(3-j)),maximum_frequency=best_freq*(1+1e-3*(3-j)),
                                                                                         samples_per_peak=500)
        best_freq = frequency[np.argmax(power)]

    falarm = ls.false_alarm_probability(np.max(power))

    return best_freq, ls.power(best_freq,normalization='psd'), falarm


def iterative_sine_fit(lc,nmax,min_period=4./24., max_period=30.):
    ff, pp, noise = [], [], []
    y_fit = 0
    lc2 = copy.copy(lc)

    for j in range(nmax):
        best_freq, maxpower, falarm = get_best_freq(lc2,min_period=min_period,max_period=max_period)

        ff.append(best_freq)
        pp.append(maxpower)
        y_fit += LombScargle(lc2.time, lc2.flux-1, lc2.flux_err).model(lc2.time, best_freq)
        lc2.flux = lc.flux - y_fit
        noise.append(lc2.estimate_cdpp())

    lc2.trtime = y_fit + np.nanmedian(lc.flux)
    lc2.flux = lc.flux
    try:
        lc2.corr_flux = lc2.corr_flux - lc2.trtime + np.nanmedian(lc2.trtime)
    except:
        lc2.corr_flux = lc2.flux - lc2.trtime + np.nanmedian(lc2.trtime)

    return lc2, np.array(ff), np.array(pp), np.array(noise)


def auto_sine_fit(lc,prob_max=1e-10, maxiter=60,min_period=4./24., max_period=30.):
    ff, pp, snr, noise = [], [], [], []
    y_fit = 0
    lc2 = copy.copy(lc)

    i = 0
    falarm = 0.0
    tq = tqdm(total=maxiter,desc='sine waves')

    components = []
    while falarm <= prob_max:
        best_freq, maxpower, falarm = get_best_freq(lc2,min_period=min_period,max_period=max_period)
        ff.append(best_freq)
        pp.append(maxpower)
        snr.append(falarm)
        new_fit = LombScargle(lc2.time, lc2.flux-1, lc2.flux_err).model(lc2.time, best_freq)
        components.append(new_fit)
        y_fit += new_fit
        lc2.flux = lc.flux - y_fit
        noise.append(lc2.estimate_cdpp())

        i += 1
        tq.update(1)
        if i > maxiter:
            break
    tq.close()


    lc2.trtime = y_fit + np.nanmedian(lc.flux)
    lc2.flux = lc.flux
    try:
        lc2.corr_flux = lc2.corr_flux - lc2.trtime + np.nanmedian(lc2.trtime)
    except:
        lc2.corr_flux = lc2.flux - lc2.trtime + np.nanmedian(lc2.trtime)

    lc2.niter = i

    return lc2, np.array(ff), np.array(pp), np.array(noise), np.array(snr), i, components

def renorm_sde(period,sde,niter=3,order=2,nsig=2.5):
    '''
    In Pope et al 2016 we noted that the BLS has a slope towards longer periods.
    To fix this we used the full ensemble of stars to build a binne median SDE as a function of period.
    With our smaller numbers here we just aggressively sigma-clip and fit a quadratic, and it seems to do basically ok.
    '''
    trend = gaussfilt(sde,20)
    sde = copy.copy(sde)
    outliers = np.zeros(len(sde))

    for j in range(niter):
        outliers = np.abs(sde-trend)>(nsig*np.std(sde-trend))
        sde[outliers] += trend[outliers]
        trend = np.poly1d(np.polyfit(period,sde,order))(period)

    return trend

def estimate_cdpp(lc, flux, transit_duration=13, savgol_window=101,
                  savgol_polyorder=2, sigma_clip=5.):
    """
    Copied from lightkurve: https://github.com/KeplerGO/lightkurve/

    Estimate the CDPP noise metric using the Savitzky-Golay (SG) method.
    A common estimate of the noise in a lightcurve is the scatter that
    remains after all long term trends have been removed. This is the idea
    behind the Combined Differential Photometric Precision (CDPP) metric.
    The official Kepler Pipeline computes this metric using a wavelet-based
    algorithm to calculate the signal-to-noise of the specific waveform of
    transits of various durations. In this implementation, we use the
    simpler "sgCDPP proxy algorithm" discussed by Gilliland et al
    (2011ApJS..197....6G) and Van Cleve et al (2016PASP..128g5002V).
    The steps of this algorithm are:
        1. Remove low frequency signals using a Savitzky-Golay filter with
           window length `savgol_window` and polynomial order `savgol_polyorder`.
        2. Remove outliers by rejecting data points which are separated from
           the mean by `sigma_clip` times the standard deviation.
        3. Compute the standard deviation of a running mean with
           a configurable window length equal to `transit_duration`.
    We use a running mean (as opposed to block averaging) to strongly
    attenuate the signal above 1/transit_duration whilst retaining
    the original frequency sampling.  Block averaging would set the Nyquist
    limit to 1/transit_duration.
    Parameters
    ----------
    transit_duration : int, optional
        The transit duration in units of number of cadences. This is the
        length of the window used to compute the running mean. The default
        is 13, which corresponds to a 6.5 hour transit in data sampled at
        30-min cadence.
    savgol_window : int, optional
        Width of Savitsky-Golay filter in cadences (odd number).
        Default value 101 (2.0 days in Kepler Long Cadence mode).
    savgol_polyorder : int, optional
        Polynomial order of the Savitsky-Golay filter.
        The recommended value is 2.
    sigma_clip : float, optional
        The number of standard deviations to use for clipping outliers.
        The default is 5.
    Returns
    -------
    cdpp : float
        Savitzky-Golay CDPP noise metric in units parts-per-million (ppm).
    Notes
    -----
    This implementation is adapted from the Matlab version used by
    Jeff van Cleve but lacks the normalization factor used there:
    svn+ssh://murzim/repo/so/trunk/Develop/jvc/common/compute_SG_noise.m
    """
    lc2 = copy.copy(lc)
    lc2.flux = flux
    return lc2.estimate_cdpp()

### this stuff is all cribbed from k2ps

str_to_dt = lambda s: [tuple(t.strip().split()) for t in s.split(',')]
dt_lcinfo    = str_to_dt('epic u8, flux_median f8, Kp f8, flux_std f8, lnlike_constant f8, type a8,'
                         'acor_raw f8, acor_corr f8, acor_trp f8, acor_trt f8')
dt_blsresult = str_to_dt('sde f8, bls_zero_epoch f8, bls_period f8, bls_duration f8, bls_depth f8,'
                         'bls_radius_ratio f8, ntr u4')

from scipy.constants import G
from exotk.utils.orbits import d_s
from seaborn import despine
from matplotlib.pyplot import setp, subplots


def rho_from_pas(period,a):
    return 1e-3*(3*np.pi)/G * a**3 * (period*d_s)**-2


def do_all(kic,auto=True,renormalize=False,planet_p_range=(1.,360.),star_p_range=(1./24.,30.),niter=60,figtype='png',outdir='./'):
    tic = clock()
    print('Loading light curve for KIC %d...' % kic)

    try:
        fnames = glob.glob('../data/lcs/*%s*llc.fits' % kic)
        assert fnames,'No files'
        lcs = []

        for fname in fnames:
            lcs.append(lightkurve.open(fname))
        print('Already downloaded %s' % kic)
    except:
        lcs = lightkurve.search_lightcurvefile(kic,cadence='long').download_all()
        print('Downloaded %s' % kic)

    lc = stitch_lc_list(lcs)


    print('Loaded!')
    min_period, max_period = star_p_range

    if renormalize:
        print('Renormalizing...')
        lc2, powers = sine_renormalize(lc,min_period=min_period, max_period=max_period)
        print('Renormalized!')
    else:
        lc2 = lc

    print('Running CLEAN')
    if auto == True:
        lc3, ff, pp, noise, snrs, niter, components = auto_sine_fit(lc2,prob_max = 1e-20, maxiter=200,min_period=min_period,max_period=max_period)
        print('Subtracted %d sine waves' % niter)
    else:
        lc3, ff, pp, noise = iterative_sine_fit(lc2, niter,min_period=min_period, max_period=max_period)
    print('Cleaned!')

    print('Correcting with CBVs...')
    lc4 = correct_all(lc3,ff)
    print('Corrected with CBVs!')

    print('Clipping outliers')
    dummy = lc4.copy()
    model = lc4.trposi + lc4.trtime
    dummy.flux -= model
    dummy, mask = dummy.remove_outliers(return_mask=True,sigma_upper=4,sigma_lower=5)
    # mask += (lc5.quarter>16)
    lc6 = lc[~mask].copy()
    lc6.quarter = lc6.quarter[~mask]
    print('Clipped %d outliers' % np.sum(mask))

    print('Running CLEAN again')
    if auto == True:
        lc3, ff, pp, noise, snrs, niter, components = auto_sine_fit(lc6,prob_max = 1e-20, maxiter=200,min_period=min_period,max_period=max_period)
        print('Subtracted %d sine waves' % niter)
    else:
        lc3, ff, pp, noise = iterative_sine_fit(lc6, niter,min_period=min_period, max_period=max_period)
    print('Cleaned!')

    print('Correcting with CBVs...')
    lc4 = correct_all(lc3,ff)
    print('Corrected with CBVs!')

    lc4.pp = pp
    lc4.ff = ff
    lc4.star_p_range = star_p_range
    lc4.niter = niter

    print('Doing Transit Search...')
    ts = BasicSearch(lc4,period_range=planet_p_range)

    ts.run_bls()

    fit = ts.fit_transit()

    print('Transit search done!')
    toc = clock()

    print('Time elapsed: %.2f s' % (toc - tic))

    plot_all(ts,save_file='%splots_%d.%s' % (outdir,kic,figtype))

    f = open('%sdata_%d.txt' % (outdir,kic),'w')
    f.write('%d %f %f %f %f %f %d' % (kic, ts.p, ts.t0, ts.bsde, ts.impact, ts.depth, ts.niter))
    f.close()
    print('Done\n')

def make_design_matrix(llc,frequencies,offsets=True):
    cols = []
    t = llc.time
    channel = llc.channel[0]

    for frequency in tqdm(frequencies,desc='frequencies'):
        cols.append(np.sin(2 * np.pi * frequency * t))
        cols.append(np.cos(2 * np.pi * frequency * t))

    quarters = llc.quarter

    for quarter in tqdm(np.unique(quarters),desc='quarters'):
        fname = find_cbv(quarter)
        cbvfile = fitsio.FITS(fname)
        m = (llc.quarter== quarter)
        cads = match_cadences(cbvfile[channel]['CADENCENO'][:],llc[m].cadenceno)

        for j in range(16):
            padded = np.zeros_like(t)
            padded[m] = cbvfile[channel]['VECTOR_%d'% (j+1)][cads]
            padded = (padded-padded.min())/(padded.max()-padded.min())*2-1. # normalize
            cols.append(padded)

        if offsets:
            padded = np.zeros_like(t)
            padded[m] = 1.
            padded = (padded-padded.min())/(padded.max()-padded.min())*2-1. # normalize
            cols.append(padded)
    if not offsets:
        cols.append(np.ones_like(t))

    return np.vstack(cols)

###-------------------------------------------------------------------###

###-------------------------------------------------------------------###


class BasicSearch(object):

    def __init__(self, d, inject=False,star_p_range=(1./24.,30.),**kwargs):
        ## Keyword arguments
        ## -----------------
        self.d = d.copy()

        self.nbin = kwargs.get('nbin', 2000)
        self.qmin = kwargs.get('qmin', 0.001)
        self.qmax = kwargs.get('qmax', 0.01)
        self.nf   = kwargs.get('nfreq', 45000)
        self.exclude_regions = kwargs.get('exclude_regions', [])
        try:
            self.pp = d.pp
            self.ff = d.ff
            self.niter = d.niter
        except:
            self.pp = []
            self.ff = []
            self.niter = np.nan

        ## Read in the data
        ## ----------------
        m  = np.isfinite(d.flux) & np.isfinite(d.time) # no mflags
        m &= ~binary_dilation((d.quality & 2**20) != 0)

        for emin,emax in self.exclude_regions:
            m[(d.time > emin) & (d.time < emax)] = 0

        try:
            self.Kp = d.Kp
        except:
            self.Kp = 12
        self.star_p_range = d.star_p_range
        self.epic   = d.targetid
        self.time   = d.time[m]
        self.flux   = (d.flux[m]
                       - d.trtime[m] + np.nanmedian(d.trtime[m])
                       - d.trposi[m] + np.nanmedian(d.trposi[m]))
        self.mflux   = np.nanmedian(self.flux)
        self.flux   /= self.mflux
        self.flux_e  = d.flux_err[m] / abs(self.mflux)

        self.flux_r  = d.flux[m] / self.mflux
        self.trtime = d.trtime[m] / self.mflux
        self.trposi = d.trposi[m] / self.mflux

        self.d.flux = self.flux

        ## Initialise BLS
        ## --------------
        self.period_range = kwargs.get('period_range', (1.,40.))
        if self.nbin > np.size(self.flux):
            self.nbin = int(np.size(self.flux)/3)

        try:
            ar,ac,ap,at = acor(self.flux_r)[0], acor(self.flux)[0], acor(self.trposi)[0], acor(self.trtime)[0]
        except RuntimeError:
            ar,ac,ap,at = np.nan,np.nan,np.nan,np.nan
        self.lcinfo = np.array((self.epic, self.mflux, self.Kp, self.flux.std(), np.nan, np.nan, ar, ac, ap, at), dtype=dt_lcinfo)

        self._rbls = None
        self._rtrf = None
        self._rvar = None
        self._rtoe = None
        self._rpol = None
        self._recl = None

        ## Transit fit pv [k u t0 p a i]
        self._pv_bls = None
        self._pv_trf = None

        self.period = None
        self.zero_epoch = None
        self.duration = 0.2

        print('Initialized search')

    def run_bls(self):
        print('Running BLS')
        model = BoxLeastSquares(self.time, self.flux, dy=0.01)
        period_grid = np.exp(np.linspace(np.log(self.period_range[0]), np.log(self.period_range[1]), 50000))
        b = model.power(period_grid,self.duration)

        new_sde = b.power-renorm_sde(b.period,b.power)
        bper = b.period[np.argmax(new_sde)]
        bsde = np.max(new_sde)

        print('Rerunning on finer grid')
        period_grid_new = np.exp(np.linspace(np.log(bper*1.02), np.log(bper*1.02), 50000))
        b = model.power(period_grid,0.2)
        new_sde_fine = b.power-renorm_sde(b.period,b.power)

        bper = b.period[np.argmax(new_sde_fine)]
        self.bsde = np.max(new_sde_fine)

        self.period_grid = period_grid
        self.period = bper
        self.zero_epoch = b.transit_time[np.argmax(new_sde)]+self.period*(self.time.min()//self.period)
        self.depth = b.depth[np.argmax(new_sde)]
        self.sde = new_sde

    def fit_transit(self,mask=None, start=None):
        # from Dan: https://exoplanet.dfm.io/en/stable/tutorials/tess/#transit-search

        x = self.time
        y = self.flux-1.
        yerr = self.flux_e

        bls_period = self.period
        bls_depth = self.depth
        bls_t0 = self.zero_epoch
        print('Fitting transit')
        if mask is None:
            mask = np.ones(len(x), dtype=bool)
        with pm.Model() as model:

            texp = 0.5/24.

            # Parameters for the stellar properties
            mean = pm.Normal("mean", mu=0.0, sd=10.0)
            u_star = xo.distributions.QuadLimbDark("u_star")

            # Stellar parameters from Sowicka et al (2017) for KIC 8197761
            M_star_sowicka = 1.384, 0.281
            R_star_sowicka = 1.717, 0.858
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)
            m_star = BoundedNormal("m_star", mu=M_star_sowicka[0], sd=M_star_sowicka[1])
            r_star = BoundedNormal("r_star", mu=R_star_sowicka[0], sd=R_star_sowicka[1])

            # Orbital parameters for the planet
            logP = pm.Normal("logP", mu=np.log(bls_period), sd=1)
            t0 = pm.Normal("t0", mu=bls_t0, sd=1)
            b = pm.Flat("b", transform=pm.distributions.transforms.logodds, testval=0.5)
            logr = pm.Normal("logr", sd=1.0,
                             mu=0.5*np.log(1e-3*np.array(bls_depth))+np.log(R_star_sowicka[0]))
            r_pl = pm.Deterministic("r_pl", tt.exp(logr))
            ror = pm.Deterministic("ror", r_pl / r_star)

            # This is the eccentricity prior from Kipping (2013):
            # https://arxiv.org/abs/1306.4982
            BoundedBeta = pm.Bound(pm.Beta, lower=0, upper=1-1e-5)
            ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, testval=0.1)
            omega = xo.distributions.Angle("omega")

            # Transit jitter & GP parameters
    #         logs2 = pm.Normal("logs2", mu=np.log(np.var(y[mask])), sd=10)
    #         logw0_guess = np.log(2*np.pi/10)
    #         logw0 = pm.Normal("logw0", mu=logw0_guess, sd=10)

            # We'll parameterize using the maximum power (S_0 * w_0^4) instead of
            # S_0 directly because this removes some of the degeneracies between
            # S_0 and omega_0
    #         logpower = pm.Normal("logpower",
    #                              mu=np.log(np.var(y[mask]))+4*logw0_guess,
    #                              sd=10)
    #         logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)

            # Tracking planet parameters
            period = pm.Deterministic("period", tt.exp(logP))

            # Orbit model
            orbit = xo.orbits.KeplerianOrbit(
                r_star=r_star, m_star=m_star,
                period=period, t0=t0, b=b,
                ecc=ecc, omega=omega)

            # Compute the model light curve using starry
            light_curves = xo.StarryLightCurve(u_star).get_light_curve(
                orbit=orbit, r=r_pl, t=x[mask], texp=texp)*1e3
            light_curve = pm.math.sum(light_curves, axis=-1) + mean
            pm.Deterministic("light_curves", light_curves)

            # GP model for the light curve
    #         kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))
    #         gp = xo.gp.GP(kernel, x[mask], tt.exp(logs2) + tt.zeros(mask.sum()), J=2)
    #         pm.Potential("transit_obs", gp.log_likelihood(y[mask] - light_curve))
    #         pm.Deterministic("gp_pred", gp.predict())

            # The likelihood function assuming known Gaussian uncertainty
            pm.Normal("obs", mu=light_curve, sd=yerr, observed=y)
            # Fit for the maximum a posteriori parameters, I've found that I can get
            # a better solution by trying different combinations of parameters in turn
            if start is None:
                start = model.test_point

            to_optimize = [[logr],[b],[logP,t0],[u_star],[logr],[b],[ecc,omega],[mean]]

            map_soln = xo.optimize(start=start, vars=to_optimize[0])

            for obj in tqdm(to_optimize[1:]):
                map_soln = xo.optimize(start=map_soln, vars=obj)
            map_soln = xo.optimize(start=map_soln)

        self.p = np.exp(map_soln['logP'])
        self.t0 = map_soln['t0']
        self.best_lc = map_soln['light_curves'].T[0]+1
        self.impact = map_soln['b']
        self.ror = map_soln['ror']
        self.depth = 1-self.best_lc.min()
        return model, map_soln

    def plot_info(self, ax):
        t0,p,tdep,rrat = self.t0,self.p, self.depth, 0
        # a = res.trf_semi_major_axis[0]
        ax.text(0.0,1.0, 'KIC {:d}'.format(self.epic), size=12, weight='bold', va='top', transform=ax.transAxes)
        ax.text(0.0,0.83, ('SDE\n'
                          'Sines\n'
                          'Zero epoch\n'
                          'Period [d]\n'
                          'Transit depth\n'
                          'Radius ratio\n'
                          'Impact parameter'), size=9, va='top')
        ax.text(0.97,0.83, ('{:9.3f}\n{:d}\n{:9.3f}\n{:9.3f}\n{:9.5f}\n'
                           '{:9.4f}\n{:9.3f}').format(np.max(self.sde),self.niter,t0,p,tdep,np.sqrt(tdep),
                                                                        self.impact),
                size=9, va='top', ha='right')
        despine(ax=ax, left=True, bottom=True)
        setp(ax, xticks=[], yticks=[])

    def plot_lc_pos(self, ax=None):
        ax.plot(self.time, self.flux_r-self.trtime+np.nanmedian(self.trtime), '.')
        ax.plot([],[])
        # ax.plot(self.time, self.trtime+2*(np.percentile(self.flux_r, [99])[0]-1), lw=1)
        ax.plot(self.time, self.trposi, lw=1)
        # ax.plot(self.time, self.flux+1.1*(self.flux_r.min()-1), lw=1)
        [ax.axvline(self.zero_epoch+i*self.period, alpha=0.25, ls='--', lw=1) for i in range(35)]
        setp(ax,xlim=self.time[[0,-1]])#, xlabel='Time', ylabel='Normalised flux')

    def plot_lc_time(self, ax=None):
        # ax.plot(self.time, self.flux_r, lw=1)
        # ax.plot(self.time, self.trposi+4*(np.percentile(self.flux_r, [99])[0]-1), lw=1)
        ax.plot(self.time, self.flux_r-self.trposi+np.nanmedian(self.trposi), '.')
        ax.plot([],[])
        ax.plot(self.time, self.trtime, lw=1)
        [ax.axvline(self.zero_epoch+i*self.period, alpha=0.25, ls='--', lw=1) for i in range(35)]
        setp(ax,xlim=self.time[[0,-1]])#, xlabel='Time', ylabel='Normalised flux')

    def plot_lc_white(self, ax=None):
        # ax.plot(self.time, self.flux_r, lw=1)
        # ax.plot(self.time, self.trtime, lw=1)
        # ax.plot(self.time, self.trposi+4*(np.percentile(self.flux_r, [99])[0]-1), lw=1)
        ax.plot(self.time, self.flux_r-self.trposi+np.nanmedian(self.trposi)
            -self.trtime+np.nanmedian(self.trtime), '.')
        [ax.axvline(self.zero_epoch+i*self.period, alpha=0.25, ls='--', lw=1) for i in range(35)]
        setp(ax,xlim=self.time[[0,-1]], xlabel='Time', ylabel='Normalised flux')

    def plot_pgram(self,ax):
        min_period, max_period = self.star_p_range

        frequency, power = LombScargle(self.time, self.flux_r, self.flux_e).autopower(minimum_frequency=1./max_period,maximum_frequency=1./min_period,
                                                                                   samples_per_peak=10,normalization='psd')
        ax.plot(frequency,power**0.5,color=colours[0])
        ax.set_yscale('log')
        ax.set_xlim(1./max_period,1/min_period)
        try:
            ax.scatter(self.ff,self.pp**0.5,c=colours[1])
        except:
            pass
        frequency2, power2 = LombScargle(self.time, self.flux, self.flux_e).autopower(minimum_frequency=1./max_period,maximum_frequency=1./min_period,
                                                                                     samples_per_peak=10,normalization='psd')
        ax.plot(frequency2,power2**0.5,color=colours[2])
        ax.set_xlabel('c/d')

    def plot_sde(self, ax=None):
        ax.plot(self.period_grid, self.sde, drawstyle='steps-mid')
        ax.axvline(self.period, alpha=0.25, ls='--', lw=1)
        setp(ax,xlim=self.period_grid[[0,-1]], xlabel='Period [d]', ylabel='SDE', ylim=(self.sde.min()-1.5,self.sde.max()+1.5))
        [ax.axhline(i, c='k', ls='--', alpha=0.5) for i in [0,5,10]]
        [ax.text(self.period_grid.max()-1,i-0.5,i, va='top', ha='right', size=7) for i in [5,10]]
        ax.text(0.5, 0.88, 'BLS search', va='top', ha='center', size=8, transform=ax.transAxes)
        setp(ax.get_yticklabels(), visible=False)

    def plot_transit_fit(self, ax=None, nbin=40):
        period, zero_epoch = self.p, self.t0

        duration = 0.5

        flux_m = self.best_lc
        phase = (fold(self.time, period, zero_epoch, 0.5, normalize=False) -period)

        sids = np.argsort(phase)
        flux_m = flux_m[sids]
        phase = phase[sids]
        folded = self.d.fold(period,t0=zero_epoch)

        binned = folded.bin(15)
        bpd,bfd = binned.time, binned.flux
        ax.plot(bpd, bfd, marker='o', ms=2,color=colours[1])
        ax.plot(folded.time, flux_m, 'k')

        ax.axhline(flux_m.min(), alpha=0.25, ls='--')

        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
    #     [ax.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur]
        fluxrange =bpd.max()-bpd.min()
        setp(ax, xlim=(-0.05,0.05), ylim=[bpd.min()-0.05*fluxrange,bpd.max()+0.05*fluxrange],
         xlabel='Phase', ylabel='Normalised flux')
        setp(ax.get_yticklabels(), visible=False)

    def plot_folded(self, ax=None, nbin=40):
        period, zero_epoch = self.p, self.t0
        depth = 1-np.min(self.best_lc)


        duration = 0.5

        flux_m = self.best_lc
        phase = (fold(self.time, period, zero_epoch, 0.5, normalize=False) -period)

        sids = np.argsort(phase)
        flux_m = flux_m[sids]
        phase = phase[sids]
        folded = self.d.fold(period,t0=zero_epoch)

        binned = folded.bin(15)
        bpd,bfd = binned.time, binned.flux
        ax.plot(bpd, bfd, marker='o', ms=2,color=colours[1])
        ax.plot(folded.time, flux_m, 'k')

        ax.axhline(flux_m.min(), alpha=0.25, ls='--')

        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
    #     [ax.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur]
        ax.set_ylim(1-1.25*depth,1+0.25*depth)
        ax.set_xlim(folded.time.min(),folded.time.max())

        setp(ax,xlabel='Phase', ylabel='Normalised flux')
        setp(ax.get_yticklabels(), visible=False)

    def plot_eclipse(self, ax=None, nbin=40):
        period, zero_epoch = self.p, self.t0

        duration = 0.5

        flux_m = self.best_lc
        phase = (fold(self.time, period, zero_epoch+period/2., 0.5, normalize=False) - period)

        sids = np.argsort(phase)
        flux_m = flux_m[sids]
        phase = phase[sids]
        folded = self.d.fold(period,t0=zero_epoch+period/2.)

        binned = folded.bin(15)
        bpd,bfd = binned.time, binned.flux
        ax.plot(bpd, bfd, marker='o', ms=2,color=colours[1])
        ax.plot(folded.time, flux_m, 'k')

        ax.axhline(flux_m.min(), alpha=0.25, ls='--')

        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        ax.axvline(0, alpha=0.25, ls='--', lw=1)
    #     [ax.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur]
        fluxrange =bpd.max()-bpd.min()
        setp(ax, xlim=(-0.2,0.2), xlabel='Phase', ylabel='Normalised flux')
        setp(ax.get_yticklabels(), visible=False)


    def plot_transits(self, ax=None):
        period, t0 = self.p, self.t0
        offset = 1.5*self.depth
        twodur = period*1

        ntransits = np.min([8,(self.time.max()-self.time.min())/self.p]).astype('int')
        k = 0
        j = 0

        while k < ntransits:
            transit_mask = np.abs(self.time-(self.t0+(period*j))) <= 0.2
            if np.sum(transit_mask) == 0:
                j += 1
                continue
            tt, ff = self.time[transit_mask], self.flux[transit_mask]+(j*offset)
            tt -= (self.t0+(period*j))
            ax.plot(tt,ff,'-o')
            j += 1
            k += 1
            if j > 100:
                break
        ax.axvline(0,linestyle='--',alpha=0.5)

    def plot_fit_and_eo(self, ax=None, nbin=40):
        nbin = nbin or self.nbin
        period, zero_epoch = self.p, self.t0
        # hdur = 24*np.array([-0.25,0.25])
        duration = 0.5
        depth = 1-np.min(self.best_lc)

        self.plot_transit_fit(ax[0])

        folded = self.d.fold(period*2,t0=zero_epoch+period/2.)
        folded.time+=0.5
        odd = folded.time>0.5
        even = ~odd

        folded.time *= 2.
        odd_lc = folded[odd].bin(20)
        even_lc = folded[even].bin(20)
        even_lc.time -= 0.5
        odd_lc.time -= 1.5

        for lcp in [odd_lc,even_lc]:
            ax[1].plot(lcp.time,lcp.flux,'-o')

        [a.axvline(0, alpha=0.25, ls='--', lw=1) for a in ax]
        # [[a.axvline(hd, alpha=0.25, ls='-', lw=1) for hd in hdur] for a in ax]
        ax[1].set_xlim(-0.05,0.05)
        ax[1].set_ylim(1-1.25*depth,1+0.25*depth)
        ax[0].set_ylim(1-1.25*depth,1+0.25*depth)

        ax[1].axhline(1-depth,linestyle='--',alpha=0.25)

        # setp(ax[1],xlim=[0,1], xlabel='Phase')
        setp(ax[1].get_yticklabels(), visible=False)
        ax[1].get_yaxis().get_major_formatter().set_useOffset(False)


def plot_all(ts,save_file=None):
    PW,PH = 8.27, 11.69
    rc('axes', labelsize=7, titlesize=8)
    rc('font', size=6)
    rc('xtick', labelsize=7)
    rc('ytick', labelsize=7)
    rc('lines', linewidth=1)
    fig = plt.figure(figsize=(PW,PH))
    gs1 = GridSpec(3,3)
    gs1.update(top=0.98, bottom = 2/3.*1.03,hspace=0.07,left=0.07,right=0.96)
    gs = GridSpec(4,3)
    gs.update(top=2/3.*0.96,bottom=0.04,hspace=0.35,left=0.07,right=0.96)

    ax_lcpos = subplot(gs1[0,:])
    ax_lctime = subplot(gs1[1,:],sharex=ax_lcpos)
    ax_lcwhite = subplot(gs1[2,:],sharex=ax_lcpos)
    ax_lcfold = subplot(gs[2,1:])
    # ax_lnlike = subplot(gs[1,2])
    ax_lcoe   = subplot(gs[0,1]),subplot(gs[0,2])
    ax_sde    = subplot(gs[3,1:])
    ax_transits = subplot(gs[1:,0])
    ax_info = subplot(gs[0,0])
    ax_ec = subplot(gs[1,1:])
    # axes = [ax_lctime,ax_lcpos,ax_lcwhite, ax_lcfold, ax_lcoe[0], ax_lcoe[1], ax_sde, ax_transits, ax_info, ax_ec]

    ts.plot_lc_pos(ax_lcpos)
    ts.plot_lc_time(ax_lctime)
    ts.plot_lc_white(ax_lcwhite)
    ts.plot_folded(ax_ec)
    ts.plot_pgram(ax_lcfold) # replace with periodogram - to do!
    ts.plot_fit_and_eo(ax_lcoe)
    ts.plot_info(ax_info)
    # ts.plot_lnlike(ax_lnlike)
    ts.plot_sde(ax_sde)
    ts.plot_transits(ax_transits)
    # ax_lnlike.set_title('Ln likelihood per orbit')
    ax_transits.set_title('Individual transits')
    ax_ec.set_title('Folded Light Curve')
    ax_lcoe[0].set_title('Folded transit and model')
    ax_lcoe[1].set_title('Even and odd transits')
    ax_lcfold.set_title('Pulsation Periodogram')

    if save_file is not None:
        plt.savefig(save_file)
