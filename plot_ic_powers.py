"""Scripts to plot the power spectra from our simulations"""
import os.path
import glob
import numpy as np
import scipy.interpolate
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

datadir = os.path.expanduser("~/data/rescale_ICtest")
savedir = "icplots/"
sims = ["rescale_omegab","single_norescale","rescale"]
lss = {"rescale_omegab":"-.", "single_norescale":"--", "rescale":"-"}

def get_camb_power(matpow):
    """Plot the power spectrum from CAMB
    (or anything else where no changes are needed)"""
    data = np.loadtxt(matpow)
    kk = data[:,0]
    ii = np.where(kk > 0.)
    return (kk[ii], data[:,1][ii])

def _get_pk(scale, ss):
    """Get the matter power spectrum"""
    sdir = os.path.join(os.path.join(datadir, ss),"output")
    matpow = glob.glob(os.path.join(sdir,"powerspectrum-"+str(scale)+"*.txt"))
    if np.size(matpow) == 0:
        return ([],[])
    matpow = matpow[0]
    return get_camb_power(matpow)

def plot_single_redshift(scale):
    """Plot all the simulations at a single redshift"""
    for ss in sims:
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        plt.loglog(k, pk,ls=lss[ss], label=ss)
    cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
    camb = os.path.join(cambdir,"ics_matterpow_"+str(int(1/scale-1))+".dat")
    (k_camb, pk_camb) = get_camb_power(camb)
    rebinned=scipy.interpolate.interpolate.interp1d(k_camb,pk_camb)
    plt.semilogx(k, rebinned(k),ls=":", label="CAMB")
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks-"+str(scale)+".pdf"))
    plt.clf()

def plot_single_redshift_rel_camb(scale):
    """Plot all the simulations at a single redshift"""
    for ss in sims:
        cambdir = os.path.join(os.path.join(datadir, ss),"camb_linear")
        camb = os.path.join(cambdir,"ics_matterpow_"+str(int(1/scale-1))+".dat")
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        (k_camb, pk_camb) = get_camb_power(camb)
        rebinned=scipy.interpolate.interpolate.interp1d(k_camb,pk_camb)
        plt.semilogx(k, pk/rebinned(k),ls=lss[ss], label=ss)
    plt.ylim(0.85,1.3)
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_camb-"+str(scale)+".pdf"))
    plt.clf()

def plot_single_redshift_rel_one(scale, divisor=1):
    """Plot all the simulations at a single redshift"""
    (k_div, pk_div) = _get_pk(scale, sims[divisor])
    for ss in sims:
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        plt.semilogx(k, pk/pk_div,ls=lss[ss], label=ss)
    plt.ylim(0.995,1.01)
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_rel-"+str(scale)+".pdf"))
    plt.clf()

if __name__ == "__main__":
    for sc in (0.3, 1):
        plot_single_redshift_rel_one(sc)
        plot_single_redshift_rel_camb(sc)
        plot_single_redshift(sc)
