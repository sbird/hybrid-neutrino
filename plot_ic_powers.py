"""Scripts to plot the power spectra from our simulations"""
import os.path
import glob
import re
import numpy as np
import scipy.interpolate
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

datadir = os.path.expanduser("~/data/rescale_ICtest")
savedir = "icplots/"
sims = ["rescale_omegab","single_norescale","rescale"]
lss = {"rescale_omegab":"-.", "single_norescale":"--", "rescale":"-"}
labels = {"rescale_omegab":"radrescale", "single_norescale":"norescale", "rescale":"noradrescale"}
#colors = {"b300p512nu0.4p": '#d62728', "b300p512nu0.4a":'#1f77b4', "b300p512nu0.4hyb":'#2ca02c',"b300p512nu0.4hyb-single":'#2ca02c',"b300p512nu0.4hyb-vcrit":'#bcbd22',"b300p512nu0.4hyb-nutime": '#ff7f0e',"b300p512nu0.4hyb-all": '#e377c2',"b300p512nu0.06a":'#1f77b4'}

plt.style.use('anjalistyle')

def munge_scale(scale):
    """Make the scale param be a string suitable for printing"""
    return re.sub(r"\.","_",str(scale))

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
    matpow = sorted(glob.glob(os.path.join(sdir,"powerspectrum-"+str(scale)+"*.txt")))
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
        plt.semilogx(k, pk/pk_div ,ls=lss[ss], label=labels[ss])
    plt.ylim(0.99,1.01)
    plt.text(0.1, 1.005,"z="+str(np.round(1/scale -1)))
    plt.yticks((1-0.01,1-0.005, 1, 1.005, 1.01), ("0.990","0.995","1.000", "1.005", "1.010"))
    plt.xlim(0.01, 10)
    plt.legend(frameon=False, loc='lower left',fontsize=12)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P}(k)$ ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks_rel-"+munge_scale(scale)+".pdf"))
    plt.clf()

if __name__ == "__main__":
    for sc in (0.1, 0.333, 1):
        plot_single_redshift_rel_one(sc)
        plot_single_redshift_rel_camb(sc)
        plot_single_redshift(sc)
