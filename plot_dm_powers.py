"""Scripts to plot the power spectra from our simulations"""
import os.path
import glob
import math
import numpy as np
import scipy.interpolate
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

datadir = os.path.expanduser("~/data/hybrid-kspace2")
savedir = "nuplots/"
sims = ["b300p512nu0.4p","b300p512nu0.4a"]
zerosims = "b300p512nu0"
lss = {"b300p512nu0.4p":"-.", "b300p512nu0.4a":"--"}

def load_genpk(path,box):
    """Load a GenPk format power spectum, plotting the DM and the neutrinos (if present)
    Does not plot baryons."""
    #Load DM P(k)
    matpow=np.loadtxt(path)
    scale=2*math.pi/box
    #Adjust Fourier convention to match CAMB.
    simk=matpow[1:,0]*scale
    Pk=matpow[1:,1]/scale**3*(2*math.pi)**3
    return (simk,Pk)

def get_nu_power(filename):
    """Reads the neutrino power spectrum.
    Format is: ( k, P_nu(k) ).
    Units are: 1/L, L^3, where L is
    Gadget internal length units for
    Gadget-2 and Mpc/h for MP-Gadget."""
    data = np.loadtxt(filename)
    k = data[:,0]
    #Convert fourier convention to CAMB.
    pnu = data[:,1]
    return (k, pnu)

def get_camb_nu_power(matpow, transfer):
    """Plot the neutrino power spectrum from CAMB.
    This is just the matter power multiplied
    by the neutrino transfer function.
    CAMB internal units are used.
    Assume they have the same k binning."""
    matter = np.loadtxt(matpow)
    trans = np.loadtxt(transfer)
    #Adjust Fourier convention to match CAMB.
    tnufac = (trans[:,5]/trans[:,6])**2
    return matter[:,0], matter[:,1]*tnufac

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

def plot_nu_single_redshift(scale):
    """Plot all the neutrino power in simulations at a single redshift"""
    snap = {0.333:'3', 0.5:'4', 1:'8'}
    for ss in sims:
        sdir = os.path.join(os.path.join(datadir, ss),"output")
        matpow = glob.glob(os.path.join(sdir,"powerspectrum-nu-"+str(scale)+"*.txt"))
        try:
            (k, pk_nu) = get_nu_power(matpow[0])
        except IndexError:
            (k, pk_nu) = load_genpk(os.path.join(os.path.join(datadir,ss),"output/PK-nu-PART_00"+snap[scale]),300)
        plt.loglog(k, pk_nu,ls=lss[ss], label=ss)
    cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
    cambmat = os.path.join(cambdir,"ics_matterpow_"+str(int(1/scale-1))+".dat")
    cambtrans = os.path.join(cambdir,"ics_transfer_"+str(int(1/scale-1))+".dat")
    (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
    rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
    plt.semilogx(k, rebinned(k),ls=":", label="CAMB")
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks-nu-"+str(scale)+".pdf"))
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
    plt.ylim(0.8,1.4)
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_camb-"+str(scale)+".pdf"))
    plt.clf()

def plot_single_redshift_rel_one(scale):
    """Plot all the simulations at a single redshift"""
    (k_div, pk_div) = _get_pk(scale, zerosims)
    zerocambdir = os.path.join(os.path.join(datadir, zerosims),"camb_linear")
    camb = os.path.join(zerocambdir,"ics_matterpow_"+str(int(1/scale-1))+".dat")
    (zero_k, zero_pk_c) = get_camb_power(camb)
    zero_reb=scipy.interpolate.interpolate.interp1d(zero_k,zero_pk_c)
    cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
    cambpath = os.path.join(cambdir,"ics_matterpow_"+str(int(1/scale-1))+".dat")
    (k_c, pk_c) = get_camb_power(cambpath)
    plt.semilogx(k_c, pk_c/zero_reb(k_c),ls=":", label="CAMB")
    for ss in sims:
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        plt.semilogx(k, pk/pk_div,ls=lss[ss], label=ss)
    if scale > 0.8:
        (kk,pkk) = load_genpk(os.path.join(os.path.join(datadir,sims[0]),"output/PK-DM-PART_009"),300)
        (kk,pkknu) = load_genpk(os.path.join(os.path.join(datadir,sims[0]),"output/PK-nu-PART_009"),300)
        (zk, zpk) = load_genpk(os.path.join(os.path.join(datadir,zerosims),"output/PK-DM-PART_009"),300)
        plt.semilogx(kk, ((0.3-0.01)*pkk**0.5/0.3+pkknu**0.5*0.01/0.3)**2/zpk, ls="-")
        (kk,pkk) = load_genpk(os.path.join(os.path.join(datadir,sims[1]),"output/PK-DM-PART_009"),300)
        matpow = glob.glob(os.path.join(os.path.join(datadir,sims[1]),"output/powerspectrum-nu-"+str(scale)+"*.txt"))
        (knu, pkknu) = get_nu_power(matpow[0])
        reb=scipy.interpolate.interpolate.interp1d(knu,pkknu, bounds_error=False)
        plt.semilogx(kk, ((0.3-0.01)*pkk**0.5/0.3+reb(kk)**0.5*0.01/0.3)**2/zpk, ls="-")
    plt.ylim(0.5,1.1)
    plt.xlim(1e-2,20)
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_rel-"+str(scale)+".pdf"))
    plt.clf()

if __name__ == "__main__":
    for sc in (0.333, 0.5, 1):
        plot_nu_single_redshift(sc)
        plot_single_redshift_rel_one(sc)
        plot_single_redshift_rel_camb(sc)
        plot_single_redshift(sc)
