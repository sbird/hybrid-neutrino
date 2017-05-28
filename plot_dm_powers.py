"""Scripts to plot the power spectra from our simulations"""
import os.path
import glob
import math
import numpy as np
import scipy.interpolate
import re
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

datadir = os.path.expanduser("~/data/hybrid-kspace2")
savedir = "nuplots/"
sims = ["b300p512nu0.4p","b300p512nu0.4a","b300p512nu0.4hyb"]
zerosim = "b300p512nu0"
lss = {"b300p512nu0.4p":"-.", "b300p512nu0.4a":"--","b300p512nu0.4hyb":":"}

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
    matpow = sorted(glob.glob(os.path.join(sdir,"powerspectrum-"+str(scale)+"*.txt")))
    if np.size(matpow) == 0:
        return ([],[])
    matpow = matpow[0]
    print(matpow)
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
    snap = {0.02: '0', 0.2:'2', 0.333:'4', 0.5:'6', 1:'9'}
    for ss in sims:
        sdir = os.path.join(os.path.join(datadir, ss),"output")
        matpow = glob.glob(os.path.join(sdir,"powerspectrum-nu-"+str(scale)+"*.txt"))
        try:
            (k, pk_nu) = get_nu_power(matpow[0])
        except IndexError:
            try:
                (k, pk_nu) = load_genpk(os.path.join(os.path.join(datadir,ss),"output/PK-nu-PART_00"+snap[scale]),300)
                #Shot noise
                shot=(300/512.)**3*np.ones_like(pk_nu)
                pk_nu -=shot
            except FileNotFoundError:
                continue
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

def plot_nu_single_redshift_rel_camb(scale):
    """Plot all neutrino powers relative to CAMB"""
    snap = {0.02: '0', 0.2:'2', 0.333:'4', 0.5:'6', 1:'9'}
    for ss in sims:
        cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
        cambmat = os.path.join(cambdir,"ics_matterpow_"+str(int(1/scale-1))+".dat")
        cambtrans = os.path.join(cambdir,"ics_transfer_"+str(int(1/scale-1))+".dat")
        (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
        rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
        sdir = os.path.join(os.path.join(datadir, ss),"output")
        matpow = glob.glob(os.path.join(sdir,"powerspectrum-nu-"+str(scale)+"*.txt"))
        try:
            (k, pk_nu) = get_nu_power(matpow[0])
        except IndexError:
            try:
                (k, pk_nu) = load_genpk(os.path.join(os.path.join(datadir,ss),"output/PK-nu-PART_00"+snap[scale]),300)
            except FileNotFoundError:
                continue
        plt.semilogx(k, pk_nu/rebinned(k),ls=lss[ss], label=ss)
    plt.ylim(0.9,1.2)
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_nu_camb-"+str(scale)+".pdf"))
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
    plt.ylim(0.94,1.06)
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_camb-"+str(scale)+".pdf"))
    plt.clf()

def plot_single_redshift_rel_one(scale, sims=sims, zerosim=zerosim, ymin=0.5,ymax=1.1,camb=True):
    """Plot all the simulations at a single redshift"""
    (k_div, pk_div) = _get_pk(scale, zerosim)
    if camb:
        zerocambdir = os.path.join(os.path.join(datadir, zerosim),"camb_linear")
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
    plt.ylim(ymin,ymax)
    plt.xlim(1e-2,20)
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_rel-"+str(scale)+str(zerosim[-1])+".pdf"))
    plt.clf()

if __name__ == "__main__":
    for sc in (0.02, 0.200, 0.333, 0.500, 1):
        plot_nu_single_redshift(sc)
        plot_single_redshift_rel_one(sc,ymin=0.7,ymax=1.)
        plot_single_redshift_rel_one(sc,sims=[sims[1],],zerosim=sims[0],ymin=0.98,ymax=1.02,camb=False)
        plot_single_redshift_rel_camb(sc)
        plot_nu_single_redshift_rel_camb(sc)
        plot_single_redshift(sc)
