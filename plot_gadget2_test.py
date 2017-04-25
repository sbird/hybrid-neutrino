"""Scripts to plot the power spectra from our simulations"""
import os.path
import glob
import math
import numpy as np
import scipy.interpolate
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

datadir = os.path.expanduser("~/data/")
savedir = "nuplots/"
sims = "test_nu_part/"
zerosim = "test_rad0/"
# zerosim = "test_nu_part0/"

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
    k = data[:,0]*1e3
    #Convert fourier convention to CAMB.
    pnu = data[:,1]/1e9
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

def get_camb_cdm_power(matpow, transfer):
    """Plot the neutrino power spectrum from CAMB.
    This is just the matter power multiplied
    by the neutrino transfer function.
    CAMB internal units are used.
    Assume they have the same k binning."""
    matter = np.loadtxt(matpow)
    trans = np.loadtxt(transfer)
    #Adjust Fourier convention to match CAMB.
    tnufac = (trans[:,7]/trans[:,6])**2
    return matter[:,0], matter[:,1]*tnufac

#nu = pm (T_nu/T_tot)^2
#cdm = pm (T_cdm/T_tot)^2
#d_tot = pm ((om-on)/om T_cdm/T_tot + (on/om) T_nu/T_tot)**2
#d_tot = pm /(T_tot om)^2 (om-on T_cdm + on T_nu)**2

def get_camb_power(matpow):
    """Plot the power spectrum from CAMB
    (or anything else where no changes are needed)"""
    data = np.loadtxt(matpow)
    kk = data[:,0]
    ii = np.where(kk > 0.)
    return (kk[ii], data[:,1][ii])

def get_camb_total_ratio(matpow, transfer):
    """Get total power from genpk"""
    omegam = 0.288
    mnu=0.3
    kt,total = get_camb_power(matpow)
    kc,cdm = get_camb_cdm_power(matpow, transfer)
    kn,nu = get_camb_nu_power(matpow, transfer)
    total_sum = ((omegam-omeganu(mnu))*cdm**0.5/omegam+nu**0.5*omeganu(mnu)/omegam)**2
    return kt, total,total_sum

def _get_pk(snap):
    """Get the matter power spectrum"""
    sdir = os.path.join(os.path.join(datadir, sims),"output")
    matpow = os.path.join(sdir,"powerspec_tot_"+snap+".txt")
    (k,pk) = get_camb_power(matpow)
    sdir = os.path.join(os.path.join(datadir, zerosim),"output")
    matpow = os.path.join(sdir,"powerspec_tot_"+snap+".txt")
    (zk,zpk) = get_camb_power(matpow)
    plt.semilogx(k*1e3, pk/zpk, ls="--")

# def plot_nu_single_redshift(snap):
#     """Plot all the neutrino power in simulations at a single redshift"""
#     sdir = os.path.join(os.path.join(datadir, sims),"output")
#     matpow = os.path.join(sdir,"powerspec_nu_"+snap+".txt")
#     (k, pk_nu) = get_nu_power(matpow)
#     plt.loglog(k, pk_nu,ls="-")
#     cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
#     cambmat = os.path.join(cambdir,"ics_matterpow_0.dat")
#     cambtrans = os.path.join(cambdir,"ics_transfer_0.dat")
#     (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
#     rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
#     plt.semilogx(k, rebinned(k),ls=":", label="CAMB")
#     plt.legend(loc=0)
#     plt.savefig(os.path.join(savedir, "pks-nu-9.pdf"))
#     plt.clf()

def plot_nu_single_redshift_rel_camb(snap,zz):
    """Plot all neutrino powers relative to CAMB"""
    cambdir = os.path.join(os.path.join(datadir, sims),"camb_linear")
    cambmat = os.path.join(cambdir,"ics_matterpow_"+zz+".dat")
    cambtrans = os.path.join(cambdir,"ics_transfer_"+zz+".dat")
    (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
    rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
    sdir = os.path.join(os.path.join(datadir, sims),"output")
    matpow = os.path.join(sdir,"powerspec_nu_"+snap+".txt")
    (k, pk_nu) = get_nu_power(matpow)
    plt.semilogx(k, pk_nu/rebinned(k),ls="-")
    plt.ylim(0.9,1.1)
#     plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_nu_camb-"+snap+".pdf"))
    plt.clf()

def plot_single_redshift_rel_camb(snap,zz, sim):
    """Plot all neutrino powers relative to CAMB"""
    cambdir = os.path.join(os.path.join(datadir, sim),"camb_linear")
    cambmat = os.path.join(cambdir,"ics_matterpow_"+zz+".dat")
    (k_nu_camb, pk_nu_camb) = get_camb_power(cambmat)
    rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
    sdir = os.path.join(os.path.join(datadir, sim),"output")
    matpow = os.path.join(sdir,"powerspec_tot_"+snap+".txt")
    (k, pk_nu) = get_camb_power(matpow)
    plt.semilogx(k*1e3, pk_nu/1e9/rebinned(k*1e3),ls="-")
    plt.ylim(0.9,1.1)
#     plt.legend(loc=0)
#     plt.savefig(os.path.join(savedir, "pks_r_camb-"+snap+".pdf"))
#     plt.clf()

# def plot_single_redshift_rel_camb(scale):
#     """Plot all the simulations at a single redshift"""
#     for ss in sims:
#         cambdir = os.path.join(os.path.join(datadir, ss),"camb_linear")
#         camb = os.path.join(cambdir,"ics_matterpow_0.dat")
#         (k, pk) = _get_pk(scale, ss)
#         if np.size(k) == 0:
#             continue
#         (k_camb, pk_camb) = get_camb_power(camb)
#         rebinned=scipy.interpolate.interpolate.interp1d(k_camb,pk_camb)
#         plt.semilogx(k, pk/rebinned(k),ls=lss[ss], label=ss)
#     plt.ylim(0.9,1.2)
#     plt.legend(loc=0)
#     plt.savefig(os.path.join(savedir, "pks_camb-"+str(scale)+".pdf"))
#     plt.clf()
#
def plot_single_redshift_rel_one(snap='009',zz='0',ymin=0.7,ymax=1.0):
    """Plot all the simulations at a single redshift"""
#     (k_div, pk_div) = _get_pk(scale, zerosim)
    try:
        get_genpk_total(sims, zerosim,snap)
    except FileNotFoundError:
        pass
    try:
        _get_pk(snap)
    except FileNotFoundError:
        pass
    zerocambdir = os.path.join(os.path.join(datadir, zerosim),"camb_linear")
    camb = os.path.join(zerocambdir,"ics_matterpow_"+zz+".dat")
    (zero_k, zero_pk_c) = get_camb_power(camb)
    zero_reb=scipy.interpolate.interpolate.interp1d(zero_k,zero_pk_c)
    cambdir = os.path.join(os.path.join(datadir, sims),"camb_linear")
    cambpath = os.path.join(cambdir,"ics_matterpow_"+zz+".dat")
    (k_c, pk_c) = get_camb_power(cambpath)
    plt.semilogx(k_c, pk_c/zero_reb(k_c),ls=":", label="CAMB")
    plt.ylim(ymin,ymax)
    plt.xlim(1e-2,20)
    plt.legend(loc=0)
    plt.savefig(os.path.join(savedir, "pks_rel-"+snap+".pdf"))
    plt.clf()

# def omeganu(mnu):
#     """Omeganu"""
#     return mnu/93.14/0.71**2

def get_genpk_total(sim, zero, snap):
    """Get total power from genpk"""
    omegam = 0.288
    mnu=0.3
    (kk,pkk) = load_genpk(os.path.join(os.path.join(datadir,sim),"output/PK-DM-snap_"+snap),300)
    nupow = os.path.join(os.path.join(datadir,sim),"output/powerspec_nu_"+snap+".txt")
    (knu, pkknu) = get_nu_power(nupow)
    nuint=scipy.interpolate.interpolate.interp1d(knu,pkknu)
    (zk, zpk) = load_genpk(os.path.join(os.path.join(datadir,zero),"output/PK-DM-snap_"+snap),300)
    ptot = ((omegam-omeganu(mnu))*pkk[1:-40]**0.5/omegam+nuint(kk[1:-40])**0.5*omeganu(mnu)/omegam)**2
    plt.semilogx(kk[1:-40], ptot/zpk[1:-40], ls="-")

def get_genpk_total2(zero, snap,zz):
    """Get total power from genpk"""
    (zk, zpk) = load_genpk(os.path.join(os.path.join(datadir,zero),"output/PK-DM-snap_"+snap),300)
    cambdir = os.path.join(os.path.join(datadir, zero),"camb_linear")
    cambmat = os.path.join(cambdir,"ics_matterpow_"+zz+".dat")
    (k_nu_camb, pk_nu_camb) = get_camb_power(cambmat)
    rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
    plt.semilogx(zk, zpk/rebinned(zk),ls="-")

if __name__ == "__main__":
    plot_single_redshift_rel_camb('000','99',sims)
    plot_single_redshift_rel_camb('000','99',zerosim)
    plt.savefig(os.path.join(savedir, "pks_r_camb-000.pdf"))
    plt.clf()
    plot_single_redshift_rel_camb('001','49',sims)
    plot_single_redshift_rel_camb('001','49',zerosim)
    plt.savefig(os.path.join(savedir, "pks_r_camb-001.pdf"))
    plt.clf()
    plot_single_redshift_rel_camb('002','9',sims)
    plot_single_redshift_rel_camb('002','9',zerosim)
    plt.savefig(os.path.join(savedir, "pks_r_camb-002.pdf"))
    plt.clf()
    plot_single_redshift_rel_camb('009','0',sims)
    plot_single_redshift_rel_camb('009','0',zerosim)
    plt.savefig(os.path.join(savedir, "pks_r_camb-009.pdf"))
    plt.clf()
    plot_single_redshift_rel_one('000','99')
    plot_single_redshift_rel_one('001','49')
    plot_single_redshift_rel_one('002','9')
    plot_single_redshift_rel_one()
    plot_nu_single_redshift_rel_camb('000','99')
    plot_nu_single_redshift_rel_camb('001','49')
    plot_nu_single_redshift_rel_camb('002','9')
    plot_nu_single_redshift_rel_camb('009','0')
