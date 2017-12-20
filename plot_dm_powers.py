"""Scripts to plot the power spectra from our simulations"""
import os.path
import glob
import math
import re
import numpy as np
import scipy.interpolate
import scipy.signal
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from nbodykit.lab import BigFileCatalog

datadir = os.path.expanduser("~/data/hybrid-kspace2")
savedir = "nuplots/"
sims = ["b300p512nu0.4hyb", "b300p512nu0.4a","b300p512nu0.4p"]
checksims = ["b300p512nu0.4hyb-all", "b300p512nu0.4hyb-nutime", "b300p512nu0.4hyb-vcrit", "b300p512nu0.4hyb", "b300p512nu0.4p", "b300p512nu0.4hyb-single"]
zerosim = "b300p512nu0"
lss = {"b300p512nu0.4p":"-.", "b300p512nu0.4a":"--","b300p512nu0.4hyb":"-","b300p512nu0.4hyb-single":"-.","b300p512nu0.4hyb-vcrit":"--","b300p512nu0.4hyb-nutime":":","b300p512nu0.4hyb-all":":","b300p512nu0.06a":"-"}
labels = {"b300p512nu0.4p":"PARTICLE", "b300p512nu0.4a":"LINRESP","b300p512nu0.4hyb":"HYBRID","b300p512nu0.4hyb-single":"HYBSING","b300p512nu0.4hyb-vcrit":"VCRIT","b300p512nu0.4hyb-nutime":"NUTIME","b300p512nu0.4hyb-all":"HYBALL","b300p512nu0.06a":"MINNU"}
scale_to_snap = {0.02: '0', 0.1: '1', 0.2:'2', 0.333:'4', 0.5:'5', 0.6667: '6', 0.8333: '7', 1:'8'}
scale_to_camb = {0.02: '49', 0.1: '9', 0.2:'4', 0.333:'2', 0.5:'1', 0.6667: '0.5', 0.8333: '0.2', 1:'0'}

def smooth(x,window_len=15,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if window_len<3:
        return x
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int((window_len-1)/2):int(-(window_len-1)/2)]

def plot_image(sim,snap, dataset=1):
    """Make a pretty picture of the mass distribution."""
    pp = os.path.join(os.path.join(datadir, sim), "output/PART_00"+str(snap))
    cat = BigFileCatalog(pp, dataset=str(dataset), header='Header')
    mesh = cat.to_mesh(Nmesh=512)
    plt.clf()
    plt.imshow(np.log10(mesh.preview(axes=(0, 1))), extent=(0,300,0,300))
    plt.xlabel("x (Mpc/h)")
    plt.ylabel("y (Mpc/h)")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "dens-plt-"+munge_scale(sim)+"t"+str(dataset)+".pdf"))
    plt.clf()

def load_genpk(path,box):
    """Load a GenPk format power spectum, plotting the DM and the neutrinos (if present)
    Does not plot baryons."""
    #Load DM P(k)
    matpow=np.loadtxt(path)
    scale=2*math.pi/box
    #Adjust Fourier convention to match CAMB.
    simk=matpow[1:,0]*scale
    Pk=matpow[1:,1]/scale**3*(2*math.pi)**3
    return modecount_rebin(simk, Pk, matpow[1:,2],minmodes=30)
#     return (simk,Pk)

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

def modecount_rebin(kk, pk, modes,minmodes=20, ndesired=200):
    """Rebins a power spectrum so that there are sufficient modes in each bin"""
    assert np.all(kk) > 0
    logkk=np.log10(kk)
    mdlogk = (np.max(logkk) - np.min(logkk))/ndesired
    istart=iend=1
    count=0
    k_list=[kk[0]]
    pk_list=[pk[0]]
    targetlogk=mdlogk+logkk[istart]
    while iend < np.size(logkk)-1:
        count+=modes[iend]
        iend+=1
        if count >= minmodes and logkk[iend-1] >= targetlogk:
            pk1 = np.sum(modes[istart:iend]*pk[istart:iend])/count
            kk1 = np.sum(modes[istart:iend]*kk[istart:iend])/count
            k_list.append(kk1)
            pk_list.append(pk1)
            istart=iend
            targetlogk=mdlogk+logkk[istart]
            count=0
    return (np.array(k_list), np.array(pk_list))

def get_camb_power(matpow, rebin=False):
    """Plot the power spectrum from CAMB
    (or anything else where no changes are needed)"""
    data = np.loadtxt(matpow)
    kk = data[:,0]
    ii = np.where(kk > 0.)
    #Rebin power so that there are enough modes in each bin
    kk = kk[ii]
    pk = data[:,1][ii]
    if rebin:
        try:
            modes = data[:,2][ii]
        except IndexError:
            modes = (6+np.arange(np.size(pk))**2)
        return modecount_rebin(kk, pk, modes,minmodes=30)
    return (kk,pk)

def _get_pk(scale, ss):
    """Get the matter power spectrum"""
    sdir = os.path.join(os.path.join(datadir, ss),"output")
    matpow = sorted(glob.glob(os.path.join(sdir,"powerspectrum-"+str(scale)+"*.txt")))
    if np.size(matpow) == 0:
        return ([],[])
    matpow = matpow[0]
#     print(matpow)
    return get_camb_power(matpow, rebin=True)

def munge_scale(scale):
    """Make the scale param be a string suitable for printing"""
    return re.sub(r"\.","_",str(scale))

#vcrit = 300:
#0.0328786
#vcrit = 500:
#0.116826
#vcrit = 1000:
#0.450869
#vcrit = 750:
#0.275691
def get_hyb_nu_power(nu_filename, genpk_neutrino, box, part_prop=0.116826, npart=512, nu_part_time=0.5, scale=1.):
    """Get the total matter power spectrum when some of it is in particles, some analytic."""
    (k_sl, pk_sl) = get_nu_power(nu_filename)
    ii = np.where(k_sl != 0.)
    if scale < nu_part_time:
        return k_sl[ii], pk_sl[ii]
    (k_part,pk_part)=load_genpk(genpk_neutrino,box)
    rebinned=scipy.interpolate.interpolate.interp1d(k_part,pk_part,fill_value='extrapolate')
    pk_part_r = rebinned(k_sl[ii])
    shot=(300/npart)**3*np.ones(np.size(pk_part_r))
    pk = (part_prop*np.sqrt(pk_part_r-shot)+(1-part_prop)*np.sqrt(pk_sl[ii]))**2
    return (k_sl[ii], pk)

def plot_single_redshift(scale):
    """Plot all the simulations at a single redshift"""
    for ss in sims:
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        plt.loglog(k, pk,ls=lss[ss], label=labels[ss])
    cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
    camb = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
    (k_camb, pk_camb) = get_camb_power(camb)
    rebinned=scipy.interpolate.interpolate.interp1d(k_camb,pk_camb)
    plt.semilogx(k, rebinned(k),ls=":", label="CAMB")
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P} (k)$ (Mpc/h)$^3$")
    plt.legend(loc=0,fontsize=12)
    plt.savefig(os.path.join(savedir, "pks-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_crosscorr(scale):
    """Plot the crosscorrelation coefficient as a function of k for neutrinos and DM."""
    cc_sims = ["b300p512nu0.4p","b300p512nu0.4hyb"]
    for ss in cc_sims:
        genpk_neutrino = os.path.join(os.path.join(datadir,ss),"output/PK-nu-PART_00"+scale_to_snap[scale])
        (_, pk_nu) = load_genpk(genpk_neutrino,300)
        genpk_dm = os.path.join(os.path.join(datadir,ss),"output/PK-DM-PART_00"+scale_to_snap[scale])
        (_, pk_dm) = load_genpk(genpk_dm,300)
        genpk_cross = os.path.join(os.path.join(datadir,ss),"output/PK-DMxnu-PART_00"+scale_to_snap[scale])
        (k_cross, pk_cross) = load_genpk(genpk_cross,300)
        corr_coeff = pk_cross / np.sqrt(pk_dm* pk_nu)
        plt.semilogx(k_cross, corr_coeff, ls=lss[ss],label=labels[ss])
    plt.axvline(x=0.8, ls=":", color="grey")
    plt.legend(loc=0,fontsize=12)
    plt.xlabel(r"k (h/Mpc)")
    plt.ylabel(r"Cross-power")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "corr_coeff-"+munge_scale(scale)+".pdf"))
    plt.clf()

def select_nu_power(scale, ss):
    """Get the neutrino power spectrum that is wanted"""
    sdir = os.path.join(os.path.join(datadir, ss),"output")
    matpow = glob.glob(os.path.join(sdir,"powerspectrum-nu-"+str(scale)+"*.txt"))
    genpk_neutrino = os.path.join(os.path.join(datadir,ss),"output/PK-nu-PART_00"+scale_to_snap[scale])
    try:
        try:
            npart = 512
            if re.search("single",ss):
                npart = 256
            #vcrit = 750
            nu_part_time = 0.5
            part_prop = 0.275691
#             part_prop = 0.116826
            if re.search("vcrit",ss):
                #vcrit = 1000
                nu_part_time = 0.5
                part_prop = 0.450869
            if re.search("all",ss) or re.search("nutime",ss):
                #vcrit = 5000
                part_prop = 1.
                nu_part_time = 0.5
                if re.search("all",ss):
                    nu_part_time = 0.25
            (k, pk_nu) = get_hyb_nu_power(matpow[0], genpk_neutrino, 300, part_prop=part_prop, npart=npart, nu_part_time = nu_part_time, scale=scale)
        except FileNotFoundError:
            if not re.search("a$",ss):
                print("Problem",genpk_neutrino)
            (k, pk_nu) = get_nu_power(matpow[0])
    except IndexError:
        (k, pk_nu) = load_genpk(genpk_neutrino,300)
        #Shot noise
        shot=(300/512.)**3*np.ones_like(pk_nu)
        pk_nu -=shot
    return (k, pk_nu)

def plot_nu_single_redshift(scale,psims=sims,fn="nu"):
    """Plot all the neutrino power in simulations at a single redshift"""
    for ss in psims:
        (k, pk_nu) = select_nu_power(scale, ss)
        plt.loglog(k, pk_nu,ls=lss[ss], label=labels[ss])
    kl = np.logspace(-2, 2)
    plt.loglog(kl, (300/512)**3*np.ones_like(kl), color="lightgrey", ls=":")
    cambdir = os.path.join(os.path.join(datadir, psims[0]),"camb_linear")
    cambmat = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
    cambtrans = os.path.join(cambdir,"ics_transfer_"+scale_to_camb[scale]+".dat")
    (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
    rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
    plt.semilogx(k, rebinned(k),ls=":", label="CAMB")
    plt.ylim(ymin=1e-5)
    plt.xlim(0.05, 10)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P}_\nu(k)$ (Mpc/h)$^3$")
    plt.legend(loc=0,fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks-"+fn+"-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_nu_single_redshift_rel_camb(scale):
    """Plot all neutrino powers relative to CAMB"""
    for ss in sims:
        cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
        cambmat = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
        cambtrans = os.path.join(cambdir,"ics_transfer_"+scale_to_camb[scale]+".dat")
        (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
        rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
        (k, pk_nu) = select_nu_power(scale, ss)
        pkfilt = smooth(pk_nu/rebinned(k))
        plt.semilogx(k, pkfilt,ls=lss[ss], label=labels[ss])
    plt.ylim(0.9,1.2)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P}_\nu / \mathrm{P}_\nu^\mathrm{CAMB}(k)$")
    plt.legend(loc=0,fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks_nu_camb-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_nu_single_redshift_rel_one(scale, psims=sims[1:], pzerosim=sims[0], ymin=0.8,ymax=1.2,fn="rel"):
    """Plot all neutrino powers relative to one simulation"""
    (k_div, pk_div) = select_nu_power(scale, pzerosim)
    rebinned=scipy.interpolate.interpolate.interp1d(k_div,pk_div,fill_value='extrapolate')
    for ss in psims:
        (k, pk_nu) = select_nu_power(scale, ss)
        pkfilt = smooth(pk_nu/rebinned(k))
        plt.semilogx(k, pkfilt,ls=lss[ss], label=labels[ss])
    plt.ylim(ymin,ymax)
    plt.legend(loc=0,fontsize=12)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P}_\nu(k)$ ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks_nu_"+fn+"-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_single_redshift_rel_camb(scale):
    """Plot all the simulations at a single redshift"""
    for ss in sims:
        cambdir = os.path.join(os.path.join(datadir, ss),"camb_linear")
        camb = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        (k_camb, pk_camb) = get_camb_power(camb)
        rebinned=scipy.interpolate.interpolate.interp1d(k_camb,pk_camb)
        pkfilt = smooth(pk/rebinned(k))
        plt.semilogx(k, pkfilt,ls=lss[ss], label=labels[ss])
    plt.ylim(0.94,1.06)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P} / \mathrm{P}^\mathrm{CAMB}(k)$")
    plt.legend(loc=0,fontsize=12)
    plt.savefig(os.path.join(savedir, "pks_camb-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_single_redshift_rel_one(scale, psims=sims, pzerosim=zerosim, ymin=0.5,ymax=1.1,camb=True,fn="rel"):
    """Plot all the simulations at a single redshift"""
    (k_div, pk_div) = _get_pk(scale, pzerosim)
    rebinned=scipy.interpolate.interpolate.interp1d(k_div,pk_div,fill_value="extrapolate")
    if camb:
        zerocambdir = os.path.join(os.path.join(datadir, pzerosim),"camb_linear")
        camb = os.path.join(zerocambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
        (zero_k, zero_pk_c) = get_camb_power(camb)
        zero_reb=scipy.interpolate.interpolate.interp1d(zero_k,zero_pk_c)
        cambdir = os.path.join(os.path.join(datadir, psims[0]),"camb_linear")
        cambpath = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
        (k_c, pk_c) = get_camb_power(cambpath)
        plt.semilogx(k_c, pk_c/zero_reb(k_c),ls=":", label="CAMB")
    for ss in psims:
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        plt.semilogx(k, pk/rebinned(k),ls=lss[ss], label=labels[ss])
    plt.ylim(ymin,ymax)
    plt.xlim(1e-2,20)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P}(k)$ ratio")
    plt.legend(loc=0,fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks_"+fn+"-"+munge_scale(scale)+str(pzerosim[-1])+".pdf"))
    plt.clf()

def plot_fermi_dirac(Mnu, zz):
    """Plot the fermi-dirac distribution for neutrinos at redshift z
    Argument is total neutrino mass."""
    tnu = 2.7255 * (4/11.)**(1./3) * 1.00328
    bolevk = 8.61734e-5
    nu_v = bolevk * tnu/ (Mnu/3) * (1+zz) * 2.99792e5
    fdk = lambda x: x*x/(np.exp(x)+1)
    xx = np.arange(0, 9*nu_v,10)
    ff = np.zeros_like(xx, dtype=np.float64)
    for i in range(np.size(xx)):
        (fd, _) = scipy.integrate.quad(fdk, 0, xx[i]/nu_v)
        ff[i] = fd / (1.5 * 1.20206)
    plt.plot(xx, ff, "-", label="Fermi-Dirac distribution", color="blue")
    plt.fill_between(xx, 0, ff, where=xx < 750, facecolor='grey', interpolate=True, alpha=0.5)
    plt.ylim(0,1)
    plt.xlim(0,np.max(xx))
    plt.xlabel(r"$v_\nu$ (km/s)")
    plt.ylabel(r"Fermi-Dirac cum. prob.")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "fermidirac.pdf"))
    plt.clf()

if __name__ == "__main__":
#     plot_image(sims[0],8)
#     plot_image(sims[2],8,1)
#     plot_image(sims[2],8,2)
#     plot_image(sims[1],8,1)
#     plot_image(sims[1],8,2)
    plot_fermi_dirac(0.4,0)
    for sc in (0.02, 0.100, 0.200, 0.333, 0.500, 0.6667, 0.8333, 1):
        plot_nu_single_redshift(sc)
        plot_nu_single_redshift(sc,checksims,fn="cknu")
        plot_crosscorr(sc)
        plot_single_redshift_rel_one(sc,ymin=0.6,ymax=1.)
        plot_nu_single_redshift_rel_one(sc)
        plot_single_redshift_rel_one(sc,psims=["b300p512nu0.06a",],fn="lowmass",ymin=0.92, ymax=1.0)
        plot_nu_single_redshift_rel_one(sc,psims=checksims[1:],pzerosim=checksims[0],fn="ckrel",ymin=0.8,ymax=1.2)
        plot_single_redshift_rel_one(sc,psims=[sims[1],sims[2]],pzerosim=sims[0],ymin=0.98,ymax=1.02,camb=False)
        plot_single_redshift_rel_one(sc,psims=checksims,pzerosim=sims[0],camb=False,ymin=0.99,ymax=1.01,fn="ckrel")
        plot_single_redshift_rel_one(sc,psims=checksims,fn="ckrel")
        plot_single_redshift_rel_camb(sc)
        plot_nu_single_redshift_rel_camb(sc)
        plot_single_redshift(sc)
