"""Scripts to plot the power spectra from our simulations"""
import os.path
import glob
import math
import re
import numpy as np
import scipy.interpolate
import scipy.signal
import bigfile
# import halo_mass_function
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from nbodykit.lab import BigFileCatalog

plt.style.use('anjalistyle')

datadir = os.path.expanduser("data")
savedir = "nuplots/"
sims = ["b300p512nu0.4hyb850", "b300p512nu0.4a","b300p512nu0.4p1024"]
checksims2 = ["b300p512nu0.4hyb850", "b300p512nu0.4p", "b300p512nu0.4p1024", "b300p512nu0.4hyb-nutime"]
checksims = ["b300p512nu0.4hyb850", "b300p512nu0.4hyb", "b300p512nu0.4hyb-nutime850", "b300p512nu0.4hyb-vcrit", "b300p512nu0.4hyb-single850"]
zerosim = "b300p512nu0"
lowmass=["b300p512nu0.06a","b300p512nu0.06p"]
lss = {"b300p512nu0.4p1024":"--", "b300p512nu0.4p":"-.","b300p512nu0.4a":"-.","b300p512nu0.4hyb850":"-","b300p512nu0.4hyb":"-.","b300p512nu0.4hyb-single850":"-.","b300p512nu0.4hyb-vcrit":"--","b300p512nu0.4hyb-nutime850":":","b300p512nu0.4hyb-nutime":":","b300p512nu0.06a":"-", "b300p512nu0.06p":"--"}
alpha = {"b300p512nu0.4p1024":1,"b300p512nu0.4p":1, "b300p512nu0.4a": 0,"b300p512nu0.4hyb":0.5,"b300p512nu0.4hyb850":0.5,"b300p512nu0.4hyb-single850":0.3,"b300p512nu0.4hyb-vcrit":0.3,"b300p512nu0.4hyb-nutime850":0.3,"b300p512nu0.4hyb-nutime":0.3,"b300p512nu0.4hyb-all":0.3,"b300p512nu0.06a":0, "b300p512nu0.06p":0}
labels = {"b300p512nu0.4p1024":"PARTICLE-1024","b300p512nu0.4p":"PARTICLE",  "b300p512nu0.4a":"LINRESP","b300p512nu0.4hyb850":"HYBRID","b300p512nu0.4hyb-single850":"HYBRID-256","b300p512nu0.4hyb":"HYBRID-v750", "b300p512nu0.4hyb-vcrit":"HYBRID-v1000","b300p512nu0.4hyb-nutime":"HYBRID-v5000","b300p512nu0.4hyb-nutime850":"HYBRID-z4","b300p512nu0.06a":"LINRESP-MINNU", "b300p512nu0.06p":"PARTICLE-MINNU"}
colors = {"b300p512nu0.4p1024": '#d62728', "b300p512nu0.4p":"#7f7f7f","b300p512nu0.4a":'#1f77b4', "b300p512nu0.4hyb850":'#2ca02c',"b300p512nu0.4hyb-single850":'#7f7f7f',"b300p512nu0.4hyb-vcrit":'#bcbd22',"b300p512nu0.4hyb-nutime": '#ff7f0e',"b300p512nu0.4hyb-nutime850": '#e377c2',"b300p512nu0.06a":'#1f77b4',"b300p512nu0.06p": '#d62728',"b300p512nu0.4hyb":'#8c564b'}

#new_colors = [,,, ,
#              ,  '#7f7f7f',
#              '#bcbd22', '#17becf']
scale_to_snap = {0.02: '0', 0.1: '1', 0.2:'2', 0.3333:'4', 0.5:'5', 0.6667: '6', 0.8333: '7', 1:'8'}
scale_to_camb = {0.02: '49', 0.1: '9', 0.2:'4', 0.3333:'2', 0.5:'1', 0.6667: '0.5', 0.8333: '0.2', 1:'0'}

def HMFFromFOF(foftable, h0=False, bins='auto'):
    """Print a conventionally normalised halo mass function from the FOF tables.
    Units returned are:
    dn/dM (M_sun/Mpc^3) (comoving) Note no little-h!
    If h0 == True, units are dn/dM (h^4 M_sun/Mpc^3)
    bins specifies the number of evenly spaced bins if an integer,
    or one of the strings understood by numpy.histogram."""
    bf = bigfile.BigFile(foftable)
    #1 solar in g
    msun_in_g = 1.989e33
    #1 Mpc in cm
    Mpc_in_cm = 3.085678e+24
    #In units of 10^10 M_sun by default.
    try:
        imass_in_g = bf["Header"].attrs["UnitMass_in_g"]
    except KeyError:
        imass_in_g = 1.989e43
    #Length in units of kpc/h by default
    try:
        ilength_in_cm = bf["Header"].attrs["UnitLength_in_cm"]
    except KeyError:
        ilength_in_cm = 3.085678e+21
    hub = bf["Header"].attrs["HubbleParam"]
    box = bf["Header"].attrs["BoxSize"]
    #Convert to Mpc from kpc/h:
    box *= ilength_in_cm / hub / Mpc_in_cm
    masses = bf["FOFGroups/Mass"][:]
    #This is N(M) evenly spaced in log(M)
    NM, Mbins = np.histogram(np.log10(masses), bins=bins)
    #Convert Mbins to Msun
    Mbins = 10**Mbins
    Mbins *= (imass_in_g / msun_in_g)
    #Find dM:
    #This is dn/dM (Msun)
    dndm = NM/(Mbins[1:] - Mbins[:-1])
    Mcent = (Mbins[1:] + Mbins[:-1])/2.
    #Now divide by the volume:
    dndm /= box**3
    if h0:
        dndm /= hub**4
    return Mcent, dndm

def smooth(x,window_len=4):
    """smooth the data using a moving average.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer

    output:
        the smoothed signal
    """
    out = np.zeros_like(x)
    diff = (window_len-1)//2
    for i in range(0,diff+1):
        out[i] = np.mean(x[:i+diff])
    for i in range(diff+1,len(x)-diff):
        out[i] = np.mean(x[i-1-diff:i+diff])
    for i in range(len(x)-diff,len(x)):
        out[i] = np.mean(x[i-1-diff:])
    return out

def plot_image(sim,snap, dataset=1, colorbar=False):
    """Make a pretty picture of the mass distribution."""
    pp = os.path.join(os.path.join(datadir, sim), "output/PART_00"+str(snap))
    cat = BigFileCatalog(pp, dataset=str(dataset), header='Header')
    mesh = cat.to_mesh(Nmesh=512)
    plt.clf()
    plt.imshow(np.log10(mesh.preview(axes=(0, 1))/512), extent=(0,3,0,3), vmin=-0.2, vmax=0.2)
    if colorbar:
        plt.colorbar()
    plt.xlabel("x (100 Mpc/h)")
    plt.ylabel("y (100 Mpc/h)")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "dens-plt-"+munge_scale(sim)+"t"+str(dataset)+".pdf"))
    plt.clf()

def load_genpk(path):
    """Load a GenPk format power spectum, plotting the DM and the neutrinos (if present)
    Does not plot baryons."""
    #Load DM P(k)
    matpow=np.loadtxt(path)
    scale = 1e3
    #Adjust Fourier convention to match CAMB.
    simk=matpow[1:,0]*scale
    Pk=matpow[1:,1] /scale**3 #*(2*math.pi)**3
#     return modecount_rebin(simk, Pk, matpow[1:,2],minmodes=15)
    return (simk,Pk)

def get_nu_power(filename, modes=None):
    """Reads the neutrino power spectrum.
    Format is: ( k, P_nu(k) ).
    Units are: 1/L, L^3, where L is
    Gadget internal length units for
    Gadget-2 and Mpc/h for MP-Gadget."""
    data = np.loadtxt(filename)
    k = data[:,0]
    #Convert fourier convention to CAMB.
    pnu = data[:,1]
    if modes is not None:
        (k, pnu) = modecount_rebin(k, pnu, modes,minmodes=30)
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

def get_hyb_nu_power(nu_filename, genpk_neutrino, part_prop=0.116826, npart=512, nu_part_time=0.5, scale=1., split=False, modes=None):
    """Get the total matter power spectrum when some of it is in particles, some analytic."""
    (k_sl, pk_sl) = get_nu_power(nu_filename, modes = modes)
    ii = np.where(k_sl != 0.)
    if scale < nu_part_time:
        return k_sl[ii], pk_sl[ii], np.zeros_like(pk_sl[ii])
    (k_part,pk_part)=load_genpk(genpk_neutrino)
    rebinned=scipy.interpolate.interpolate.interp1d(k_part,pk_part,fill_value='extrapolate')
    pk_part_r = rebinned(k_sl[ii])
    shot=(300/npart)**3*np.ones(np.size(pk_part_r))
    pk = (part_prop*np.sqrt(pk_part_r-shot)+(1-part_prop)*np.sqrt(pk_sl[ii]))**2
    if split:
        return k_sl[ii], pk_part_r - shot, pk_sl[ii]
    return (k_sl[ii], pk, part_prop*shot)

def plot_single_redshift(scale):
    """Plot all the simulations at a single redshift"""
    for ss in sims:
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        plt.loglog(k, pk,ls=lss[ss], label=labels[ss], color=colors[ss])
    cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
    camb = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
    (k_camb, pk_camb) = get_camb_power(camb)
    rebinned=scipy.interpolate.interpolate.interp1d(k_camb,pk_camb)
    plt.semilogx(k, rebinned(k),ls=":", label="CAMB", color="black")
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P} (k)$ (Mpc/h)$^3$")
    plt.legend(frameon=False, loc=0,fontsize=12)
    plt.text(0.02, 0.1,"z="+str(np.round(1/scale -1)))
    plt.savefig(os.path.join(savedir, "pks-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_crosscorr(scale):
    """Plot the cross-correlation coefficient as a function of k for neutrinos and DM."""
    cc_fast_sims = ["b300p512nu0.4p1024", ] #,"b300p512nu0.4p"]
    shots = {"b300p512nu0.4p1024": 300**3/(1024**3 - 371714852), "b300p512nu0.4p":300**3/(512 - 46462529)}
    for ss in cc_fast_sims:
        genpk_neutrino = os.path.join(os.path.join(datadir,ss),"output/power-fast-nu-%.4f.txt" % scale)
        (_, pk_nu) = load_genpk(genpk_neutrino)
        genpk_dm = os.path.join(os.path.join(datadir,ss),"output/power-DM-%.4f.txt" % scale)
        (_, pk_dm) = load_genpk(genpk_dm)
        genpk_cross = os.path.join(os.path.join(datadir,ss),"output/power-fast-nuDM-%.4f.txt" % scale)
        (k_cross, pk_cross) = load_genpk(genpk_cross)
        shot = shots[ss]*np.ones_like(pk_nu)
        pksq = pk_dm * (pk_nu - shot)
        pksq[np.where(pksq <=0)] = shots[ss] * 0.03
        corr_coeff = pk_cross / np.sqrt(pksq)
        ii = np.where(k_cross > 1)
        corr_coeff[ii] = smooth(corr_coeff[ii])
        plt.semilogx(k_cross, corr_coeff, ls="-.",label="PARTICLE 1024 (fast)", color="blue")
    cc_sims = ["b300p512nu0.4hyb850","b300p512nu0.4p1024"]
    shots = {"b300p512nu0.4p1024":(300/1024)**3, "b300p512nu0.4hyb850":(300/512)**3, "b300p512nu0.4p":(300/512)**3}
    for ss in cc_sims:
        genpk_neutrino = os.path.join(os.path.join(datadir,ss),"output/power-nu-%.4f.txt" % scale)
        (_, pk_nu) = load_genpk(genpk_neutrino)
        genpk_dm = os.path.join(os.path.join(datadir,ss),"output/power-DM-%.4f.txt" % scale)
        (_, pk_dm) = load_genpk(genpk_dm)
        genpk_cross = os.path.join(os.path.join(datadir,ss),"output/power-DMnu-%.4f.txt" % scale)
        (k_cross, pk_cross) = load_genpk(genpk_cross)
        shot = shots[ss]*np.ones_like(pk_nu)
        pksq = pk_dm * (pk_nu - shot)
        pksq[np.where(pksq <=0)] = shots[ss] * 0.03
        corr_coeff = pk_cross / np.sqrt(pksq)
        ii = np.where(k_cross > 1)
        corr_coeff[ii] = smooth(corr_coeff[ii])
        plt.semilogx(k_cross, corr_coeff, ls=lss[ss],label=labels[ss], color=colors[ss])
    plt.axvline(x=1.2, ls="-", color="black")
    plt.ylim(0.75,1.05)
    plt.xlim(0.01, 10)
    plt.legend(frameon=False, loc='lower left',fontsize=12)
    plt.xlabel(r"k (h/Mpc)")
    plt.ylabel(r"Cross-correlation coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "corr_coeff-"+munge_scale(scale)+".pdf"))
    plt.clf()

#vcrit = 300:
#0.0328786
#vcrit = 500:
#0.116826
#vcrit = 1000:
#0.450869
#vcrit = 750:
#0.275691
#vcrit = 850:
#0.346203
def select_nu_power(scale, ss):
    """Get the neutrino power spectrum that is wanted"""
    sdir = os.path.join(os.path.join(datadir, ss),"output")
    #Get the modes (hack as we didn't save them before)
    try:
        mpk = glob.glob(os.path.join(sdir,"powerspectrum-"+str(scale)+"*.txt"))
        mat = np.loadtxt(mpk[0])
        ii = np.where(mat[:,2] > 0)
        modes = mat[:,2][ii]
    except IndexError:
        modes = None
    #Get neutrino power
    matpow = glob.glob(os.path.join(sdir,"powerspectrum-nu-"+str(scale)+"*.txt"))
    genpk_neutrino = os.path.join(os.path.join(datadir,ss),"output/power-nu-%.4f.txt" % scale)
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
            elif re.search("nutime850", ss):
                nu_part_time = 0.25
                part_prop = 0.346203
            elif re.search("850",ss):
                #vcrit = 850
                nu_part_time = 0.5
                part_prop = 0.346203
            elif re.search("all",ss) or re.search("nutime",ss):
                #vcrit = 5000
                part_prop = 1.
                nu_part_time = 0.5
                if re.search("all",ss):
                    nu_part_time = 0.25
            (k, pk_nu, shot) = get_hyb_nu_power(matpow[0], genpk_neutrino, part_prop=part_prop, npart=npart, nu_part_time = nu_part_time, scale=scale, modes=modes)
        except (IOError,FileNotFoundError):
            if not re.search("a$",ss):
                print("Problem",genpk_neutrino)
            (k, pk_nu) = get_nu_power(matpow[0], modes=modes)
            shot = np.zeros_like(k)
    except IndexError:
        (k, pk_nu) = load_genpk(genpk_neutrino)
        #So it matches the binning of the lin resp code.
        rebinned=scipy.interpolate.interpolate.interp1d(k, pk_nu, fill_value='extrapolate')
        k = np.concatenate([[2*math.pi/300,], k])
        pk_nu = rebinned(k)
        #Shot noise
        if re.search("1024",ss):
            shot=(300/1024.)**3*np.ones_like(pk_nu)
        else:
            shot=(300/512.)**3*np.ones_like(pk_nu)
        pk_nu -=shot
    return (k, pk_nu, shot)

def plot_nu_single_redshift_split(scale,ss,fn="nu-split"):
    """Plot all the neutrino power in simulations at a single redshift, splitting the fast and slow components."""
    #Get total neutrino power
    (k, pk_nu, shot) = select_nu_power(scale, ss)
    plt.loglog(k, pk_nu,ls="-", label=labels[ss], color=colors[ss])
    #Plot split power.
    sdir = os.path.join(os.path.join(datadir, ss),"output")
    #Get the modes (hack as we didn't save them before)
    mpk = glob.glob(os.path.join(sdir,"powerspectrum-"+str(scale)+"*.txt"))
    mat = np.loadtxt(mpk[0])
    ii = np.where(mat[:,2] > 0)
    modes = mat[:,2][ii]
    matpow = glob.glob(os.path.join(sdir,"powerspectrum-nu-"+str(scale)+"*.txt"))
    genpk_neutrino = os.path.join(os.path.join(datadir,ss),"output/power-nu-%.4f.txt" % scale)
    (k, pk_nu_slow, pk_nu_fast) = get_hyb_nu_power(matpow[0], genpk_neutrino, npart=512, scale=scale, split=True,modes=modes)
    plt.loglog(k, pk_nu_slow, ls="-.",label=r"Slow $\nu$", color="blue")
    plt.loglog(k, pk_nu_fast, ls=":",label=r"Fast $\nu$", color="black")
    plt.loglog(np.concatenate([[0.005,],k]), np.concatenate([[shot[0],],shot]), color="lightgrey", ls=":")
    plt.text(0.02, 1e-3,"z="+str(np.round(1/scale-1,2)))
    plt.ylim(ymin=1e-5)
    plt.xlim(0.01, 10)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P}_\nu(k)$ (Mpc/h)$^3$")
    plt.legend(frameon=False, loc='upper right',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks-"+fn+"-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_hmf_rel_one(scale, psims=sims, pzerosim = zerosim, rel=True):
    """Plot the halo mass function relative to one simulation."""
    sdir = os.path.join(os.path.join(datadir, pzerosim),"output")
    foftable = os.path.join(sdir,"PIG_00"+scale_to_snap[scale])
    (MMz, dndmz) = HMFFromFOF(foftable, bins=40)
    if not rel:
        plt.loglog(MMz, dndmz, ls="-", label=r"$M_\nu = 0$", color="black")
    for ss in sims:
        sdir = os.path.join(os.path.join(datadir, ss),"output")
        foftable = os.path.join(sdir,"PIG_00"+scale_to_snap[scale])
        try:
            (MMa, dndm) = HMFFromFOF(foftable, bins = 40)
            if rel:
                plt.semilogx(MMa, dndm/dndmz, ls=lss[ss], label=labels[ss], color=colors[ss])
            else:
                plt.loglog(MMa, dndm, ls=lss[ss], label=labels[ss], color=colors[ss])
        except bigfile.pyxbigfile.Error:
            pass
    plt.xlabel(r"Halo Mass ($M_\odot$)")
    if rel:
        plt.ylabel(r"dn/dM (ratio)")
    else:
        plt.ylabel(r"dn/dM ($M^{-1}_\odot \mathrm{Mpc}^{-3}$)")
    plt.ylim(0.9,1.1)
    plt.legend(frameon=False, loc='lower left',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "hmf-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_hmf_rel_tinker(scale, psims=sims, pzerosim = zerosim, rel=True):
    """Plot the halo mass function relative to analytic."""
    sdir = os.path.join(os.path.join(datadir, pzerosim),"output")
    foftable = os.path.join(sdir,"PIG_00"+scale_to_snap[scale])
    (MMz, dndmz) = HMFFromFOF(foftable, bins=40)
    scale_sigma8_0 = {0.02: 0.0219, 0.1: 0.1087, 0.2:0.2164, 0.3333:0.3561, 0.5:0.516, 0.6667: 0.6505, 0.8333: 0.7567, 1:0.8375}
    scale_sigma8_mnu = {0.02: 0.0204, 0.1: 0.986, 0.2:0.1943, 0.3333:0.3173, 0.5:0.4572, 0.6667: 0.5746, 0.8333: 0.6669, 1:0.7372}
    mf = halo_mass_function.HaloMassFunction.tinker_200
    h0 = halo_mass_function.HaloMassFunction(1/scale-1, omega_m=0.288,omega_b=0.0454,hubble=0.7,ns=0.97,omega_l=0.712,sigma8=scale_sigma8_0[1], mass_function=mf)
    hmnu = halo_mass_function.HaloMassFunction(1/scale-1, omega_m=0.288-0.4/96.14/0.7**2,omega_b=0.0454,hubble=0.7,ns=0.97,omega_l=0.712,sigma8=scale_sigma8_mnu[1],mass_function=mf)
    if not rel:
        plt.loglog(MMz, dndmz, ls="-", label=r"$M_\nu = 0$", color="black")
    else:
        plt.semilogx(MMz, dndmz/0.7**4/h0.dndm(MMz*0.7), ls="-", label=r"$M_\nu = 0$", color="black")
    for ss in sims:
        sdir = os.path.join(os.path.join(datadir, ss),"output")
        foftable = os.path.join(sdir,"PIG_00"+scale_to_snap[scale])
        try:
            (MMa, dndm) = HMFFromFOF(foftable, bins = 40)
            if rel:
                plt.semilogx(MMa, dndm/0.7**4/h0.dndm(MMz*0.7), ls=lss[ss], label=labels[ss], color=colors[ss])
            else:
                plt.loglog(MMa, dndm, ls=lss[ss], label=labels[ss], color=colors[ss])
        except bigfile.pyxbigfile.Error:
            pass
    plt.xlabel(r"Halo Mass ($M_\odot$)")
    if rel:
        plt.semilogx(MMz,hmnu.dndm(MMz*0.7)/h0.dndm(MMz*0.7),ls=":",label="Tinker HMF",color="grey")
        plt.ylabel(r"dn/dM (ratio)")
    else:
        plt.ylabel(r"dn/dM ($M^{-1}_\odot \mathrm{Mpc}^{-3}$)")
    plt.ylim(0.,1.5)
    plt.legend(frameon=False, loc='lower left',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "hmf-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_nu_single_redshift(scale,psims=sims,fn="nu"):
    """Plot all the neutrino power in simulations at a single redshift"""
    for ss in psims:
        (k, pk_nu, shot) = select_nu_power(scale, ss)
        plt.loglog(k, pk_nu,ls=lss[ss], label=labels[ss], color=colors[ss])
        plt.loglog(np.concatenate([[0.005,],k]), np.concatenate([[shot[0],],shot]), color="lightgrey", ls=":", alpha = alpha[ss])
    cambdir = os.path.join(os.path.join(datadir, psims[0]),"camb_linear")
    cambmat = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
    cambtrans = os.path.join(cambdir,"ics_transfer_"+scale_to_camb[scale]+".dat")
    (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
    rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
    plt.semilogx(k, rebinned(k),ls=":", label="CAMB", color="black")
    plt.text(0.02, 1e-3,"z="+str(np.round(1/scale-1,2)))
    plt.ylim(ymin=1e-5)
    plt.xlim(0.01, 10)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P}_\nu(k)$ (Mpc/h)$^3$")
    plt.legend(frameon=False, loc='upper right',fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks-"+fn+"-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_nu_single_redshift_rel_camb(scale, ymin=0.9, ymax=1.1):
    """Plot all neutrino powers relative to CAMB"""
    for ss in sims:
        cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
        cambmat = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
        cambtrans = os.path.join(cambdir,"ics_transfer_"+scale_to_camb[scale]+".dat")
        (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
        rebinned=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
        (k, pk_nu, _) = select_nu_power(scale, ss)
        pkfilt = smooth(pk_nu/rebinned(k))
        plt.semilogx(k, pkfilt,ls=lss[ss], label=labels[ss], color=colors[ss])
    plt.ylim(ymin, ymax)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P}_\nu / \mathrm{P}_\nu^\mathrm{CAMB}(k)$")
    plt.legend(frameon=False, loc=0,fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks_nu_camb-"+munge_scale(scale)+".pdf"))
    plt.clf()

def plot_nu_single_redshift_rel_one(scale, psims=sims[1:], pzerosim=sims[0], ymin=0.8,ymax=1.2,fn="rel", camb=False):
    """Plot all neutrino powers relative to one simulation"""
    (k_div, pk_div, _) = select_nu_power(scale, pzerosim)
    rebinned=scipy.interpolate.interpolate.interp1d(k_div,pk_div,fill_value='extrapolate')
    if camb:
        cambdir = os.path.join(os.path.join(datadir, sims[0]),"camb_linear")
        cambmat = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
        cambtrans = os.path.join(cambdir,"ics_transfer_"+scale_to_camb[scale]+".dat")
        (k_nu_camb, pk_nu_camb) = get_camb_nu_power(cambmat, cambtrans)
        pk_camb=scipy.interpolate.interpolate.interp1d(k_nu_camb,pk_nu_camb)
        pkfilt = smooth(pk_camb(k_div)/pk_div)
        plt.semilogx(k_div, pkfilt,ls=":", label="CAMB", color="black")
    for ss in psims:
        (k, pk_nu, _) = select_nu_power(scale, ss)
        pkfilt = smooth(pk_nu/rebinned(k))
        plt.semilogx(k, pkfilt,ls=lss[ss], label=labels[ss], color=colors[ss])
    plt.ylim(ymin,ymax)
    plt.xlim(0.01, 5)
    plt.text(1, 1.07,"z="+str(np.round(1/scale-1,2)))
    plt.legend(frameon=False, loc='upper left',fontsize=12)
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
        plt.semilogx(k, pkfilt,ls=lss[ss], label=labels[ss], color=colors[ss])
    plt.ylim(0.94,1.06)
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{P} / \mathrm{P}^\mathrm{CAMB}(k)$")
    plt.legend(frameon=False, loc=0,fontsize=12)
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
        zero_reb=scipy.interpolate.interpolate.interp1d(zero_k,zero_pk_c, fill_value='extrapolate')
        cambdir = os.path.join(os.path.join(datadir, psims[0]),"camb_linear")
        cambpath = os.path.join(cambdir,"ics_matterpow_"+scale_to_camb[scale]+".dat")
        (k_c, pk_c) = get_camb_power(cambpath)
        plt.semilogx(k_c, pk_c/zero_reb(k_c),ls=":", label="CAMB", color="black")
    for ss in psims:
        (k, pk) = _get_pk(scale, ss)
        if np.size(k) == 0:
            continue
        plt.semilogx(k, pk/rebinned(k),ls=lss[ss], label=labels[ss], color=colors[ss])
    plt.ylim(ymin,ymax)
    plt.xlim(0.01,10)
    plt.xlabel("k (h/Mpc)")
    if len(psims) > 1:
        plt.text(0.05, 0.95,"z="+str(np.round(1/scale -1,2)))
    else:
        plt.text(0.05, 0.99,"z="+str(np.round(1/scale -1,2)))
    plt.ylabel(r"$\mathrm{P}(k)$ ratio")
    plt.legend(frameon=False, loc=0,fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "pks_"+fn+"-"+munge_scale(scale)+str(pzerosim[-1])+".pdf"))
    plt.clf()

def plot_fermi_dirac(Mnu, zz):
    """Plot the fermi-dirac distribution for neutrinos at redshift z
    Argument is total neutrino mass."""
    tnu = 2.7255 * (4/11.)**(1./3) * 1.00328
    bolevk = 8.61734e-5
    colors = ["#8c564b", "blue"]
    for j, Mm in enumerate(Mnu):
        nu_v = bolevk * tnu/ (Mm/3) * (1+zz) * 2.99792e5
        fdk = lambda x: x*x/(np.exp(x)+1)
        xx = np.arange(0, 9*nu_v,50)
        ff = np.zeros_like(xx, dtype=np.float64)
        for i in range(np.size(xx)):
            (fd, _) = scipy.integrate.quad(fdk, 0, xx[i]/nu_v)
            ff[i] = fd / (1.5 * 1.20206)
        plt.plot(xx, ff, "-", label=r"Cum. F-D: $M_\nu = "+str(Mm)+"$ eV", color=colors[j %len(colors)])
    plt.plot(xx, fdk(xx/nu_v), "--", label=r"F-D: $M_\nu = "+str(Mnu[-1])+"$ eV", color="black")
    plt.fill_between(xx, 0, ff, where=xx < 850, facecolor='grey', interpolate=True, alpha=0.5)
    plt.text(400, 0.02, "Slow")
    plt.text(2000, 0.5, "Fast")
    plt.ylim(0,1)
    plt.xlim(0,np.max(xx))
    plt.xlabel(r"$v_\nu$ (km/s)")
    plt.ylabel(r"Probability")
    plt.legend(loc='upper left',frameon=False,fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "fermidirac.pdf"))
    plt.clf()

if __name__ == "__main__":
    plot_fermi_dirac([0.15, 0.4],0)
    plot_crosscorr(1)
    for sc in (0.100, 0.200, 0.3333, 0.500, 0.6667, 0.8333, 1):
        plot_nu_single_redshift_split(sc, ss="b300p512nu0.4hyb850")
        plot_nu_single_redshift(sc)
        plot_nu_single_redshift(sc,checksims,fn="cknu")
        plot_nu_single_redshift(sc,checksims2,fn="cknu2")
        plot_single_redshift_rel_one(sc,psims=sims + [checksims2[1],], ymin=0.6,ymax=1.)
        plot_nu_single_redshift_rel_one(sc, ymin=0.9, ymax=1.1, camb=True)
        plot_single_redshift_rel_one(sc,psims=lowmass,fn="lowmass",ymin=0.92, ymax=1.0)
        plot_nu_single_redshift(sc, psims=lowmass, fn="lowmass_nu")
        plot_nu_single_redshift_rel_one(sc,psims=checksims[:],pzerosim=checksims[0],fn="ckrel",ymin=0.89,ymax=1.1)
        plot_nu_single_redshift_rel_one(sc,psims=checksims2[:],pzerosim=checksims2[0],fn="ckrel2",ymin=0.89,ymax=1.1, camb=True)
        plot_single_redshift_rel_one(sc,psims=[sims[1],sims[2]],pzerosim=sims[0],ymin=0.98,ymax=1.02,camb=False,fn="rel0")
        plot_single_redshift_rel_one(sc,psims=checksims,pzerosim=checksims[0],camb=False,ymin=0.999,ymax=1.001,fn="ckrelh")
        plot_single_redshift_rel_one(sc,psims=checksims2,pzerosim=checksims2[0],camb=False,ymin=0.995,ymax=1.005,fn="ckrel2h")
        plot_single_redshift_rel_one(sc,psims=checksims,fn="ckrel")
        plot_single_redshift_rel_camb(sc)
        plot_nu_single_redshift_rel_camb(sc,ymin=0.95, ymax=1.05)
        plot_single_redshift(sc)
    #This will only work with the full simulation data
    for sc in (0.6667, 0.8333, 1):
        plot_hmf_rel_one(sc, psims=sims[1:], pzerosim=sims[0])
    plot_image(zerosim,8,1)
    plot_image(sims[2],8,1)
    plot_image(sims[2],8,2, colorbar=True)
    plot_image(sims[0],8,1)
    plot_image(sims[0],8,2)
