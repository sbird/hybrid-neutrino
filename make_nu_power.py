"""Make the per-species power spectrum using NBodyKit"""
import os.path as path
import sys
import glob
import numpy
from nbodykit.lab import BigFileCatalog,FFTPower
import timeit

def sptostr(sp):
    """Get a string from a species"""
    if sp == 2:
        return "nu"
    elif sp == 1:
        return "DM"
    return ""

def wrapper(a, b):
    "Wrap np.isin so we can specify arguments more explicitly"""
    t1 = timeit.default_timer()
    #Use binary search as logN
    ind = numpy.searchsorted(b,a, side='left')
    try:
        data = (b[ind] != a)
    except IndexError:
        #This happens when the insertion point is at the end of the array
        i2 = numpy.where(ind >= numpy.shape(b))
        #In this case we assign a new value inside the array, since isin is definitely false.
        ind[i2] -=1
        data = (b[ind] != a)
    t2 = timeit.default_timer()
    print("t: ",t2-t1, "shape: ",numpy.shape(a), numpy.shape(b) )
    return data


def compute_fast_power(output, ICS, vthresh=850, Nmesh=1024, species=2, spec2 = None):
    """Compute the compensated power spectrum from a catalogue."""
    sp = sptostr(species)
    sp2 = sptostr(spec2)
    catnu = BigFileCatalog(output, dataset=str(species)+'/', header='Header')
    outfile = path.join(output,"../power-fast-"+sp+sp2+"-%.4f.txt" % catnu.attrs["Time"][0])
    if path.isfile(outfile):
        return
    catics = BigFileCatalog(ICS, dataset=str(species)+'/', header='Header')
    fast = (catics['Velocity']**2).sum(axis=1) < vthresh**2/catics.attrs["Time"]**3
    fastids = catics["ID"][fast].compute()
    fastids.sort()
    #Note: map_blocks runs elementwise over blocks.
    #So we need to pre-compute fastids: if we do not we will
    #end up checking whether elements in a block of catnu["ID"]
    #are in the equivalent block in fastids.
    select = catnu["ID"].map_blocks(wrapper, fastids, dtype=numpy.bool,chunks = catnu["ID"].chunks)
    catnu[select].to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    if spec2 is not None:
        catcdm = BigFileCatalog(output, dataset=str(spec2)+'/', header='Header')
        catcdm.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
        pkcross = FFTPower(catnu[select], mode='1d', Nmesh=Nmesh,second = catcdm, dk=5.0e-6)
        power = pkcross.power
    else:
        pknu = FFTPower(catnu[select], mode='1d', Nmesh=Nmesh, dk=5.0e-6)
        power = pknu.power
    numpy.savetxt(outfile,numpy.array([power['k'], power['power'].real,power['modes']]).T)
    return power

def compute_power(output, Nmesh=1024, species=2, spec2 = None):
    """Compute the compensated power spectrum from a catalogue."""
    catnu = BigFileCatalog(output, dataset=str(species)+'/', header='Header')
    sp = sptostr(species)
    sp2 = sptostr(spec2)
    outfile = path.join(output,"../power-"+sp+sp2+"-%.4f.txt" % catnu.attrs["Time"][0])
    if path.isfile(outfile):
        return
    catnu.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    if spec2 is not None:
        catcdm = BigFileCatalog(output, dataset=str(spec2)+'/', header='Header')
        catcdm.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
        pkcross = FFTPower(catnu, mode='1d', Nmesh=1024,second = catcdm, dk=5.0e-6)
        power = pkcross.power
    else:
        pknu = FFTPower(catnu, mode='1d', Nmesh=1024, dk=5.0e-6)
        power = pknu.power
    numpy.savetxt(outfile,numpy.array([power['k'], power['power'].real,power['modes']]).T)
    return power

def all_compute(directory):
    """Do computation for all snapshots in a directory"""
    snaps = glob.glob(path.join(directory, "output/PART_*"))
    for ss in snaps:
        compute_power(ss)
    for ss in snaps:
        compute_power(ss,species=1)
        compute_power(ss,species=1,spec2=2)

def all_fast_compute(directory):
    """Do computation for all snapshots in a directory"""
    snaps = glob.glob(path.join(directory, "output/PART_*"))
    snaps.sort()
    snaps.reverse()
    ics = glob.glob(path.join(directory, "ICS/*"))
    for ss in snaps:
        print(ss)
        compute_fast_power(ss,ics)
        compute_fast_power(ss,ics, spec2=1)

if __name__ == "__main__":
#     all_fast_compute(sys.argv[1])
    all_compute(sys.argv[1])
