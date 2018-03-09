"""Make the per-species power spectrum using NBodyKit"""
import os.path as path
import sys
import glob
import numpy
from nbodykit.lab import BigFileCatalog,FFTPower

def sptostr(sp):
    """Get a string from a species"""
    if sp == 2:
        return "nu"
    elif sp == 1:
        return "DM"
    return ""

def compute_fast_power(output, ICS, vthresh=850, Nmesh=1024, species=2, spec2 = None):
    """Compute the compensated power spectrum from a catalogue."""
    sp = sptostr(species)
    sp2 = sptostr(spec2)
    catnu = BigFileCatalog(output, dataset=str(species)+'/', header='Header')
    outfile = path.join(output,"../power-fast-"+sp+sp2+"-%.4f.txt" % catnu.attrs["Time"][0])
    if path.isfile(outfile):
        return
    catics = BigFileCatalog(ICS, dataset=str(species)+'/', header='Header')
    fast = numpy.sum(catics['Velocity']**2, axis=1) > vthresh**2/catics.attrs["Time"]**3
    fastids = catics["ID"][fast]
    select = numpy.isin(catnu["ID"], fastids)
    if spec2 is not None:
        catcdm = BigFileCatalog(output, dataset=str(spec2)+'/', header='Header')
        pkcross = FFTPower(catnu[select], mode='1d', Nmesh=Nmesh,second = catcdm)
        power = pkcross.power
    else:
        pknu = FFTPower(catnu[select], mode='1d', Nmesh=Nmesh)
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
        pkcross = FFTPower(catnu, mode='1d', Nmesh=1024,second = catcdm)
        power = pkcross.power
    else:
        pknu = FFTPower(catnu, mode='1d', Nmesh=1024)
        power = pknu.power
    numpy.savetxt(outfile,numpy.array([power['k'], power['power'].real,power['modes']]).T)
    return power

def all_compute(directory):
    """Do computation for all snapshots in a directory"""
    snaps = glob.glob(path.join(directory, "PART_*"))
    for ss in snaps:
        compute_power(ss)
        compute_power(ss,species=1)
        compute_power(ss,species=1,spec2=2)

def all_fast_compute(directory):
    """Do computation for all snapshots in a directory"""
    snaps = glob.glob(path.join(directory, "output/PART_*"))
    ics = glob.glob(path.join(directory, "ICS/*"))
    for ss in snaps:
        compute_fast_power(ss,ics)
        compute_fast_power(ss,ics, species=1,spec2=2)

if __name__ == "__main__":
    all_fast_compute(sys.argv[1])
