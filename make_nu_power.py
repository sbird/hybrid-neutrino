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

def compute_power(output, Nmesh=1024, species=2, spec2 = None):
    """Compute the compensated power spectrum from a catalogue."""
    catnu = BigFileCatalog(output, dataset=str(species)+'/', header='Header')
    catnu.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
    if spec2 is not None:
        catcdm = BigFileCatalog(output, dataset=str(spec2)+'/', header='Header')
        catcdm.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
        pkcross = FFTPower(catnu, mode='1d', Nmesh=1024,second = catcdm)
        power = pkcross.power
    else:
        pknu = FFTPower(catnu, mode='1d', Nmesh=1024)
        power = pknu.power
    sp = sptostr(species)
    sp2 = sptostr(spec2)
    numpy.savetxt(path.join(output,"../power-"+sp+sp2+"- %.4f.txt" % catnu.attrs["Time"][0]),numpy.array([power['k'], power['power'].real,power['modes']]).T)
    return pknu

def all_compute(directory):
    """Do computation for all snapshots in a directory"""
    snaps = glob.glob(path.join(directory, "PART_*"))
    for ss in snaps:
        compute_power(ss)
        compute_power(ss,species=1)
        compute_power(ss,species=1,spec2=2)

if __name__ == "__main__":
    all_compute(sys.argv[1])
