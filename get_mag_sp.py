import argparse
import skrf as rf
from pathlib import Path
from scipy.interpolate import interp1d
from numpy import log10


parser = argparse.ArgumentParser(description = 'script to return the amplitude of a circuit at a given freqeuncy based on its measured S-paramter file')
parser.add_argument("file", help="The S-paramter file", type=Path)
parser.add_argument("-f", help="Frequency.", type=float)
parser.add_argument("-pin", help="Input port. Default is 1.", type=int, default=1)
parser.add_argument("-pout", help="Output port. Default is 2.", type=int, default=2)
parser.add_argument('-l','--linear', action='store_true', help="Prints the results in linear value. Otherwise the result is in dB")
args = parser.parse_args()

# skrf.Network can not accept a Path type as argument
nw = rf.Network(str(args.file.resolve()))
sp = abs(nw.s[:,(args.pout-1),(args.pin-1)])

f = interp1d(nw.f, sp, kind="cubic")
ret = f(args.f)
txt = "S%d%d: " % (args.pout, args.pin)
if args.linear:
    print(txt + f"{ret:.2}")
else:
    db = 20*log10(ret)
    print(txt + f"{db:.2f} dB")
