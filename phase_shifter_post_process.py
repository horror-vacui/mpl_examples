""" Post-process the phase shifter measurement restuls
"""

import pandas as pd
import numpy as np
import logging, time, datetime, pathlib, subprocess, csv, matplotlib, pathlib
from importlib import reload
import matplotlib.pyplot as plt


reload(logging)
meas_date = time.strftime("%Y_%m_%d")      
t_script_start = datetime.datetime.now().replace(microsecond=0)
s_now = f"{t_script_start:%Y%m%d_%H%M}"
log_file = f"ber_{s_now}"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

# phase change per value
dph = 9.2 # degree

d_csv = "/mnt/home/documents/Measurements/MPW2224/RFFE/"
f_csv = "interferometer_rffe_mpw2224_phase_shift_sensitivity_2_61.25GHz_5uA_20201210_1858.csv"

df = pd.read_csv(d_csv + f_csv)

if True:
    matplotlib.use('pgf')
    pgf_with_custom_preamble = {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif", # use serif/main font for text elements
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        "figure.figsize": (3.3914487339144874, 2.0960305886619515*0.8),
        "axes.labelsize": 8,
        "axes.grid": True,
        "font.size": 7,
        "legend.fontsize": 8,
        "legend.handlelength": 2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1,
        # 'legend.title_fonstize':7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction":"in",
        "ytick.direction":"in",
        "xtick.major.size":1.5,
        "ytick.major.size":1.5,
        "xtick.minor.size":0.25,
        "ytick.minor.size":0.25,
        "grid.alpha": 0.6,
        "lines.markersize": 4,
        # "lines.markeredgecolor": None,
        "savefig.pad_inches":0,
        "savefig.bbox":"tight",
        "savefig.dpi":300,
        "pgf.preamble": r"\usepackage[detect-all,locale=US]{siunitx}\usepackage{amsmath}\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}"
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
    matplotlib.rcParams['axes.unicode_minus'] = False



df2 = df[['det0[V]','det1[V]','det2[V]']]
mean = df2.mean()
m = mean.to_numpy()
det = df2.to_numpy() - m
print(det)

fig, ax = plt.subplots()

for i in range(len(m)):
    y=det[:,i]
    ax.plot(np.arange(len(y))*dph,y*1e3)
ax.set_xlabel(r"Phase change in the path [\si{\degree}]")
ax.set_ylabel(r"V\textsubscript{det} [\si{\mV}]")
ax.set_xlim(90,450)
ax.set_ylim(-100,100)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(15))
ax.grid(True,which='minor', alpha=0.1, axis='both')
ax.grid(True,which='major', alpha=0.7, axis='both')

df = pd.DataFrame(det, columns=('det1','det2','det3'), index=df.index*dph)

def sinusoid(x,A,offset,omega,phase):
     return A*np.sin(omega*x+phase)

from scipy.optimize import curve_fit
l_fit = []
for i in ('det1','det2','det3'):
    par, cov = curve_fit(sinusoid, (df.index*dph).to_numpy(), df[i])
    l_fit.append(par)

ph = np.array([i[-1] for i in l_fit])/np.pi*180
ph.sort()

for ext in ("png", "pgf"):
    f_out = pathlib.Path.home().joinpath("publications", "interferometer", "trunk", "pictures", "phase_shift." + ext)
    fig.savefig(f_out)

