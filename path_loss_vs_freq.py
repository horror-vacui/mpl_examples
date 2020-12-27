import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, abs, log10
import matplotlib
from scipy.interpolate import interp1d

if True:
    matplotlib.use('pgf')
    pgf_with_custom_preamble = {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif", # use serif/main font for text elements
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        "figure.figsize": (3.3914487339144874*1.1, 2.0960305886619515*0.8*1.1),
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

def path_loss(f,d):
    c = 299792458 # m/s
    return (4*pi*d*f/c)**2



# All units are in SI
d = 10
x = np.linspace(100e6,2.5e9,200)
y = 10*log10(path_loss(d=d, f=x))

fig, ax = plt.subplots()
ax.grid(True,which='minor', alpha=0.1, axis='both')
ax.grid(True,which='major', alpha=0.7, axis='both')
ax.set_xlabel("Frequency [\si{\GHz}]", labelpad=2)
ax.set_ylabel("Path loss [\si{dB}]", labelpad=2)
ax.tick_params(axis='both', which='major', pad=2)
ax.set_ylim(30,65)

ax.plot(x*1e-9, y)
xlim= ax.get_xlim()
ylim= ax.get_ylim()
f_pl = interp1d(x,y, kind='cubic')
x0=868e6
x0g = x0*1e-9
y0 = f_pl(x0)
ax.scatter(x0g,y0, marker="o", color="gray")
ax.hlines(y0,xlim[0],x0g, color="gray", linestyle="--")
ax.vlines(x0g,ylim[0],y0, color="gray", linestyle="--")
ax.text(x0g-0.4, ylim[0]+1,s=r"\SI{%.0f}{\MHz}" % (x0g*1e3), color="gray")
ax.text(xlim[0]+0.05, y0+0.5,s="\SI{%.1f}{\dB}" % y0, color="gray")
x0=2.4e9
x0g = x0*1e-9
y0 = f_pl(x0)
ax.scatter(x0g,y0, marker="o", color="gray")
ax.hlines(y0,xlim[0],x0g, color="gray", linestyle="--")
ax.vlines(x0g,ylim[0],y0, color="gray", linestyle="--")
ax.text(x0g-0.37, ylim[0]+1,s="\SI{%.1f}{\GHz}" % x0g, color="gray")
ax.text(xlim[0]+0.05, y0+0.5,s="\SI{%.1f}{\dB}" % y0, color="gray")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
for ext in ['png']:
    fig.savefig("path_loss_vs_freq_at10m." + ext)
