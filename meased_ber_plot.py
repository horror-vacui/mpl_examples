import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, abs, sqrt, exp
import matplotlib

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
        "pgf.preamble": r"\usepackage[detect-all,locale=US]{siunitx}\usepackage{amsmath}\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}\DeclareSIUnit{\Bm}{Bm} \DeclareSIUnit{\dBm}{\deci\Bm}"
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
    matplotlib.rcParams['axes.unicode_minus'] = False

def ber_snr(x):
    return 1/sqrt(2*pi) * exp((-x**2)/2)/x

cable_loss = 6.52 + 1.05 # 2nd is the loss on the PCB
prx = [i-cable_loss for i in (-22, -21, -20, -19)] # these are the set up output power of the signal generator
ber = (1.4e-2, 6.6e-4, 9.4e-6, 1e-7) # measured BER


def my_formatter(x, pos):
    if x.is_integer():
        # return f"{x:.0}"
        return str(int(x))
    else:
        return f"{x:.1f}"
###############################################

fig, ax = plt.subplots()
ax.grid(True,which='minor', alpha=0.1, axis='both')
ax.grid(True,which='major', alpha=0.7, axis='both')
ax.set_xlabel(r"Received power [\si{\dBm}]", labelpad=2)
ax.set_ylabel(r"Bit error rate [ ]", labelpad=2)
ax.tick_params(axis='both', which='major', pad=2)
ax.set_xlim(-22.2-cable_loss, -19.8-cable_loss)
ax.set_ylim(bottom=1e-6, top=2e-2)
ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numdecs=6))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(my_formatter))
# markers
# m = ['o','s','^','v','x']
# me = 50
# for i in range(2,7):
#     y = np.array([worst_sum(i,x,l) for l in l_l])
#     ax.plot(l_l, y, label=i, marker=m[i-2], markevery=(int(me*(i-1)/3), me))
ax.plot(prx, ber, marker="o")
ax.set_yscale('log')
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(my_formatter))
for ext in ("png", "pgf"):
    fig.savefig("meas_ber." + ext)
