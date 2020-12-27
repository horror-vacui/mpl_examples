import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, abs
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
        "pgf.preamble": r"\usepackage[detect-all,locale=US]{siunitx}\usepackage{amsmath}\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}"
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
    matplotlib.rcParams['axes.unicode_minus'] = False

l_tl=0.184
l_tl=0.284
l_tl=0.384

l_ym = []
l_xm = []
l_l = np.linspace(0.01,0.5,200)
x = np.linspace(0, 0.5, 200)
# for l in l_l:
#     y = abs(cos(x*2*pi)) + abs(cos(2*pi*(l-x)))
#     ymin = y.min()
#     xmin = x[np.where( y == y.min())]
#     l_ym.append(ymin)
#     l_xm.append(xmin)
#     # plt.plot(x,y, label="%.2f"%l)

# plt.plot(l_l,l_ym)    
# plt.xlabel("Line length [$l/\lambda$]")
# plt.ylabel("Worst case sum")

# # print(np.amin(y))
# # print(np.min(y))

# plt.grid()
# plt.legend()
# plt.show()
# plt.close()

###############################################
def pd_sum(n, x, l):
    s = 0
    for i in range(n):
        s += abs(cos(2*pi*(i*l/(n-1) - x)))
    return s

def sum_minmax(n, x, l):
    """ Worst case of the standing wave for 'n' detectors. """
    i = pd_sum(n,x,l)
    print(i.min(), i.max())
    return (i.min(), i.max())

def my_formatter(x, pos):
    if x.is_integer():
        # return f"{x:.0}"
        return str(int(x))
    else:
        return f"{x:.1}"
###############################################

fig, ax = plt.subplots()
ax.grid(True,which='minor', alpha=0.1, axis='both')
ax.grid(True,which='major', alpha=0.7, axis='both')
ax.set_xlabel("Electrical length of the transmission line", labelpad=2)
ax.set_ylabel("Worst case detectection", labelpad=2)
ax.tick_params(axis='both', which='major', pad=2)
ax.set_xlim(0,0.5)
ax.set_ylim(bottom=0,top=6)
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
# markers
m = ['o','s','^','v','x']
me = 50
for i in range(2,7):
    ymin = np.array([sum_minmax(i,x,l)[0] for l in l_l])
    ax.plot(l_l, ymin, label=i, marker=m[i-2], markevery=(int(me*(i-1)/3), me))
ax.set_prop_cycle(None)
for i in range(2,7):
    ymax = np.array([sum_minmax(i,x,l)[1] for l in l_l])
    # ax.plot(l_l, ymax, marker=m[i-2], markevery=(int(me*(i-1)/3), me), linestyle="--")
    ax.plot(l_l, ymax, markevery=(int(me*(i-1)/3), me), linestyle=":")
legend_style = {"frameon":False,  "handletextpad":0.4, "borderaxespad":0, "ncol":5, "loc":'lower center', "mode":"expand", "handlelength":1}
ax.legend(bbox_to_anchor=(0.1, 1.0, 0.9, 0.04), **legend_style)
ax.annotate("$N$:", xy=(0.01,1.05), xycoords="axes fraction")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(my_formatter))
for ext in ("png", "pgf"):
    fig.savefig("interferometer_calc." + ext)
###############################################
if False:
    n = 4
    a=0
    color = ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f","#bf5b17"]
    for l_tl in np.linspace(0.25,0.5,21):
        for i in range(n):
            plt.plot(x, abs(cos(2*pi*(i*l_tl/(n-1) - x))))
        plt.plot(x, pd_sum(n,x,l_tl), label=l_tl)

        plt.xlabel("SW peak location")
        plt.ylabel("")
        plt.grid()
        plt.savefig(f"./interfero_n4_tl{l_tl:.2}sum.png")
