# two y-axis w/ ellipses and arrow towards the axes
# nummerical equation solving
# built-in constant library

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as lines
from scipy.constants import speed_of_light, pi
import sympy as sym

latex=True
if latex:
    matplotlib.use('pgf')
    pgf_with_custom_preamble = {
        "font.family": "serif", # use serif/main font for text elements
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        # Use 10pt font in plots, to match 10pt font in document
        # "axes.labelsize": 10,
        "axes.labelsize": 7,
        "font.size": 9,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pgf.preamble": [
            r"\usepackage{siunitx}",     
            r"\usepackage{amsmath}",      
            r"\usepackage{metalogo}",
            r"\usepackage{unicode-math}",  
            # r"\usepackage{mathspec}",  
            # r"\usepackage{fontspec}",  
            r"\DeclareSIUnit{\Bm}{Bm}",
            r"\DeclareSIUnit{\dBm}{\deci\Bm}",
            r"\sisetup{detect-weight=true, detect-family=true, per-mode=fraction, fraction-function=\tfrac,range-phrase=--, range-units=single}",
            r"\newcommand{\da}{\textsuperscript{$\dagger$}}"
            # r"\setmathfont{xits-math.otf}",
            # r"\setmainfont{DejaVu Serif}", # serif font via preamble
            ]
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
    matplotlib.rcParams['lines.markersize'] = 4
#-----------------------------------------------------------


legend_style = {"frameon":False, "fontsize":6, "handletextpad":0.4, "borderaxespad":0, "ncol":2, "loc":'lower center', "mode":"expand", "handlelength":1, "bbox_to_anchor":(-0.2,1,1.4,0.04)}

# Approximateion of implicit frequency equation with t-line. 3rd order Maclaurin series. # not used in this version, though error was not noticable on the plots
def fosc(eps, Z0, C, l):
    return speed_of_light/(2*np.sqrt(2*eps)*pi*l) * np.sqrt(-3+np.sqrt(9+12*(np.sqrt(eps)*l)/(speed_of_light*Z0*C) ))

def f_lc(L,C):
    return 1/(2*pi*np.sqrt(L*C))

# paramters
e = 7.16
Z0 = np.abs(65.2062487139-1.2056007791j)
l_sonnet = 110e-6
l = l_sonnet
C = 66.3e-15 # 66.3pF from 124pH and 55.5GHz
# C *= 2 # l is referenced for the half resonator

f = sym.Symbol('f', real=True)
# f_tl = sym.Eq(1/(2*pi*C*Z0*sym.tan(pi*l/speed_of_light *f*2*sym.sqrt(e))) - f,0)
# # sol = sym.nsolve(1/(2*pi*C*Z0*sym.tan(pi*l/speed_of_light *f*2*sym.sqrt(e))) - f,f,55e9)
# sol = sym.nsolve(f_tl,f,55e9)


# Plot the tuning range between C_min and C_max
x = np.linspace(40e-15,100e-15,num=50)

L = 124e-12*1.05
# for i in x:
#     print("%.1f pF\t %.1f GHz\t%.1f GHz" % (i*1e15, fosc(eps=e, Z0=Z0, l=l, C=2*i)*1e-9, f_lc(L,i)*1e-9))

fig, ax = plt.subplots(figsize=(3.3914487339144874*0.5*0.85, 2.0960305886619515*4/8*0.85))
ax2 = ax.twinx()

y_lc = np.array([f_lc(L,i)*1e-9 for i in x])
ax.plot(x*1e15,y_lc, label="lumped LC", color="green" )

y_tline = np.array([sym.nsolve(1/(2*pi*2*i*Z0*sym.tan(pi*l/speed_of_light *f*2*sym.sqrt(e))) - f,f,55e9)*1e-9 for i in x])
ax.plot(x*1e15, y_tline, label="T-line", color="purple" )

y_er = (y_lc - y_tline)/y_tline*100 
ax2.plot(x*1e15, y_er, color="grey", linestyle="--",zorder=2, label="difference")

ax.add_artist(mpatches.Ellipse((60,56.6),1.5,3,edgecolor="black",facecolor='none', alpha=0.8, lw=0.5,zorder=2))
ax.plot([60,60],[55.1,53], c="black", alpha=0.8,lw=0.5,zorder=4)
ax.arrow(60,53,-5,0,head_width=1.51, head_length=1,lw=0.5, alpha=0.7,facecolor="black",zorder=4)

ax2.add_artist(mpatches.Ellipse((80,0.73),1.5,0.22,edgecolor="black",facecolor='none', alpha=0.8,zorder=2, lw=0.6))
ax2.arrow(80,0.52,5,0,head_width=0.1, head_length=1,lw=0.4, alpha=0.7,zorder=10,fc="black", ec="black")
ax2.plot([80,80],[0.62,0.52], c="black", alpha=0.8,lw=0.6)

ax.grid()
ax.tick_params(pad=2, length=1.5, direction="in", axis="y")
ax2.tick_params(pad=1, length=1.5, direction="in", axis="y")
ax.tick_params(length=1.5, direction="in", axis="x")
ax.set_xlabel(r"Resonator capacitance $\left [ \si{\femto\farad} \right ]$",labelpad=0)
ax.set_ylabel(r"Frequency $\left[\si{\GHz}\right]$", labelpad=1)
ax2.set_ylabel(r"Difference $\left[\si{\percent}\right]$", labelpad=1)
ax.set_ylim(40,70)
ax2.set_ylim(0.25,2.25)
ax.set_xlim(40,100)
ax.legend(frameon=False, fontsize=7, handlelength=0.8, handletextpad=0.35, labelspacing=0.3, bbox_to_anchor=(-0.05,1,1.1,0.04), ncol=2, loc='lower center', borderaxespad=0, mode='expand')

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))

for ext in ["png","pgf"]:
    fig.savefig("tl_vs_lc." + ext, bbox_inches='tight', pad_inches = 0, dpi=600)


