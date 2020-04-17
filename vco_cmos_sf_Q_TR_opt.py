import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as lines
from scipy.constants import pi
import sympy as sym
import pandas as pd
from vcsv_parser import vcsv_cols
import logging

sim_dir = "/home/zoltan/publications/vco_cmos_sf/bin/"
f_vcsv = sim_dir + "varactor_Q_TR_vs_L_2.vcsv"    

logger = logging.getLogger("pn_plot_all")
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

latex=True
# latex=False
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
        "legend.title_fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pgf.preamble": [
            "\\usepackage{siunitx}",         # load additional packages
            "\\usepackage{amsmath}",         # load additional packages
            "\\usepackage{metalogo}",
            "\\usepackage{unicode-math}",  # unicode math setup
            r"\DeclareSIUnit{\Bm}{Bm}",
            r"\DeclareSIUnit{\dBm}{\deci\Bm}",
            r"\sisetup{detect-weight=true, detect-family=true, per-mode=fraction, fraction-function=\tfrac,range-phrase=--, range-units=single}",
            r"\newcommand{\da}{\textsuperscript{$\dagger$}}"
            r'\mathchardef\mhyphen="2D'
            # r"\setmathfont{xits-math.otf}",
            # r"\setmainfont{DejaVu Serif}", # serif font via preamble
            ]
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
    matplotlib.rcParams['lines.markersize'] = 4
    matplotlib.rcParams['axes.unicode_minus'] = False


col_names_all = vcsv_cols(f_vcsv,'L')
col_names = [x for i, x in enumerate(col_names_all) if col_names_all.index(x) == i]
col_idx = [i for i, x in enumerate(col_names_all) if col_names_all.index(x) == i]
opt_vcsv = {'header':None, 'skiprows':6, 'dtype':np.float64, 'usecols':col_idx,'names':col_names} 
legend_style = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":3, "loc":'lower center', "mode":"expand", "handlelength":1, "bbox_to_anchor":(-0.1,1,1.2,0.02)}

C_fix = 28.8e-15 # note varactor in the Q testbench needs to be taken 2x to have the same size as in the vco_cmos_sf circuit

def TR_osc(C_var, TR_var, C_fix):
    # TR_var = ((ymax(Cin) - ymin(Cin)) / Cmid)
    # Cmid = ((ymax(Cin) + ymin(Cin)) / 2)
    Cmin = C_var * (1-TR_var/2)
    Cmax = C_var * (1+TR_var/2)
    C_osc_max = (C_fix + Cmax)
    C_osc_min = (C_fix + Cmin)
    f_max = 1/np.sqrt(C_osc_min)
    f_min = 1/np.sqrt(C_osc_max)
    df = f_max - f_min
    f0 = (f_max + f_min)/2
    TR_osc = df/f0
    return TR_osc

df = pd.read_csv(f_vcsv, **opt_vcsv)
df['TR_osc'] = TR_osc(C_var=df.Cmid, TR_var=df.TR_var, C_fix=C_fix)
df['QTR']=df.Q_Vmid * df.TR_osc
df['QTR2']=(df.Q_Vmid * df.TR_osc)**2
df['TR_osc'] *= 100 # scale to percent
df['L']*=1e9
logger.info(df.columns)


fig, ax = plt.subplots(figsize=(3.3914487339144874*0.5*0.85, 2.0960305886619515*4/8*0.85))
ax2 = ax.twinx()
# ax.plot(df.L, df.Q_Vmid, color='#377eb8', label="Q", marker="o", markevery=3, zorder=1)
# ax2.plot(df.L, df.TR_osc, color='#4daf4a', label="TR", marker="s", markevery=(1,3), zorder=2)
# ax.plot(df.L, df.QTR2, color='#e41a1c', label="$(Q\!\cdot\!TR)^2$", marker="v", markevery=(2,3), zorder=10)
# remove the markers
ax.plot(df.L, df.Q_Vmid, color='#377eb8', label="Q", zorder=1)
ax2.plot(df.L, df.TR_osc, color='#4daf4a', label="TR", zorder=2)
ax.plot(df.L, 10*np.log10(df.QTR2), color='#e41a1c', label="$(Q\!\cdot\!TR)^2$", zorder=10)
ax.grid()
ax.set_xlim(20,200)
ax2.set_ylim(10,45)
ax.set_ylim(-6,8)
ax.tick_params(pad=2, length=1.5, direction="in",axis="y")
ax2.tick_params(pad=1, length=1.5, direction="in",axis="y")
ax.tick_params(length=1.5, direction="in",axis="x")
ax.set_xlabel(r"Channel length $\left[\si{\um}\right]$",labelpad=0)
ax.set_ylabel(r"$Q \left[\, \right]$; $\left ( Q\!\cdot\!TR \right )^2\left [\si{\dB} \right ]$", labelpad=1)
ax2.set_ylabel(r"Tuning range\,$\left[\si{\percent}\right]$", labelpad=1)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base=30,offset=0))

handles, labels   = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles += handles2
labels  += labels2

leg = ax.legend(handles=handles, labels=labels,  **legend_style) 

ax2.add_artist(mpatches.Ellipse((190,35),6,5,edgecolor="black",facecolor='none', alpha=0.8, lw=0.5,zorder=2))
ax2.plot([190,190],[32.5,30], c="black", alpha=0.8,lw=0.5,zorder=4)
ax2.arrow(190,30,-15,0,head_width=1.75, head_length=5,lw=0.5, alpha=0.7,facecolor="black",zorder=4)

ax2.add_artist(mpatches.Ellipse((80,24),6,5,edgecolor="black",facecolor='none', alpha=0.8, lw=0.5,zorder=2))
ax2.plot([80,80],[21.5,20], c="black", alpha=0.8,lw=0.5,zorder=4)
ax2.arrow(80,20,15,0,head_width=1.75, head_length=5,lw=0.5, alpha=0.7,facecolor="black",zorder=4)
for ext in ["png","pgf"]:
    fig.savefig("QTR_optimization." + ext, bbox_inches='tight', pad_inches = 0, dpi=300)

