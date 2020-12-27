import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, abs
import pandas as pd
import matplotlib, logging, pathlib
from vcsv_parser import vcsv_cols

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

if True:
    matplotlib.use('pgf')
    pgf_with_custom_preamble = {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif", # use serif/main font for text elements
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        "figure.figsize": (3.3914487339144874*0.5, 2.0960305886619515*0.8),
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

comment = "20p_100n_50ohm"
dir_out = f"{pathlib.Path.home()}/publications/interferometer/trunk/pictures/"
f_dir = "/mnt/home/documents/Design/pictures/"
f_bode = f_dir + "pd_buffer_bode_20p_100n_50.vcsv"
f_tran = f_dir + "pd_buffer_tran_20p_100n_50.vcsv"
f_bode_noload = f_dir + "pd_buffer_bode_100p_50.vcsv"
f_tran_noload = f_dir + "pd_buffer_tran_100p_50.vcsv"

col_names_all = vcsv_cols(f_bode,'f')
col_names = [x for i, x in enumerate(col_names_all) if col_names_all.index(x) == i]
col_idx = [i for i, x in enumerate(col_names_all) if col_names_all.index(x) == i]
opt_vcsv = {'header':None, 'skiprows':6, 'dtype':np.float64, 'usecols':col_idx,'names':col_names} 
# legend_style = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":3, "loc":'lower center', "mode":"expand", "handlelength":1, "bbox_to_anchor":(-0.1,1,1.2,0.02)}
legend_style = {"frameon":False, "fontsize":8, "handletextpad":0.4, "borderaxespad":0, "ncol":3, "loc":'lower center', "mode":"expand", "handlelength":2, "bbox_to_anchor":(0,1,1,0.02)}

df_bode = pd.read_csv(f_bode, **opt_vcsv)
df_bode.rename(columns={'Loop Gain Phase':'p', 'Loop Gain dB20':'g'}, inplace=True)
df_bode = df_bode[(df_bode.f >=100) & (df_bode.f <=1e7)]

df_bode2 = pd.read_csv(f_bode_noload, **opt_vcsv)
df_bode2.rename(columns={'Loop Gain Phase':'p', 'Loop Gain dB20':'g'}, inplace=True)
df_bode2 = df_bode2[(df_bode2.f >=100) & (df_bode2.f <=1e7)]

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.grid(True,which='minor', alpha=0.1, axis='both')
ax.grid(True,which='major', alpha=0.7, axis='both')
ax.set_xlabel("Frequency [\si{\Hz}]", labelpad=2)
ax.set_xscale('log')
ax.set_ylabel("Gain [\si{\dB}]", labelpad=2)
ax2.set_ylabel("Phase [\si{\degree}]", labelpad=2)
ax.tick_params(axis='both', which='major', pad=2)
# ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(15))
# ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(30))
# ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[1],numticks=9))
# ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs='all'))

l1 = ax.plot( df_bode.f, df_bode.g, marker="^", markevery=(0,8), color="#00adef", label="gain")
l2 = ax.plot( df_bode2.f, df_bode2.g, marker="^", markevery=(0,8), linestyle="--", color="#00adef")
ax2.plot([],[])
l3 = ax2.plot(df_bode.f, df_bode.p, marker="v", markevery=(4,8), color="#ed1c24", label="phase")
l4 = ax2.plot(df_bode2.f, df_bode2.p, marker="v", markevery=(6,8), linestyle="--", color="#ed1c24")

labels = [l.get_label() for l in l1+l3]

ax.set_ylim(-10,80)
ax2.set_ylim(0,180)

ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(7))
# ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(7))
ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(7))
ax_tick = np.array([0,15,30,45,60,75])
ax2.set_yticks((ax_tick+10)*2)
ax2.set_yticklabels([int(i) for i in (ax_tick+10)*2])
legend_style = {"frameon":False,  "handletextpad":0.4, "borderaxespad":0, "loc":'upper right', "handlelength":1}
ax.legend(l1+l3, labels, **legend_style)

for ext in ("png", "pgf"):
    f_out = dir_out + "pd_output_buffer_bode" + comment + "." + ext
    logger.info(f_out)
    fig.savefig(f_out)

########################################################
col_names_all = vcsv_cols(f_tran,'t')
col_names = [x for i, x in enumerate(col_names_all) if col_names_all.index(x) == i]
col_idx = [i for i, x in enumerate(col_names_all) if col_names_all.index(x) == i]
opt_vcsv = {'header':None, 'skiprows':6, 'dtype':np.float64, 'usecols':col_idx,'names':col_names} 
df_tran = pd.read_csv(f_tran, **opt_vcsv)
df_tran.rename(columns={'VT("/pad")':'out', 'VT("/vinp")':'vin'}, inplace=True)
df_tran2 = pd.read_csv(f_tran_noload, **opt_vcsv)
df_tran2.rename(columns={'VT("/pad")':'out', 'VT("/vinp")':'vin'}, inplace=True)
logger.info(df_tran.columns)

fig, ax = plt.subplots()
ax.grid(True,which='minor', alpha=0.1, axis='both')
ax.grid(True,which='major', alpha=0.7, axis='both')
ax.set_xlabel(r"Time [\si{\micro\second}]", labelpad=2)
ax.set_ylabel(r"Voltage [\si{\volt}]", labelpad=2)
ax.tick_params(axis='both', which='major', pad=2)
ax.set_ylim(0.585,0.655)
ax.set_xlim(0,2)
# ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(7))
# ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(8))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
# ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=[1],numticks=9))
# ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs='all'))

ax.plot( df_tran.t*1e6, df_tran.out, markevery=(0,160), color="#00adef")
# ax.plot( df_tran.t*1e6, df_tran.vin, marker="^", markevery=(40,160), color="#ed1c24")
ax.plot( df_tran.t*1e6, df_tran.vin, markevery=(40,160), color="#ed1c24")
# ax.plot( df_tran2.t*1e6, df_tran2.out, marker="^", markevery=(80,160), color="#00adef", linestyle="--")
ax.plot( df_tran2.t*1e6, df_tran2.out, markevery=(80,160), color="#00adef", linestyle="--")
# ax.plot( df_tran2.t*1e6, df_tran2.vin, marker="^", markevery=(120,160), color="#ed1c24", linestyle="--")

for ext in ("png", "pgf"):
    f_out = dir_out + "pd_output_buffer_tran" + comment + "." + ext
    logger.info(f_out)
    fig.savefig(f_out)
