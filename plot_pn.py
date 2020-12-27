import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re,os,logging,matplotlib,csv
from scipy.signal import savgol_filter
import pdb

do_etspc = False
# do_etspc = True
do_elvt = False
do_elvt = True
# do_slvt = True
do_slvt = False

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)

IEEE_width = 516 #pt
d_ls = {10: (0,(1,3)), 24: 'dotted', 61.25:'solid', 70:'dashed'} # linestyles

df_sg_61p25 = pd.read_csv('Keysight_E8257D_61.25GHz_8dBm.dat', skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";") 
df_sg_67    = pd.read_csv('Keysight_E8257D_67GHz_8dBm.dat', skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")  
df_sg_24    = pd.read_csv('Keysight_E8257D_24GHz_4p6dBm.dat', skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")  # Remeasure? No peak supposed to be there if PC is turned off.
df_sg_10    = pd.read_csv('Keysight_E8257D_10GHz_3dBm.dat', skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")

if False: # old, first measurement. Without the source meter unit    
    df_et_0p5_0p5        = pd.read_csv('etspc_0p5_2_m1_0p44_6629only_24GHz_4p6dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p8_0          = pd.read_csv('etspc_0p8_0_0_0p54_6629only_24GHz_4p6dBm.dat',     skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p8_2          = pd.read_csv('etspc_0p8_2_m1_0p44_6629only_61p25GHz_8dBm.dat',   skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p9_0          = pd.read_csv('etspc_0p9_0_0_0p54_6629only_24GHz_4p6dBm.dat',     skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p9_2_61p25    = pd.read_csv('etspc_0p9_2_-1_0p44_6629only_61p25GHz_8dBm.dat',   skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p9_2_61p25_54 = pd.read_csv('etspc_0p9_2_-1_0p54_6629only_61p25GHz_8dBm.dat',   skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p9_2_70       = pd.read_csv('etspc_0p9_2_-1_0p44_6629only_70GHz_8dBm.dat',      skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p9_2_70_8p5   = pd.read_csv('etspc_0p9_2_-1_0p44_6629only_70GHz_8p5dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p9_3_61p25    = pd.read_csv('etspc_0p9_3_-1p5_0p54_6629only_61p25GHz_8dBm.dat', skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
    df_et_0p9_3_70       = pd.read_csv('etspc_0p9_3_-1p5_0p54_6629only_70GHz_8dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")

df_sg_10           = pd.read_csv('Keysight_10G_3dBm.DAT',    skiprows=78, names=['df','pn'],usecols=[0,1],delimiter=';')
df_sg_24           = pd.read_csv('Keysight_24G_4p4dBm.DAT',  skiprows=78, names=['df','pn'],usecols=[0,1],delimiter=';')
df_sg_61           = pd.read_csv('Keysight_61p25G_8dBm.DAT', skiprows=78, names=['df','pn'],usecols=[0,1],delimiter=';')

df_et_0p5_3        = pd.read_csv('etspc_24G_4p4dBm_0p5V_0p225V_3V_m1V.DAT',     skiprows=78, names=["df","pn"], usecols=[0,1], delimiter=";")
df_et_0p9_0        = pd.read_csv('etspc_24G_4p4dBm_0p9V_0p5V_0V_0V.DAT',        skiprows=78, names=["df","pn"], usecols=[0,1], delimiter=";")
df_et_0p8_2        = pd.read_csv('etspc_61p25G_8dBm_0p8V_0p475V_2V_m1V.DAT',    skiprows=78, names=["df","pn"], usecols=[0,1], delimiter=";")
df_et_0p9_2        = pd.read_csv('etspc_61p25G_8dBm_0p9V_0p525V_2V_m1V.DAT',    skiprows=78, names=["df","pn"], usecols=[0,1], delimiter=";")
df_et_0p9_2_70     = pd.read_csv('etspc_70G_8dBm_0p9V_0p5V_2V_m1V_001.DAT',     skiprows=78, names=["df","pn"], usecols=[0,1], delimiter=";")
df_et_0p9_3_70     = pd.read_csv('etspc_70G_8dBm_0p9V_0p5V_3V_m1V.DAT',         skiprows=78, names=["df","pn"], usecols=[0,1], delimiter=";")


df_et_sim_0p5_3    = pd.read_csv('/mnt/home/documents/Design/pictures/tspc_div/etspc_pn_24GHz_0p5V_0p225V_0_0.vcsv',     skiprows=6, names=["df","pn"], usecols=[0,1])  # wrong backbias voltages
df_et_sim_0p9_0    = pd.read_csv('/mnt/home/documents/Design/pictures/tspc_div/etspc_pn_61p25GHz_0p9V_0p45V_0_0.vcsv', skiprows=6, names=["df","pn"], usecols=[0,1])
df_et_sim_0p8_2    = pd.read_csv('/mnt/home/documents/Design/pictures/tspc_div/etspc_pn_61p25GHz_0p8V_0p375V_2_-1.vcsv', skiprows=6, names=["df","pn"], usecols=[0,1])
df_et_sim_0p9_2    = pd.read_csv('/mnt/home/documents/Design/pictures/tspc_div/etspc_pn_61p25GHz_0p9V_0p45V_2_-1.vcsv', skiprows=6, names=["df","pn"], usecols=[0,1])
df_et_sim_0p9_2_70 = pd.read_csv('/mnt/home/documents/Design/pictures/tspc_div/etspc_pn_70GHz_0p9V_0p45V_2_-1.vcsv',     skiprows=6, names=["df","pn"], usecols=[0,1])
df_et_sim_0p9_3_70 = pd.read_csv('/mnt/home/documents/Design/pictures/tspc_div/etspc_pn_70GHz_0p9V_0p45V_3_-1.vcsv',     skiprows=6, names=["df","pn"], usecols=[0,1])
# print(df_et_0p5_0p5.head())
# pdb.set_trace()

df_el_0p5_0p5        = pd.read_csv('elvt_0p5_0p5_0_0p3_6629only_10GHz_3dBm_v2.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
df_el_0p5_2          = pd.read_csv('elvt_0p5_2_m1_0p3_6629only_24GHz_4p6dBm_v2.dat',   skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
df_el_0p8_0p8_44     = pd.read_csv('elvt_0p8_0p8_0_0p44_6629only_24GHz_4p6dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
df_el_0p8_0p8_48     = pd.read_csv('elvt_0p8_0p8_0_0p48_6629only_24GHz_4p6dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
df_el_0p8_0p8_56     = pd.read_csv('elvt_0p8_0p8_0_0p56_6629only_24GHz_4p6dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
df_el_0p8_0p8_58     = pd.read_csv('elvt_0p8_0p8_0_0p58_6629only_61p25GHz_8dBm.dat',   skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
df_el_0p8_2_48       = pd.read_csv('elvt_0p8_2_m1_0p48_6629only_61p25GHz_8dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
df_el_0p8_2_58       = pd.read_csv('elvt_0p8_2_m1_0p58_6629only_61p25GHz_8dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")
df_el_0p9_2          = pd.read_csv('elvt_0p9_2_m1_0p54_6629only_61p25GHz_8dBm.dat',    skiprows=80, names=["df","pn"], usecols=[0,1], delimiter=";")

df_sl_0p4_2   = pd.read_csv('slvt_10G_3dBm_0p4V_0p2V_2V_0V.DAT',     skiprows=78, names=['df','pn'], usecols=[0,1], delimiter=';')
df_sl_0p4_3   = pd.read_csv('slvt_10G_3dBm_0p4V_0p14V_3V_m1V.DAT',    skiprows=78, names=['df','pn'], usecols=[0,1], delimiter=';')
df_sl_0p5_3   = pd.read_csv('slvt_24G_4p4dBm_0p5V_0p2V_3V_m1V.DAT',  skiprows=78, names=['df','pn'], usecols=[0,1], delimiter=';')
df_sl_0p5_2   = pd.read_csv('slvt_24G_4p4dBm_0p5V_0p25V_2V_m1V.DAT', skiprows=78, names=['df','pn'], usecols=[0,1], delimiter=';')
df_sl_0p8_0p8 = pd.read_csv('slvt_24G_4p4dBm_0p8V_0p4V_0p8V_0V.DAT', skiprows=78, names=['df','pn'], usecols=[0,1], delimiter=';')
df_sl_0p8_3   = pd.read_csv('slvt_61p25_8dBm_0p8V_0p4V_3V_m1V.DAT',  skiprows=78, names=['df','pn'], usecols=[0,1], delimiter=';')
df_sl_0p9_3   = pd.read_csv('slvt_61p25_8dBm_0p9V_0p4V_3V_m1V.DAT',  skiprows=78, names=['df','pn'], usecols=[0,1], delimiter=';')
markers=[ "^","v","s","D","o","x"]
d_vindc  = { 0.5:0.23, 0.8:0.44, 0.9:0.54}
d_vdd_ax = {0.5:0 , 0.8:1, 0.9:2}
d_color  = { # color dict. VDD, VBN, VBP
        (0.5,0,0): "orange",
        (0.5,0.5,0.0):   "green",
        (0.5,2,-1.0):    "blue",
        (0.5,3,-1.5):  "red",

        (0.8,0.0,0.0):     "orange",
        (0.8,0.8,0):   "green",
        (0.8,2,-1.0):    "blue",
        # (0.8,3,0):    "purple",
        (0.8,3,-1.5):  "red",

        (0.9,0.0,0.0):     "orange",
        (0.9,0.9,0):   "green",
        (0.9,2,-1):    "blue",
        # (0.9,3,-1.5):     "purple",
        (0.9,3,-1.5):  "red",
        }

d_style = { # color dict. VDD, VBN, VBP
        (0.5,3,-1):    ("red","^"),
        (0.8,2,-1.0):  ("blue","s"),
        (0.9,0,0):     ("orange","v"),
        (0.9,2,-1):    ("green","x"),
        (0.9,3,-1):    ("purple","o")
        }

latex = True
# latex = False
# matplotlib.use('TkAgg')
# matplotlib.rcParams['interactive']=True

if latex:
    matplotlib.use('pgf')
    pgf_with_custom_preamble = {
        "font.family": "serif", # use serif/main font for text elements
        "text.usetex": True,    # use inline math for ticks
        "pgf.rcfonts": False,   # don't setup fonts from rc parameters
        # Use 10pt font in plots, to match 10pt font in document
        # "axes.labelsize": 10,
        "axes.labelsize": 8,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pgf.preamble": [
            "\\usepackage{siunitx}",         # load additional packages
            "\\usepackage{amsmath}",         # load additional packages
            "\\usepackage{metalogo}",
            "\\usepackage{unicode-math}",  # unicode math setup
            r"\setmathfont{xits-math.otf}",
            r"\setmainfont{DejaVu Serif}", # serif font via preamble
            ]
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
    matplotlib.rcParams['lines.markersize'] = 4

def set_size(width, fraction=1, height_fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    height_fraction: float        
            Fraction of the height which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width * fraction # Width of figure
    inches_per_pt = 1 / 72.27 # Convert from pt to inches
    golden_ratio = (5**.5 - 1) / 2  # Golden ratio to set aesthetic figure height
    fig_width_in = fig_width_pt * inches_per_pt # Figure width in inches
    fig_height_in = fig_width_in * golden_ratio * height_fraction  # Figure height in inches
    return (fig_width_in, fig_height_in)
#---------------------------------------------------
def power_comp(freq, table):
# calculates the power difference needed to compensate the losses in the cables at the given frequency    
    table_f = np.asarray([i[0] for i in table], dtype=np.float32)
    ind = np.abs(table_f - freq).argmin()
    return table[ind][1]
#---------------------------------------------------

def scale_to_61(x):
    return 20*np.log10(61.25/x)

with open("/mnt/home/documents/Measurements/MPW2221_TSPC_div/sn10347321+biasT+INF10031+biasT+sn11167428_slow.csv",newline='') as csvfile:
    att_freq_data = list(csv.reader(csvfile))[1:] 

plt_cnt = 0
line_handle = []
mev = 63
msize=2

# fig, ax = plt.subplots()
# ax.set_xlabel("f [Hz]")
# ax.set_ylabel("Phase Noise [dBc/Hz]")
# ax.grid(which='major', alpha=0.7)
# ax.grid(which='minor', alpha=0.4)
# ax.tick_params(length=2)
if False:
    ax.semilogx(df_sg_10['df']/10*61.25, savgol_filter(df_sg_10['pn'],21,3),label="10GHz")
    ax.semilogx(df_sg_24['df']/24*61.25, savgol_filter(df_sg_24['pn'],21,3),label="24GHz")
    ax.semilogx(df_sg_61p25['df'], savgol_filter(df_sg_61p25['pn'],21,3),   label="61.25GHz")
    ax.semilogx(df_sg_67['df']/67*61.25, savgol_filter(df_sg_67['pn'],21,3),label="67GHz")
    ax.set_title("Phase Noise; offset frequency normalized to 61.25")
    fig.legend(loc="upper right", bbox_to_anchor=(0.85,0.85))
    fig.savefig("generator_phase_noise_df_scaled.png", bbox_inches='tight',pad_inches = 0)
if False:
    ax.semilogx(df_sg_10['df'], savgol_filter(df_sg_10['pn'],21,3),label="10GHz")
    ax.semilogx(df_sg_24['df'], savgol_filter(df_sg_24['pn'],21,3),label="24GHz")
    ax.semilogx(df_sg_61p25['df'], savgol_filter(df_sg_61p25['pn'],21,3),   label="61.25GHz")
    ax.semilogx(df_sg_67['df'], savgol_filter(df_sg_67['pn'],21,3),label="67GHz")
    ax.set_title("Phase Noise")
    fig.legend(loc="upper right", bbox_to_anchor=(0.85,0.85))
    fig.savefig("generator_pahse_noise.png", bbox_inches='tight',pad_inches = 0)
if False:
    ax.semilogx(df_sg_10['df'],    savgol_filter(df_sg_10['pn'],21,3)+scale_to_61(10),label="10GHz")
    ax.semilogx(df_sg_24['df'],    savgol_filter(df_sg_24['pn'],21,3)+scale_to_61(24),label="24GHz")
    ax.semilogx(df_sg_61p25['df'], savgol_filter(df_sg_61p25['pn'],21,3),   label="61.25GHz")
    ax.semilogx(df_sg_67['df'],    savgol_filter(df_sg_67['pn'],21,3)+scale_to_61(67),label="67GHz")
    ax.set_title("Phase Noise; normed to 61.25GHz with 20dB/dec ")
    fig.legend(loc="upper right", bbox_to_anchor=(0.85,0.85))
    fig.savefig("generator_phase_noise_scaled.png", bbox_inches='tight',pad_inches = 0)

# s = set_size(IEEE_width, fraction=0.5, height_fraction=0.5)
# fig, ax = plt.subplots(figsize=(s[0],s[1]))


if False:
    col, mrk, my_label = "red", markers[plt_cnt], "0.5V 0.5V 0V" # 24GHz
    ax.semilogx(df_et_0p5_0p5['df'],savgol_filter(df_et_0p5_0p5['pn'],21,3), color=col, marker=mrk,ms=msize,markevery=mev); plt_cnt +=1
    line_handle.append( matplotlib.lines.Line2D([],[], color=col, label=my_label, marker=mrk))

    col, mrk, my_label = "blue", markers[plt_cnt], "0.8V 0V 0V" # 24GHz
    ax.semilogx(df_et_0p8_0['df'],savgol_filter(df_et_0p8_0['pn'],21,3), color=col, marker=mrk,ms=msize,markevery=mev); plt_cnt +=1
    line_handle.append( matplotlib.lines.Line2D([],[], color=col, label=my_label, marker=mrk))

    col, mrk, my_label = "green", markers[plt_cnt], "0.9V 0V 0V" # 24GHz
    ax.semilogx(df_et_0p9_0['df'],savgol_filter(df_et_0p9_0['pn'],21,3), color=col, marker=mrk,ms=msize,markevery=mev); plt_cnt +=1
    line_handle.append( matplotlib.lines.Line2D([],[], color=col, label=my_label, marker=mrk))

    col, mrk, my_label = "grey", markers[plt_cnt], "generator 24GHz" # 24GHz
    ax.semilogx(df_sg_24['df']/4, savgol_filter(df_sg_24['pn'],21,3), color="grey", linestyle=":", marker=None); plt_cnt +=1
    line_handle.append( matplotlib.lines.Line2D([],[], color=col, label=my_label, marker=None, linestyle=":"))
    # ax.semilogx(df_sg_6125_tr2['df']/4, df_sg_6125_tr2['pn']+float(power_comp(freq=61.25e9,table=att_freq_data)), color="orange", linestyle=":", marker=markers[plt_cnt],label="61.25GHz; 2dBm"); plt_cnt +=1

    ax.annotate(r"V\textsubscript{DD} V\textsubscript{BN} V\textsubscript{BP}:",xy=(0,1.07),xycoords="axes fraction",fontsize=7)
    ax.legend(handles = line_handle, frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(0.16,1.0,0.86,0.04), borderaxespad=0, ncol=len(line_handle)+1, loc='lower center', mode="expand", handlelength=1)

if False:
    ####################################################################################################
    # 24GHz
    fig, ax = plt.subplots()
    ax.semilogx(df_et_0p5_0p5['df'],savgol_filter(df_et_0p5_0p5['pn'],13,3), label="0.5V 0.5V 0V 24GHz")
    ax.semilogx(df_et_0p8_0['df'],savgol_filter(df_et_0p8_0['pn'],13,3), label="0.8V 0.5V 0V 24GHz")
    ax.semilogx(df_et_0p9_0['df'],savgol_filter(df_et_0p9_0['pn'],13,3), label="0.9V 0V 0V 24GHz")
    ax.semilogx(df_sg_24['df'],savgol_filter(df_sg_24['pn'],13,3), label="generator 24GHz")

    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel("df [Hz]")
    ax.set_ylabel("Phase Noise [dBc/Hz]")
    ax.set_xlim(1e3,1e6)
    ax.set_ylim(-130,-90)
    fig.legend(loc="upper right", bbox_to_anchor=(0.85,0.85))
    # plt.show()
    plt.tight_layout()
    fig.savefig("./etspc_phase_noise_24GHz.png", bbox_inches='tight',pad_inches = 0)
    if latex:
        fig.savefig("./etspc_phase_noise_24GHz.pgf", bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

    ####################################################################################################
    # 61.25GHz
    fig, ax = plt.subplots()
    ax.semilogx(df_et_0p8_2['df'],savgol_filter(df_et_0p8_2['pn'],13,3), label="0.8V 2V -1V 61.25GHz")
    ax.semilogx(df_et_0p9_2_61p25['df'],savgol_filter(df_et_0p9_2_61p25['pn'],13,3), label="0.9V 2V -1V 61.25GHz")
    # ax.semilogx(df_et_0p9_2_61p25_54['df'],savgol_filter(df_et_0p9_2_61p25_54['pn'],13,3), label="0.9V 2V -1V 61.25GHzX") # same...
    ax.semilogx(df_et_0p9_3_61p25['df'],savgol_filter(df_et_0p9_3_61p25['pn'],13,3), label="0.9V 3V -1V 61.25GHz")
    ax.semilogx(df_sg_61p25['df'],savgol_filter(df_sg_61p25['pn'],13,3), label="generator 61.25GHz")

    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel("df [Hz]")
    ax.set_ylabel("Phase Noise [dBc/Hz]")
    ax.set_xlim(1e3,1e6)
    # ax.set_ylim(-130,-90)
    fig.legend(loc="lower left", bbox_to_anchor=(0.1,0.1))
    # plt.show()
    plt.tight_layout()
    fig.savefig("./etspc_phase_noise_61p25GHz.png", bbox_inches='tight',pad_inches = 0)
    if latex:
        fig.savefig("./etspc_phase_noise_61p25GHz.pgf", bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

    ####################################################################################################
    # 70 GHz
    fig, ax = plt.subplots()
    ax.semilogx(df_et_0p9_2_70['df'],savgol_filter(df_et_0p9_2_70['pn'],13,3), label="0.9V 2V -1V 70GHz")
    # ax.semilogx(df_et_0p9_2_70_8p5['df'],savgol_filter(df_et_0p9_2_70_8p5['pn'],13,3), label="0.9V 2V -1V 70GHzX")
    ax.semilogx(df_et_0p9_3_70['df'],savgol_filter(df_et_0p9_3_70['pn'],13,3), label="0.9V 3V -1V 70GHz")
    ax.semilogx(df_sg_67['df'],savgol_filter(df_sg_67['pn'],13,3), label="generator 67GHz")

    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel("df [Hz]")
    ax.set_ylabel("Phase Noise [dBc/Hz]")
    ax.set_xlim(1e3,1e6)
    # ax.set_ylim(-130,-90)
    fig.legend(loc="lower left", bbox_to_anchor=(0.1,0.1))
    # plt.show()
    plt.tight_layout()
    fig.savefig("./etspc_phase_noise_70GHz.png", bbox_inches='tight',pad_inches = 0)
    if latex:
        fig.savefig("./etspc_phase_noise_70GHz.pgf", bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

####################################################################################################
if do_etspc:
    # All ETSPC
    # fig, ax = plt.subplots()
    s = set_size(IEEE_width, fraction=0.5)
    fig, ax = plt.subplots(figsize=(s[0],s[1]))
    plt_cnt = 0
    d_ls = {10: (0,(1,3)), 24: 'dotted', 61.25:'solid', 70:'dashed'} # linestyles
    l05_3_1 , = ax.semilogx(df_et_0p5_3['df'],savgol_filter(df_et_0p5_3['pn'],13,3), label="0.5V 3V -1V", marker=d_style[(0.5,3,-1)][1], color=d_style[(0.5,3,-1)][0], markevery=200, linestyle=d_ls[24]); plt_cnt+=1
    ax.semilogx(df_et_sim_0p5_3['df'],savgol_filter(df_et_sim_0p5_3['pn'],13,3), marker=d_style[(0.5,3,-1)][1], color=d_style[(0.5,3,-1)][0], markevery=7, linestyle=d_ls[24], alpha=0.5); plt_cnt+=1
    # ax.semilogx(df_et_0p8_0['df'],savgol_filter(df_et_0p8_0['pn'],13,3), label="0.8V 0.5V 0V 24GHz")
    l09_0_0, = ax.semilogx(df_et_0p9_0['df'],savgol_filter(df_et_0p9_0['pn'],13,3), label="0.9V 0V 0V", marker=d_style[(0.9,0,0)][1], color=d_style[(0.9,0,0)][0], markevery=(40,200), linestyle=d_ls[24]); plt_cnt+=1
    ax.semilogx(df_et_sim_0p9_0['df'],savgol_filter(df_et_sim_0p9_0['pn'],13,3), label="0.9V 0V 0V", marker=d_style[(0.9,0,0)][1], color=d_style[(0.9,0,0)][0], markevery=7, linestyle=d_ls[24],alpha=0.5); plt_cnt+=1
    l08_2_1, = ax.semilogx(df_et_0p8_2['df'],savgol_filter(df_et_0p8_2['pn'],13,3), label="0.8V  2V  -1V", marker=d_style[(0.8,2,-1)][1], color=d_style[(0.8,2,-1)][0], markevery=(75,200), linestyle=d_ls[61.25]); plt_cnt+=1
    ax.semilogx(df_et_sim_0p8_2['df'],savgol_filter(df_et_sim_0p8_2['pn'],13,3), label="0.8V  2V  -1V", marker=d_style[(0.8,2,-1)][1], color=d_style[(0.8,2,-1)][0], markevery=7, linestyle=d_ls[61.25], alpha=0.5); plt_cnt+=1
    # l09_61, = ax.semilogx(df_et_0p9_2['df'],savgol_filter(df_et_0p9_2['pn'],13,3), label="0.9V  2V  -1V", marker=d_style[(0.9,2,-1)][1], color=d_style[(0.9,2,-1)][0], markevery=(110,200), linestyle=d_ls[61.25]); plt_cnt+=1
    # ax.semilogx(df_et_0p9_3_61p25['df'],savgol_filter(df_et_0p9_3_61p25['pn'],13,3), label="0.9V 3V -1V 61.25GHz")
    # ax.semilogx(df_et_0p9_2_70['df'],savgol_filter(df_et_0p9_2_70['pn'],13,3), label="0.9V 2V -1V 70GHz")
    l09_3_1, = ax.semilogx(df_et_0p9_3_70['df'],savgol_filter(df_et_0p9_3_70['pn'],13,3), label="0.9V  3V  -1V", marker=d_style[(0.9,3,-1)][1], color=d_style[(0.9,3,-1)][0], markevery=(150,200), linestyle=d_ls[70]); plt_cnt+=1
    ax.semilogx(df_et_sim_0p9_3_70['df'],savgol_filter(df_et_sim_0p9_3_70['pn'],13,3), label="0.9V  3V  -1V", marker=d_style[(0.9,3,-1)][1], color=d_style[(0.9,3,-1)][0], markevery=7, linestyle=d_ls[70], alpha=0.5); plt_cnt+=1
    l09_2_1, = ax.semilogx(df_et_0p9_2_70['df'],savgol_filter(df_et_0p9_2_70['pn'],13,3), label="0.9V  2V  -1V", marker=d_style[(0.9,2,-1)][1], color=d_style[(0.9,2,-1)][0], markevery=(180,200), linestyle=d_ls[70]); plt_cnt+=1
    ax.semilogx(df_et_sim_0p9_2_70['df'],savgol_filter(df_et_sim_0p9_2_70['pn'],13,3), label="0.9V  2V  -1V", marker=d_style[(0.9,2,-1)][1], color=d_style[(0.9,2,-1)][0], markevery=7, linestyle=d_ls[70], alpha=0.5); plt_cnt+=1
    # ax.semilogx(df_sg_24['df']/4,savgol_filter(df_sg_24['pn'],13,3), label="SG 24GHz", color='black', markevery=(180,200), linestyle=d_ls[24]); plt_cnt+=1
    ax.semilogx(df_sg_24['df'],savgol_filter(df_sg_24['pn'],13,3), label="SG 24GHz", color='black', markevery=(180,200), linestyle=d_ls[24]); plt_cnt+=1
    # lsg, = ax.semilogx(df_sg_61['df']/4,savgol_filter(df_sg_61['pn'],13,3), label="SG 61GHz", color='black', markevery=(180,200), linestyle=d_ls[61.25]); plt_cnt+=1
    ax.semilogx(df_sg_61['df'],savgol_filter(df_sg_61['pn'],13,3), label="SG 61GHz", color='black', markevery=(180,200), linestyle=d_ls[61.25]); plt_cnt+=1
    

    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel(r"$\Delta$f [Hz]", labelpad=0)
    ax.set_ylabel("Phase Noise [dBc/Hz]",labelpad=2)
    ax.set_xlim(1e3,1e6)
    ax.set_ylim(-130,-90)

    # fig.legend(loc="lower left", bbox_to_anchor=(0.1,0.1),title="VDD VBN VBP  fin")
    lines = [matplotlib.lines.Line2D([0], [0], color='black', linestyle=d_ls[freq]) for freq in [24,61.25,70]]
    labels = ["24 GHz", "61.25 GHz", "70 GHz"]
    leg_freq = ax.legend(lines, labels, frameon=False, fontsize=7, bbox_to_anchor=(0.75,0.82,0.25,0.2))
    
    leg1 = ax.legend(handles=[l05_3_1, l08_2_1], frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(0.303,1.1,0.727,0.1),borderaxespad=0,loc="center right", mode="expand", handlelength=3, ncol=2)
    leg2 = ax.legend(handles=[l09_0_0, l09_2_1 , l09_3_1], frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(-0.05,0.99,1.08,0.1),borderaxespad=0,loc="lower center", mode="expand", handlelength=3, ncol=3)
    
    ax.annotate(r"V\textsubscript{DD} V\textsubscript{BN} V\textsubscript{BP}:",xy=(0,1.14),xycoords="axes fraction", fontsize=7)
    ax.add_artist(leg1)
    ax.add_artist(leg_freq)
    plt.tight_layout()
    fig.savefig("./etspc_phase_noise_all.png", bbox_inches='tight',pad_inches = 0)
    if latex:
        fig.savefig("./etspc_phase_noise_all.pgf", bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

if False:
    ####################################################################################################
    # All ETSPC scaled
    fig, ax = plt.subplots()
    ax.semilogx(df_et_0p5_0p5['df'],savgol_filter(df_et_0p5_0p5['pn'],13,3)+scale_to_61(24), label="0.5V 0.5V 0V  24GHz")
    ax.semilogx(df_et_0p8_0['df'],savgol_filter(df_et_0p8_0['pn'],13,3)+scale_to_61(24), label="0.8V 0.5V 0V  24GHz")
    ax.semilogx(df_et_0p9_0['df'],savgol_filter(df_et_0p9_0['pn'],13,3)+scale_to_61(24), label="0.9V 0.5V 0V  24GHz")
    ax.semilogx(df_et_0p8_2['df'],savgol_filter(df_et_0p8_2['pn'],13,3), label="0.8V 2V -1V  61.25GHz")
    ax.semilogx(df_et_0p9_2_61p25['df'],savgol_filter(df_et_0p9_2_61p25['pn'],13,3), label="0.9V 2V -1V  61.25GHz")
    ax.semilogx(df_et_0p9_3_61p25['df'],savgol_filter(df_et_0p9_3_61p25['pn'],13,3), label="0.9V 3V -1V  61.25GHz")
    ax.semilogx(df_et_0p9_2_70['df'],savgol_filter(df_et_0p9_2_70['pn'],13,3)+scale_to_61(70), label="0.9V 2V -1V  70GHz")
    ax.semilogx(df_et_0p9_3_70['df'],savgol_filter(df_et_0p9_3_70['pn'],13,3)+scale_to_61(70), label="0.9V 3V -1V  70GHz")

    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel("df [Hz]")
    ax.set_ylabel("Phase Noise [dBc/Hz]")
    ax.set_xlim(1e3,1e6)
    # ax.set_ylim(-130,-90)
    fig.legend(loc="lower left", bbox_to_anchor=(0.1,0.1))
    # plt.show()
    plt.tight_layout()
    fig.savefig("./etspc_phase_noise_all_scaled.png", bbox_inches='tight',pad_inches = 0)
    if latex:
        fig.savefig("./etspc_phase_noise_all_scaled.pgf", bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

    ####################################################################################################
    # All ETSPC scaled df
    fig, ax = plt.subplots()
    ax.semilogx(df_et_0p5_0p5['df']/24*61.25,savgol_filter(df_et_0p5_0p5['pn'],13,3), label="0.5V 0.5V 0V 24GHz")
    ax.semilogx(df_et_0p8_0['df']/24*61.25,savgol_filter(df_et_0p8_0['pn'],13,3), label="0.8V 0.5V 0V 24GHz")
    ax.semilogx(df_et_0p9_0['df']/24*61.25,savgol_filter(df_et_0p9_0['pn'],13,3), label="0.9V 0.5V 0V 24GHz")
    ax.semilogx(df_et_0p8_2['df'],savgol_filter(df_et_0p8_2['pn'],13,3), label="0.8V 2V -1V 61.25GHz")
    ax.semilogx(df_et_0p9_2_61p25['df'],savgol_filter(df_et_0p9_2_61p25['pn'],13,3), label="0.9V 2V -1V 61.25GHz")
    ax.semilogx(df_et_0p9_3_61p25['df'],savgol_filter(df_et_0p9_3_61p25['pn'],13,3), label="0.9V 3V -1V 61.25GHz")
    ax.semilogx(df_et_0p9_2_70['df']/70*61.25,savgol_filter(df_et_0p9_2_70['pn'],13,3), label="0.9V 2V -1V 70GHz")
    ax.semilogx(df_et_0p9_3_70['df']/70*61.25,savgol_filter(df_et_0p9_3_70['pn'],13,3), label="0.9V 3V -1V 70GHz")

    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel("df [Hz]")
    ax.set_ylabel("Phase Noise [dBc/Hz]")
    ax.set_xlim(1e3,1e6)
    # ax.set_ylim(-130,-90)
    fig.legend(loc="lower left", bbox_to_anchor=(0.1,0.1))
    # plt.show()
    plt.tight_layout()
    fig.savefig("./etspc_phase_noise_all_scaled_df.png", bbox_inches='tight',pad_inches = 0)
    if latex:
        fig.savefig("./etspc_phase_noise_all_scaled_df.pgf", bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

if do_elvt:
    ####################################################################################################
    # All ELVT
    # fig, ax = plt.subplots()
    # s = set_size(IEEE_width, fraction=0.5, height_fraction=0.5)
    s = set_size(IEEE_width, fraction=0.5)
    fig, ax = plt.subplots(figsize=(s[0],s[1]*0.7))
    d_ls = {10: 'dotted', 24: 'dashed', 61.25:'solid'} # linestyles
    plt_cnt = 0
    el_05_05_10, = ax.semilogx(df_el_0p5_0p5['df'],savgol_filter(df_el_0p5_0p5['pn'],13,3),       label="0.5V 0.5V 0V", marker=markers[plt_cnt], markevery=700, linestyle=d_ls[10] ); plt_cnt+=1
    el_05_2_24,  = ax.semilogx(df_el_0p5_2['df'],savgol_filter(df_el_0p5_2['pn'],13,3),           label="0.5V  2V  0V", marker=markers[plt_cnt], markevery=(117,700), linestyle=d_ls[24] ); plt_cnt+=1
    # ax.semilogx(df_el_0p8_0p8_44['df'],savgol_filter(df_el_0p8_0p8_44['pn'],13,3), label="0.8V 0.8V 0V  24GHz1"); plt_cnt+=1
    el_08_24, = ax.semilogx(df_el_0p8_0p8_48['df'],savgol_filter(df_el_0p8_0p8_48['pn'],13,3), label="0.8V 0.8V 0V", marker=markers[plt_cnt], markevery=(233,700), linestyle=d_ls[24] ); plt_cnt+=1
    # ax.semilogx(df_el_0p8_0p8_56['df'],savgol_filter(df_el_0p8_0p8_56['pn'],13,3), label="0.8V 0.8V 0V  24GHz3"); plt_cnt+=1
    el_08_61, = ax.semilogx(df_el_0p8_0p8_58['df'],savgol_filter(df_el_0p8_0p8_58['pn'],13,3), label="0.8V 0.8V 0V", marker=markers[plt_cnt], markevery=(350,700), linestyle=d_ls[61.25] ); plt_cnt+=1
    # el_08_61b,= ax.semilogx(df_el_0p8_2_48['df'],savgol_filter(df_el_0p8_2_48['pn'],13,3),     label="0.8V  2V -1V", marker=markers[plt_cnt], markevery=(467,700), linestyle=d_ls[61.25] ); plt_cnt+=1
    # ax.semilogx(df_el_0p8_2_58['df'],savgol_filter(df_el_0p8_2_58['pn'],13,3), label="0.8V 2V -1V 61.25GHz2"); plt_cnt+=1
    el_09_61, = ax.semilogx(df_el_0p9_2['df'],savgol_filter(df_el_0p9_2['pn'],13,3),           label="0.9V  2V -1V", marker=markers[plt_cnt], markevery=(583,700), linestyle=d_ls[61.25] ); plt_cnt+=1

    # generate a continous linestyle for the legend:
    plt_cnt=0
    figx, axx = plt.subplots(figsize=(s[0],s[1]))
    leg_el_05_05_10, = axx.semilogx([],[], label="0.5V 0.5V 0V", marker=markers[plt_cnt] ); plt_cnt+=1
    leg_el_05_2_24,  = axx.semilogx([],[], label="0.5V  2V  0V", marker=markers[plt_cnt] ); plt_cnt+=1
    leg_el_08_24,    = axx.semilogx([],[], label="0.8V 0.8V 0V", marker=markers[plt_cnt] ); plt_cnt+=1
    leg_el_08_61,    = axx.semilogx([],[], label="0.8V 0.8V 0V", marker=markers[plt_cnt] ); plt_cnt+=1
    leg_el_09_61,    = axx.semilogx([],[], label="0.9V  2V -1V", marker=markers[plt_cnt] ); plt_cnt+=1

    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel("$\Delta$f [Hz]", labelpad=0)
    ax.set_ylabel("Phase Noise [dBc/Hz]")
    ax.set_xlim(1e3,1e6)
    ax.set_ylim(-130,-90)

    lines = [matplotlib.lines.Line2D([0], [0], color='black', linestyle=d_ls[freq]) for freq in [10,24,61.25]]
    labels = ["\SI{10}{\GHz}", "24 GHz", "61.25 GHz"]
    leg_freq = ax.legend(lines, labels, frameon=False, fontsize=7, bbox_to_anchor=(0.8,0.82,0.25,0.2))
    leg1 = ax.legend(handles=[leg_el_05_05_10, leg_el_05_2_24], frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(0.35,1.09,0.64,0.1),borderaxespad=0,loc="center right", mode="expand", handlelength=2, ncol=2)
    leg2 = ax.legend(handles=[leg_el_08_24, leg_el_08_61 ,leg_el_09_61], frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(0,0.99,1,0.1),borderaxespad=0,loc="lower center", mode="expand", handlelength=2, ncol=3)
    ax.annotate(r"V\textsubscript{DD} V\textsubscript{BN} V\textsubscript{BP}:",xy=(0,1.12),xycoords="axes fraction", fontsize=7)
    ax.add_artist(leg1)
    ax.add_artist(leg_freq)

    # fig.legend(loc="lower left", bbox_to_anchor=(0.1,0.1))
    # plt.show()
    plt.tight_layout()
    fig.savefig("./elvt_phase_noise_all.png", bbox_inches='tight',pad_inches = 0)
    if latex:
        fig.savefig("./elvt_phase_noise_all.pgf", bbox_inches='tight',pad_inches = 0)
    plt.close(fig)

def my_plt_args(l):
    return {'label':"V ".join([str(i) for i in l]) +"V", 'color':d_style[l][0], 'marker':d_style[l][1]}


if do_slvt:
    ####################################################################################################
    # All SLVT
    # fig, ax = plt.subplots()
    # s = set_size(IEEE_width, fraction=0.5, height_fraction=0.5)
    s = set_size(IEEE_width, fraction=0.5)
    fig, ax = plt.subplots(figsize=(s[0],s[1]))
    d_ls = {10: 'dotted', 24: 'dashed', 61.25:'solid'} # linestyles
    plt_cnt = 0
    d_style = { # color dict. VDD, VBN, VBP
        (0.4,2,0):   ("yellow","^"),
        (0.4,3,-1):  ("green","^"),
        (0.5,3,-1):  ("blue","^"),
        (0.5,2,-1):  ("darkblue","s"),
        (0.8,0.8,0): ("orange","v"),
        (0.8,3,-1):  ("purple","v"),
        (0.9,3,-1):  ("red","o")
        }
    plt_cnt = 0
    me = 200
    n_plt = 4
    # bb = (0.4,2,0); df = df_sl_0p4_2; 
    # ls_04_2_0, = ax.semilogx(df['df'],savgol_filter(df['pn'],13,3), markevery=700, linestyle=d_ls[10], **my_plt_args(bb) );
    bb = (0.4,3,-1); df = df_sl_0p4_3;     
    ls_04_3_1, = ax.semilogx(df['df'],savgol_filter(df['pn'],13,3), markevery=(int(plt_cnt/n_plt)*me,me), linestyle=d_ls[10], **my_plt_args(bb) )
    bb = (0.5,3,-1); df = df_sl_0p5_3; plt_cnt +=1    
    ls_05_3_1, = ax.semilogx(df['df'],savgol_filter(df['pn'],13,3), markevery=(int(plt_cnt/n_plt*me),me), linestyle=d_ls[24], **my_plt_args(bb) )
    # bb = (0.5,2,-1); df = df_sl_0p5_2;     
    # ls_05_2_1, = ax.semilogx(df['df'],savgol_filter(df['pn'],13,3), markevery=700, linestyle=d_ls[24], **my_plt_args(bb) ); 
    bb = (0.8,0.8,0); df = df_sl_0p8_0p8; plt_cnt +=1    
    ls_08_0p8_0, = ax.semilogx(df['df'],savgol_filter(df['pn'],13,3), markevery=(int(plt_cnt/n_plt*me),me), linestyle=d_ls[24], **my_plt_args(bb) )
    # bb = (0.8,3,-1); df = df_sl_0p8_3;     
    # ls_08_3_1, = ax.semilogx(df['df'],savgol_filter(df['pn'],13,3), markevery=700, linestyle=d_ls[61.25], **my_plt_args(bb) );
    bb = (0.9,3,-1); df = df_sl_0p9_3; plt_cnt +=1    
    ls_09_3_1, = ax.semilogx(df['df'],savgol_filter(df['pn'],13,3), markevery=(int(plt_cnt/n_plt*me),me), linestyle=d_ls[61.25], **my_plt_args(bb) ) 
    
    # ax.semilogx(df_sg_10['df'],savgol_filter(df_sg_10['pn'],13,3), label="SG 10GHz", color='black', markevery=(180,200), linestyle=d_ls[10]); 
    # ax.semilogx(df_sg_24['df'],savgol_filter(df_sg_24['pn'],13,3), label="SG 24GHz", color='black', markevery=(180,200), linestyle=d_ls[24]); 
    # ax.semilogx(df_sg_61['df'],savgol_filter(df_sg_61['pn'],13,3), label="SG 24GHz", color='black', markevery=(180,200), linestyle=d_ls[24]); 

    ax.grid(which='major', alpha=0.7)
    ax.grid(which='minor', alpha=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel("$\Delta$f [Hz]", labelpad=0)
    ax.set_ylabel("Phase Noise [dBc/Hz]")
    ax.set_xlim(1e3,1e6)
    ax.set_ylim(-130,-90)

    lines = [matplotlib.lines.Line2D([0], [0], color='black', linestyle=d_ls[freq]) for freq in [10,24,61.25]]
    labels = ["\SI{10}{\GHz}", "\SI{24}{\GHz}", "\SI{61.25}{\GHz}"]
    leg_freq = ax.legend(lines, labels, frameon=False, fontsize=7, bbox_to_anchor=(0.75,0.82,0.25,0.2))
    leg1 = ax.legend(handles=[ls_04_3_1,ls_05_3_1], frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(0.25,1.09,0.75,0.1),borderaxespad=0,loc="center right", mode="expand", handlelength=3, ncol=2)
    leg2 = ax.legend(handles=[ls_08_0p8_0, ls_09_3_1], frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(0.25,0.99,0.75,0.1),borderaxespad=0,loc="lower center", mode="expand", handlelength=3, ncol=2)
    ax.annotate(r"V\textsubscript{DD} V\textsubscript{BN} V\textsubscript{BP}:",xy=(0,1.13),xycoords="axes fraction", fontsize=7)
    ax.add_artist(leg1)
    ax.add_artist(leg_freq)

    # fig.legend(loc="lower left", bbox_to_anchor=(0.1,0.1))
    # plt.show()
    plt.tight_layout()
    fig.savefig("./slvt_phase_noise_all.png", bbox_inches='tight', pad_inches = 0)
    if latex:
        fig.savefig("./slvt_phase_noise_all.pgf", bbox_inches='tight', pad_inches = 0)
    plt.close(fig)



# if False: # too small output power. Why???
#     fig, ax = plt.subplots(figsize=(s[0],s[1]))
#     df=pd.read_csv('ETSPC_61p25GHz_8p1dBm_0p8_3_m1p5_0p4V_spectrum.DAT', usecols=[0,1],sep=';',skiprows=31,names=["f","p"],dtype=np.float64)
#     df["freq [GHz]"] = df["f"]*1e-9
#     df["df"] = (df["f"]-df["f"].median())*1e-6 # MHz
#     df["power_comp"] = np.array([float(power_comp(float(i),table=att_freq_data)) for i in df["f"]])
#     df["power [dBm]"] = df["p"] - df['power_comp']
#     df.plot(x='df',y='p',ax=ax)
#     ax.set_xlabel(r'dfreq [MHz];\n  f\textsubscript{center}=15.3125 GHz',labelpad=0)
#     ax.set_ylabel(r'Output power [dBm]',labelpad=0)
#     ax.grid(which='major', alpha=0.5)
#     ax.grid(which='minor', alpha=0.2)
#     ax.tick_params(length=2)
#     plt.show()
