# matplotlib, multi-label, on-plot annotation of result summary, pandas, curve_fit,
# - Using self-defined styles for different lines in the plots
# - cable loss determinitation function (from .s2p file) and its use to deembed cable losses.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import optimize
from functools import partial
import skrf as rf

do_pn       = False
do_title    = True

suffix = "cmos_sf"

meas_dir = "/mnt/home/documents/Measurements/MPW2215_VCO/"
meas_dir = "/home/zoltan/ccn/Measurements/MPW2215_VCO/"
sim_dir = "/mnt/home/documents/Measurements/MPW2215_VCO/sim/"
sim_dir = "/home/zoltan/ccn/Measurements/MPW2215_VCO/sim/"


style_sim  = {"linestyle":"--", "color":"grey", "marker":"None", 'alpha':0.5} #, "label":"sim"} # for plot
style_meas = {"linestyle":":", "color":"black", "marker":"o","s":40} # , "label":"meas"}  # for scatter
style_meas1 = {"linestyle":"-", "edgecolor":None,"facecolor":"blue",  "s":20, "marker":"o"} #, "label":"meas", "marker":"o"}
style_meas2 = {"linestyle":"-", "edgecolor":None,"facecolor":"green", "s":20, "marker":"v"} #, "label":"meas", "marker":"o"}
style_meas3 = {"linestyle":"-", "edgecolor":None,"facecolor":"orange", "s":20, "marker":"s"} #, "label":"meas", "marker":"o"}
style_meas4 = {"linestyle":"-", "edgecolor":None,"facecolor":"purple", "s":20, "marker":"^"} #, "label":"meas", "marker":"o"}
# style_meas = {"linestyle":":", "color":"black", "marker":"o","s":75, "label":"meas"}
legend_style = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":3, "loc":'lower center', "mode":"expand", "handlelength":2}
legend_style2 = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":1, "loc":'lower center', "mode":"expand", "handlelength":2}
legend_style2_ncol2 = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":2, "loc":'lower center', "mode":"expand", "handlelength":2}
legend_style2_ncol2_ul = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":2, "loc":'upper left', "mode":"expand", "handlelength":2}

l_style_meas = [style_meas1, style_meas2, style_meas3, style_meas4]

plot_style = {"figsize":(3.3914487339144874, 2.0960305886619515*0.85)}
plot_style_3 = {"figsize":(3.3914487339144874, 2.0960305886619515*0.8*3)}

opt_vcsv = {'header':None, 'skiprows':6, 'dtype':np.float64, 'usecols':[0,1,3,7],'names':['vtune','freq','pout','pdc']} 

def eff(pout, vdc, idc, in_dBm=False):
    """ Calculating the DC-to-RF efficiency.
        
        Paramters
        ---------
        pout:       float
                    output power
        vdc:        float
                    dc supply voltage
        idc:        float
                    dc suply current
        in_dBm:     boolean, optional
                    is pout value in dBm?
        
        Returns
        -------
        float
                   DC-to-RF efficiency in percentage.
    """
    if in_dBm:
        ret = (10**(pout/10))*1e-3/vdc/idc*100
    else:
        ret = pout/vdc/idc*100
    return ret

latex=True
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
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pgf.preamble": [
            "\\usepackage{siunitx}",         # load additional packages
            "\\usepackage{amsmath}",         # load additional packages
            "\\usepackage{metalogo}",
            "\\usepackage{unicode-math}",  # unicode math setup
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

IEEE_width = 516 #pt

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    fig_width_pt = width * fraction # Width of figure
    inches_per_pt = 1 / 72.27 # Convert from pt to inches
    golden_ratio = (5**.5 - 1) / 2  # Golden ratio to set aesthetic figure height
    fig_width_in = fig_width_pt * inches_per_pt # Figure width in inches
    fig_height_in = fig_width_in * golden_ratio # Figure height in inches
    return (fig_width_in, fig_height_in)

#######################################
#   Cable loss interpolation 
#######################################
f_loss = "Probe_842922_cable_05439601_dcblock_avg.s2p"
nw_loss = rf.Network(meas_dir + f_loss)

def loss_func(f, dc,sq,lin):
    return dc + sq*np.sqrt(f) + lin*f

x = nw_loss.f
y = nw_loss.s_db[:,1,0]
par, cov = optimize.curve_fit(loss_func, x, y)
cable_loss = partial(loss_func, dc=par[0], sq=par[1], lin=par[2])

#################################
# Measurement results from Excel
#################################
sim = "/mnt/home/documents/Design/pictures/vco_mpw2215/sf_20200209.vcsv" 
sim = "/home/zoltan/ccn/Design/pictures/vco_mpw2215/sf_20200209.vcsv" 
# sim_cmos_60G = "/mnt/home/documents/Design/pictures/vco_mpw2215/cmos_60G_20200208.vcsv"
meas_dir = "/mnt/home/documents/Measurements/MPW2215_VCO/"
meas_dir = "/home/zoltan/ccn/Measurements/MPW2215_VCO/"
meas_xlsx = meas_dir + "vco_cmos_60G_SF.xlsx"
df_meas = pd.read_excel(meas_xlsx, usecols="B:H", sheet_name="combined", header=0).rename(columns={"IDD":"idd", "Pout":"pout", "Vtune":"vtune", "VDD":"vdd", "Ib":"ib", "Ibuf":"ibuf"})

# p_out supposed to be the DIFFERENTIAL output power available at the PADS
# The single ended output was measured
df_meas.pout += 3 - cable_loss(df_meas.freq*1e9) # it should be moved to the common part
df_meas["eff"] = eff(df_meas.pout, df_meas.vdd, df_meas.idd*1e-3, in_dBm=True) 

l_cond = [{"vdd":0.8, "ib":3},
          {"vdd":1,   "ib":3},
          {"vdd":1.2, "ib":3},
          {"vdd":1.4, "ib":3}
        ]


# Read in the different simulation resutls and add the operation point information to them
df_sim0p8 = pd.read_csv(sim_dir + 'cmos_sf_vdd0p8V_ib3mA.vcsv', **opt_vcsv)
df_sim0p8['vdd'] = 0.8
df_sim0p8['ib'] = 3
df_sim1 = pd.read_csv(sim_dir + 'cmos_sf_vdd1V_ib3mA.vcsv', **opt_vcsv)
df_sim1['vdd'] = 1
df_sim1['ib'] = 3
df_sim1p2 = pd.read_csv(sim_dir + 'cmos_sf_vdd1p2V_ib3mA.vcsv', **opt_vcsv)
df_sim1p2['vdd'] = 1.2
df_sim1p2['ib'] = 3
df_sim1p4 = pd.read_csv(sim_dir + 'cmos_sf_vdd1p4V_ib3mA.vcsv', **opt_vcsv)
df_sim1p4['vdd'] = 1.4
df_sim1p4['ib'] = 3

# Create one dataframe for easier use
df_sim = pd.concat([df_sim0p8, df_sim1, df_sim1p2, df_sim1p4], ignore_index=True)
df_sim.loc[:,'freq'] *= 1e-9
df_sim.loc[:,'pout'] += 3
df_sim["eff"] = np.power(10,(df_sim.pout-df_sim.pdc)/10) * 100


######################################
# Tuning range
######################################
# fig_s, ax_s = plt.subplots(figsize=(3.3914487339144874, 2.0960305886619515))
fig_s, ax_s = plt.subplots(**plot_style)

tr_label = []
cnt = 0
for cond in l_cond:
    mask = pd.DataFrame([df_meas[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfm = df_meas[mask]
    mask = pd.DataFrame([df_sim[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfs = df_sim[mask]
    
    assert dfm.ib.nunique()==1, "Multiple bias current values"
    ax_s.plot(dfs.vtune, dfs.freq,**style_sim)
    ax_s.scatter(dfm.vtune, dfm.freq, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    cnt +=1

    fmin, fmax = dfm.freq.min(), dfm.freq.max()
    favg = (fmin+fmax)/2
    fper = (fmax-fmin)/favg*100/2
    print(cond, fmin, fmax, favg, fmax-fmin, fper*2)
    tr_label.append(r"$\SI{%.1f}{\GHz}\pm\SI{%.1f}{\percent}$" % (favg,fper))

ax_s.plot([],[],**style_sim, label="sim") # to add label
ax_s.set_xlim(-0.5, 1.6)
ax_s.set_ylim(45, None)
ax_s.set_xlabel(r"Tuning voltage $\left[\si{\volt}\right]$")
ax_s.set_ylabel(r"Frequency $\left [ \si{\GHz} \right ]$")
ax_s.grid()
handles, labels = ax_s.get_legend_handles_labels()
# # checking order of the labels;  DEBUG only
# print(handles)
# print(labels)
hand = [*handles[1:], handles[0]]
lab = [*labels[1:], labels[0]]
leg = ax_s.legend(handles=hand, labels=lab, bbox_to_anchor=(0.16,1.0,0.86,0.04), **legend_style) 
ax_s.legend(handles=handles[1:], labels=tr_label, **legend_style2, bbox_to_anchor=(0.55,0.02,0.4,0.2), title="Tuning range")
ax_s.add_artist(leg)
ax_s.annotate(r"V\textsubscript{DD} I\textsubscript{b}:",xy=(0,1.08),xycoords="axes fraction",fontsize=8)
for ext in ["png","pgf"]:
    fig_s.savefig(meas_dir + suffix + "_tune." + ext, bbox_inches='tight', pad_inches = 0)

######################################
# Output power
######################################
# The differential output power is plotted
fig,axes = plt.subplots(nrows=3, sharex=True, **plot_style_3)

pout_label = []
cnt = 0
for cond in l_cond:
    mask = pd.DataFrame([df_meas[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfm = df_meas[mask]
    mask = pd.DataFrame([df_sim[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfs = df_sim[mask]
    assert dfm.ib.nunique()==1, "Multiple bias current values"

    axes[0].plot(dfs.freq, dfs.pout, **style_sim)
    axes[0].scatter(dfm.freq, dfm.pout, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    cnt += 1

    p_avg = 10*np.log10(sum(10**(dfm.pout/10))/float(len(dfm.pout))) 
    # Use LaTeX/SIunitx or not to use?
    # pout_label.append(r"\SIrange[range-phrase=..]{%.1f}{%.1f}{}; \SI{%.1f}{\dBm}" % (dfm.pout.min(), dfm.pout.max(), p_avg)   )
    pout_label.append("%.1f..%.1f; %.1f\,dBm" % (dfm.pout.min(), dfm.pout.max(), p_avg)   )


axes[0].plot([],[],**style_sim, label="sim") # to add label
axes[0].set_ylabel(r"P\textsubscript{out} $\left[ \si{\dBm} \right ]$")
axes[0].set_xlim(45,65)
axes[0].set_ylim(-22,0)
axes[0].grid()

handles, labels = axes[0].get_legend_handles_labels()
# # checking order of the labels; DEBUG only
# print(handles)
# print(labels)
handles = [*handles[1:], handles[0]]
labels = [*labels[1:], labels[0]]
leg = axes[0].legend(handles=handles, labels=labels, bbox_to_anchor=(0.16,1.0,0.86,0.04), **legend_style) 
axes[0].legend(handles=handles[:len(handles)-1], labels=pout_label, **legend_style2_ncol2, bbox_to_anchor=(0,-0.02,1,0.18), title="P\\textsubscript{out}: min-max; avg",labelspacing=0.3)
axes[0].add_artist(leg)
axes[0].annotate(r"V\textsubscript{DD} I\textsubscript{b}:",xy=(0,1.08),xycoords="axes fraction",fontsize=8)
# for ext in ["png","pgf"]:
#     fig_s.savefig(meas_dir + suffix + "_pout." + ext, bbox_inches='tight', pad_inches = 0)

######################################
# Power consumption
######################################
# The differential output power is plotted
pdc_label = []
pdc_label2 = []
cnt = 0
for cond in l_cond:
    mask = pd.DataFrame([df_meas[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfm = df_meas[mask]
    mask = pd.DataFrame([df_sim[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfs = df_sim[mask]
    assert dfm.ib.nunique()==1, "Multiple bias current values"

    axes[1].plot(dfs.freq, dfs.pdc, **style_sim)
    meas_pdc_mW = dfm.vdd * dfm.idd
    meas_pdc = 10*np.log10(meas_pdc_mW)
    axes[1].scatter(dfm.freq, meas_pdc, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    cnt += 1

    # p_avg = 10*np.log10(sum(10**(dfm.pdc/10))/float(len(dfm.pdc))) 
    # pout_label.append(r"\SIrange[range-phrase=..]{%.1f}{%.1f}{}; \SI{%.1f}{\dBm}" % (dfm.pout.min(), dfm.pout.max(), p_avg)   )
    pdc_label.append("\SIrange{%.1f}{%.1f}{}; \SI{%.1f}{\dBm}" % (meas_pdc.min(), meas_pdc.max(), meas_pdc.mean())   )


axes[1].plot([],[],**style_sim, label="sim") # to add label
axes[1].set_ylabel(r"P\textsubscript{DC} $\left [ \si{\dBm} \right ]$")
axes[1].set_xlim(45,65)
axes[1].set_ylim(0,11)
axes[1].grid()

handles, labels = ax_s.get_legend_handles_labels()
axes[1].legend(handles=handles[1:], labels=pdc_label, **legend_style2_ncol2, bbox_to_anchor=(0,-0.02,1,0.2), title="P\\textsubscript{out}: min-max; avg", labelspacing=0.3)
# ax[1].add_artist(leg)

######################################
# DC to RF efficiency
######################################
eff_label = []
cnt = 0
for cond in l_cond:
    mask = pd.DataFrame([df_meas[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfm = df_meas[mask]
    mask = pd.DataFrame([df_sim[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfs = df_sim[mask]
    assert dfm.ib.nunique()==1, "Multiple bias current values"
    axes[2].plot(dfs.freq, dfs.eff,**style_sim)
    axes[2].scatter(dfm.freq, dfm.eff, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    cnt +=1
    
    eff_min, eff_max, eff_mean = dfm.eff.min(), dfm.eff.max(), dfm.eff.mean()
    eff_label.append("%.1f-%.1f%%; %.1f%%" % (eff_min, eff_max, eff_mean))

axes[2].set_xlabel(r"Frequency $\left[\si{\GHz} \right ]$")
axes[2].set_ylabel(r"${\eta  \left[ \si{\percent} \right] }$")
axes[2].set_ylim(0,9)
axes[2].set_xlim(45,65)
# axes[2].set_ylabel("eff [%]")
axes[2].grid()

handles, labels = ax_s.get_legend_handles_labels()
axes[2].legend(handles=handles[1:], labels=eff_label, **legend_style2_ncol2_ul, bbox_to_anchor=(0.0,0.63,0.8,0.4), title="$\eta$: min-max; avg", labelspacing=0.3)

fig.subplots_adjust(hspace=0.1)
for ext in ["png","pgf"]:
    fig.savefig(meas_dir + suffix + "_pout_pdc_eff." + ext, bbox_inches='tight', pad_inches = 0)
