import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import optimize
from functools import partial
import skrf as rf
import re

do_pn       = False
do_title    = True

# meas_dir = "/mnt/home/documents/Measurements/MPW2215_VCO/"
meas_dir = "/home/zoltan/ccn/Measurements/MPW2215_VCO/"
sim_dir = "/home/zoltan/ccn/Measurements/MPW2215_VCO/sim/"

style_sim  = {"linestyle":"--", "color":"grey", "marker":"None"} #, "label":"sim"} # for plot
style_meas1 = {"linestyle":"-", "edgecolor":None,"facecolor":"blue",  "s":20, "marker":"o"} #, "label":"meas", "marker":"o"}
style_meas2 = {"linestyle":"-", "edgecolor":None,"facecolor":"green", "s":20, "marker":"v"} #, "label":"meas", "marker":"o"}
style_meas3 = {"linestyle":"-", "edgecolor":None,"facecolor":"orange", "s":20, "marker":"s"} #, "label":"meas", "marker":"o"}
style_meas4 = {"linestyle":"-", "edgecolor":None,"facecolor":"purple", "s":20, "marker":"^"} #, "label":"meas", "marker":"o"}
# style_meas1 = {"linestyle":"-", "edgecolor":None,"facecolor":"blue",  "s":20, "marker":"o"} #, "label":"meas", "marker":"o"}
# style_meas2 = {"linestyle":"-", "edgecolor":None,"facecolor":"green", "s":20, "marker":"d"} #, "label":"meas", "marker":"o"}
# style_meas = {"linestyle":":", "color":"black", "marker":"o","s":75, "label":"meas"}
legend_style = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":3, "loc":'lower center', "mode":"expand", "handlelength":2, "bbox_to_anchor":(0.16,1.0,0.86,0.04)}
legend_style2 = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":1, "loc":'lower center', "mode":"expand", "handlelength":2}
legend_style2_ncol2 = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":2, "loc":'lower center', "mode":"expand", "handlelength":2}
legend_style2_ncol2_ul = {"frameon":False, "fontsize":7, "handletextpad":0.4, "borderaxespad":0, "ncol":2, "loc":'upper left', "mode":"expand", "handlelength":0.6}
vdd_ib_annot = {"s":r"V\textsubscript{DD} I\textsubscript{b}:","xy":(0,1.12),"xycoords":"axes fraction","fontsize":8}
l_style_meas = [style_meas1, style_meas2,style_meas3,style_meas4]

plot_style = {"figsize":(3.3914487339144874, 2.0960305886619515*0.7)}

xlabel = {"xlabel":r"Frequency $\left[\si{\GHz}\right]$", "labelpad":0}
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
    matplotlib.rcParams['grid.alpha'] = 0.5

#######################################
#   Cable loss interpolation 
#######################################
f_loss = "Probe_842922_cable_05439601_dcblock_avg.s2p"
f_loss = "2tier_cable_SN05439598_B9805_CSR8_Thru_DCblock.s2p"
f_loss = "2tier_cable_SN06149358_D7032_CSR8_Thru_DCblock.s2p"
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
meas_xlsx = meas_dir + "vco_cmos_60G.xlsx"
df_meas = pd.read_excel(meas_xlsx, usecols="B:G", sheet_name="20200302", header=0).rename(columns={"IDD":"idd", "Pout [dBm]":"pout", "Vtune":"vtune", "VDD":"vdd", "Iosc[mA]":"ib", "Ibuf":"ibuf", "freq [GHz]":"freq"})
df_meas = pd.read_excel(meas_xlsx, usecols="B:G", nrows=70, sheet_name="20200522", header=0).rename(columns={"IDD":"idd", "Pout [dBm]":"pout", "Vtune":"vtune", "VDD":"vdd", "Iosc[mA]":"ib", "Ibuf":"ibuf", "freq [GHz]":"freq"})
# p_out supposed to be the DIFFERENTIAL output power available at the PADS
# The single ended output was measured
df_meas.pout += 3 - cable_loss(df_meas.freq*1e9) # it should be moved to the common part
df_meas["eff"] = eff(df_meas.pout, df_meas.vdd, df_meas.idd*1e-3, in_dBm=True) 

l_cond = [
        {"vdd":1,"ib":4},
        {"vdd":1.2,"ib":4},
        # {"vdd":1.4,"ib":7},
        {"vdd":1.4,"ib":6},
        {"vdd":1.6,"ib":6},
        ]

suffix = "cmos"
outfile = suffix + ".png"

df_sim1 = pd.read_csv(sim_dir + 'cmos_60G_vdd1p0V_ib4mA_tt_60C.vcsv', **opt_vcsv)
df_sim1['vdd'] = 1
df_sim1['ib'] = 4
df_sim1p2 = pd.read_csv(sim_dir + 'cmos_60G_vdd1p2V_ib4mA_tt_60C.vcsv', **opt_vcsv)
df_sim1p2['vdd'] = 1.2
df_sim1p2['ib'] = 4
df_sim1p4 = pd.read_csv(sim_dir + 'cmos_60G_vdd1p4V_ib6mA_tt_60C.vcsv', **opt_vcsv)
df_sim1p4['vdd'] = 1.4
df_sim1p4['ib'] = 6
df_sim1p6 = pd.read_csv(sim_dir + 'cmos_60G_vdd1p6V_ib6mA_tt_60C.vcsv', **opt_vcsv)
df_sim1p6['vdd'] = 1.6
df_sim1p6['ib'] = 6

# Create one dataframe for easier use
df_sim = pd.concat([df_sim1, df_sim1p2, df_sim1p4, df_sim1p6], ignore_index=True)
df_sim.loc[:,'freq'] *= 1e-9
df_sim.loc[:,'pout'] += 3
df_sim["eff"] = np.power(10,(df_sim.pout-df_sim.pdc)/10) * 100

######################################
# Tuning range
######################################
fig_s, ax_s = plt.subplots(**plot_style)
cnt = 0
tr_label = []
for cond in l_cond:
    mask = pd.DataFrame([df_meas[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfm = df_meas[mask]
    mask = pd.DataFrame([df_sim[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfs = df_sim[mask]

    assert dfm.ib.nunique()==1, "Multiple bias current values"
    ax_s.plot(dfs.vtune, dfs.freq, **style_sim) 
    ax_s.scatter(dfm.vtune, dfm.freq, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    cnt += 1

    fmin, fmax = dfm.freq.min(), dfm.freq.max()
    favg = (fmin+fmax)/2
    fper = (fmax-favg)/favg*100
    tr_label.append(r"$\SI{%.1f}{\GHz}\pm\SI{%.1f}{\percent}$" % (favg,fper))

ax_s.plot([],[],**style_sim, label="sim") # to add label
ax_s.set_xlim(-0.5, 1.6)
ax_s.set_ylim(48, None)
ax_s.set_xlabel(r"Tuning voltage $\left[\si{\volt}\right]$", labelpad=0)
ax_s.set_ylabel(r"Frequency $\left [ \si{\GHz} \right ]$")

ax_s.grid()
handles, labels = ax_s.get_legend_handles_labels()
# # checking order of the labels
print(handles)
print(labels)
# label_order = [1, 2, 0]
# handles = [handles[i] for i in label_order]
# labels  = [labels[i]  for i in label_order]
handles = [*handles[1:], handles[0]]
labels  = [*labels[1:],labels[0]]
leg = ax_s.legend(handles=handles, labels=labels, **legend_style) 
ax_s.legend(handles=handles[:len(handles)-1], labels=tr_label, **legend_style2, bbox_to_anchor=(0.57,0.0,0.4,0.2), title="Tuning range", labelspacing=0.3)
ax_s.add_artist(leg)
ax_s.annotate(**vdd_ib_annot)
for ext in ["png","pgf"]:
    fig_s.savefig(meas_dir + suffix + "_tune." + ext, bbox_inches='tight', pad_inches = 0)

######################################
# Output power
######################################
# The differential output power is plotted
fig_s,ax_s = plt.subplots(**plot_style)
pout_label = []
cnt = 0

for cond in l_cond:
    mask = pd.DataFrame([df_meas[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfm = df_meas[mask]
    mask = pd.DataFrame([df_sim[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfs = df_sim[mask]

    assert dfm.ib.nunique()==1, "Multiple bias current values"
    ax_s.plot(dfs.freq, dfs.pout, **style_sim) 
    ax_s.scatter(dfm.freq, dfm.pout, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    cnt += 1
    p_avg = 10*np.log10(sum(10**(dfm.pout/10))/float(len(dfm.pout))) 
    pout_label.append("%.1f-%.1f; %.1fdBm" % (dfm.pout.min(), dfm.pout.max(), p_avg)   )


ax_s.plot([],[],**style_sim, label="sim") # to add label
ax_s.set_xlabel(**xlabel)
ax_s.set_ylabel(r"P\textsubscript{out} $\left[ \si{\dBm} \right ]$")
ax_s.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2.5))
ax_s.set_xlim(48,67)
ax_s.set_ylim(-2.5,11.5)
ax_s.grid()

handles, labels = ax_s.get_legend_handles_labels()
# # checking order of the labels
# print(handles)
# print(labels)
# label_order = [1, 2, 0]
# handles = [handles[i] for i in label_order]
# labels  = [labels[i]  for i in label_order]
handles = [*handles[1:], handles[0]]
labels  = [*labels[1:],  labels[0]]
leg = ax_s.legend(handles=handles, labels=labels, **legend_style) 
ax_s.legend(handles=handles[:len(handles)-1], labels=pout_label, **legend_style2_ncol2, bbox_to_anchor=(0.025,0.6,0.8,0.4), title="P\\textsubscript{out}: min-max; avg")
ax_s.add_artist(leg)
ax_s.annotate(**vdd_ib_annot)
for ext in ["png","pgf"]:
    fig_s.savefig(meas_dir + suffix + "_pout." + ext, bbox_inches='tight', pad_inches = 0)


######################################
# Power consumption
######################################
# The differential output power is plotted
fig_s,ax_s = plt.subplots(**plot_style)
fig_s2,ax_s2 = plt.subplots(**plot_style)

pdc_label = []
pdc_label2 = []
cnt = 0
for cond in l_cond:
    mask = pd.DataFrame([df_meas[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfm = df_meas[mask]
    mask = pd.DataFrame([df_sim[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfs = df_sim[mask]
    assert dfm.ib.nunique()==1, "Multiple bias current values"

    ax_s.plot(dfs.freq, dfs.pdc, **style_sim)
    ax_s2.plot(dfs.freq, np.power(10,dfs.pdc/10), **style_sim)
    meas_pdc_mW = dfm.vdd * dfm.idd
    meas_pdc = 10*np.log10(meas_pdc_mW)
    ax_s.scatter(dfm.freq, meas_pdc, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    ax_s2.scatter(dfm.freq, meas_pdc_mW, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    cnt += 1

    pdc_label.append("\SIrange{%.1f}{%.1f}{}; \SI{%.1f}{\dBm}" % (meas_pdc.min(), meas_pdc.max(), meas_pdc.mean())   )
    pdc_label2.append("\SIrange{%.1f}{%.1f}{}; \SI{%.1f}{\mW}" % (meas_pdc_mW.min(), meas_pdc_mW.max(), meas_pdc_mW.mean())   )


ax_s.plot([],[],**style_sim, label="sim") # to add label
ax_s2.plot([],[],**style_sim, label="sim") # to add label
ax_s.set_xlabel(**xlabel)
ax_s.set_ylabel(r"P\textsubscript{DC} $\left [ \si{\dBm} \right ]$")
ax_s.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2.5))
ax_s.set_xlim(48,67)
ax_s.set_ylim(2.5,13.5)
ax_s.grid()
ax_s2.set_xlabel(**xlabel)
ax_s2.set_ylabel(r"P\textsubscript{DC} $\left [ \si{\mW} \right ]$")
ax_s2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax_s2.set_xlim(48,67)
ax_s2.set_ylim(4,36)
ax_s2.grid()

handles, labels = ax_s.get_legend_handles_labels()
# # checking order of the labels
# print(handles)
# print(labels)
hand = [*handles[1:], handles[0]]
lab = [*labels[1:], labels[0]]
leg = ax_s.legend(handles=hand, labels=lab, **legend_style) 
ax_s.legend(handles=handles[1:], labels=pdc_label, **legend_style2_ncol2, bbox_to_anchor=(0,-0.02,1,0.2), title="P\\textsubscript{DC}: min-max; avg", labelspacing=0.3)
ax_s.add_artist(leg)
leg = ax_s2.legend(handles=hand, labels=lab, **legend_style) 
ax_s2.legend(handles=handles[1:], labels=pdc_label2, **legend_style2_ncol2, bbox_to_anchor=(0,0.65,1,0.3), title="P\\textsubscript{DC}: min-max; avg", labelspacing=0.3)
ax_s2.add_artist(leg)
ax_s.annotate(**vdd_ib_annot)
ax_s2.annotate(**vdd_ib_annot)
for ext in ["png","pgf"]:
    fig_s.savefig(meas_dir + suffix + "_pdc." + ext, bbox_inches='tight', pad_inches = 0)
    fig_s2.savefig(meas_dir + suffix + "_pdc_mW." + ext, bbox_inches='tight', pad_inches = 0)


######################################
# DC to RF efficiency
######################################
# l_eff = [eff(p_out[i], VDD, IDD[i], in_dBm=True) for i in range(len(p_out))]
fig_s,ax_s = plt.subplots(**plot_style)
eff_label = []
cnt = 0

for cond in l_cond:
    mask = pd.DataFrame([df_meas[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfm = df_meas[mask]
    mask = pd.DataFrame([df_sim[key] == val for key, val in cond.items()]).T.all(axis=1)
    dfs = df_sim[mask]
    assert dfm.ib.nunique()==1, "Multiple bias current values"
    
    # ax_s.plot(dfs.freq, dfs.eff,**style_sim)
    ax_s.scatter(dfm.freq, dfm.eff, **l_style_meas[cnt], label=r"\SI{%.1f}{\volt} \SI{%d}{\mA}" % (cond["vdd"],cond["ib"])) 
    cnt += 1

    eff_min, eff_max, eff_mean = dfm.eff.min(), dfm.eff.max(), dfm.eff.mean()
    eff_label.append("%.f-%.f%%; %.f%%" % (eff_min, eff_max, eff_mean))

ax_s.plot([],[],**style_sim, label="sim") # to add label
ax_s.set_xlabel(**xlabel)
ax_s.set_ylabel(r"${\eta  \left[ \si{\percent} \right] }$")
ax_s.set_ylim(8,18)
ax_s.set_xlim(48,67)
ax_s.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
# ax_s.set_ylabel("eff [%]")
ax_s.grid()

handles, labels = ax_s.get_legend_handles_labels()
# # checking order of the labels
# print(handles)
# print(labels)
handles = [*handles[1:], handles[0]]
labels  = [*labels[1:],  labels[0]]
leg = ax_s.legend(handles=handles, labels=labels, **legend_style) 
ax_s.legend(handles=handles[1:], labels=eff_label, **legend_style2_ncol2_ul, bbox_to_anchor=(0.02,0.62,0.65,0.4), title="$\eta$: min-max; avg", labelspacing=0.3)
ax_s.add_artist(leg)
ax_s.annotate(**vdd_ib_annot)
# if suffix == "cmos_tl270u":
#   ax_s.text(max(f_tune)*0.97,30,"%.1f%% - %.1f%%\nAvg: %.1f%%" %( min(l_eff),max(l_eff),sum(l_eff)/float(len(l_eff))),ha="center")
for ext in ["png","pgf"]:
    fig_s.savefig(meas_dir + suffix + "_eff." + ext, bbox_inches='tight', pad_inches = 0)
