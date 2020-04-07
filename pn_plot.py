import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.interpolate import interp1d
from scipy import optimize
import os.path
import re, os, logging

# my_circuit =  "cmos_60G_tl270u"
my_circuit =  "cmos_60G_sf"
# my_circuit =  "cmos_60G"
# my_circuit =  "nmos_60G_sf"
w_dac = False
w_short = True
do_show_all =False
# annotate_vdd = True
annotate_vdd = False

# style_sim  = {"linestyle":"--", "color":"grey", "marker":"None", "label":"sim"} # for plot
style_sim  = {"linestyle":"--", "color":"grey", "marker":"None", "label":"simulation"} # for plot

pn_dir = "/mnt/home/documents/Measurements/MPW2215_VCO/Phase_noise/"
pn_sim_dir = "/mnt/home/documents/Design/pictures/vco_mpw2215/"
if my_circuit == "cmos_60G_sf":
    # re_pn = re.compile("(?P<circuit>.*)_set_values_vtune(?P<vtune>[0-9]+(.[0-9]+)?)V_vdd(?P<vdd>[0-9]p[0-9])V_ib(?P<ib>[0-9])mA_ibuf(?P<ibuf>[0-9]+)uA(?P<note>.*)") # older meas
    re_pn = re.compile("(?P<circuit>.*)_vtune(?P<vtune>[0-9]+(.[0-9]+)?)V_vdd(?P<vdd>[0-9]p[0-9])V_ib(?P<ib>[0-9])mA_ibuf(?P<ibuf>[0-9]+)uA(?P<note>.*)")
elif my_circuit == "nmos_60G_sf":    
    re_pn = re.compile("(?P<circuit>.*)_vtune(?P<vtune>[0-9]+(.[0-9]+)?)V_vdd(?P<vdd>[0-9]p[0-9])V_ib(?P<ib>[0-9]+)mA?_ibuf(?P<ibuf>[0-9]+)uA(?P<note>.*)")
else:    
    re_pn = re.compile("(?P<circuit>.*)_vtune(?P<vtune>[0-9]+(.[0-9]+)?)V_vdd(?P<vdd>[0-9]p[0-9])V_ib(?P<ib>[0-9]+)mA_(?P<note>.*)")


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

# vcsv_cols is used to select the columns corresponding to the different supply voltages. It was easier than writing a parser, because the name of the vcsv columns might be changed by hand --> the universal formula has to handle different ways of defining the VDD.
d_sim = {'nmos_60G_sf':{ 'f_sim'     : 'nmos_sf_pn_all_supply_10mA_cc_only.vcsv',
                         'vcsv_cols' : {'0p8':1,'1p0':3,'1p2':5}  
                         }, # nmos_60G_sf
        'cmos_60G_sf':{ 'f_sim'     : 'cmos_sf_pn_3mA_275uA_allsupply.vcsv',
                         'vcsv_cols' : {'0p8':1,'1p0':3,'1p2':5,'1p4':7}
                         }, # cmos_60G_sf
        'cmos_tl270u_60G':{ 'f_sim'     : 'cmos_tl270u_pn_all_supply_4mA.vcsv',
                         'vcsv_cols' : {'1p0':1,'1p2':3,'1p4':5,'1p6':7}
                         }, # cmos_tl270u_60G
        'cmos_60G_tl270u':{ 'f_sim'     : 'cmos_tl270u_pn_all_supply_4mA.vcsv',
                         'vcsv_cols' : {'1p0':1,'1p2':3,'1p4':5,'1p6':7}
                         }, # cmos_tl270u_60G
        'cmos_60G':      { 'f_sim'     : 'cmos_60G_pn_merged.csv',
                         'vcsv_cols' : {'1p0':1,'1p2':3,'1p4':5,'1p6':7}
                         }, # cmos_60G
                         }

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
        "pgf.texsystem": "lualatex",
        "pgf.preamble": [
            "\\usepackage{siunitx}",         
            "\\usepackage{amsmath}",         
            "\\usepackage{metalogo}",
            "\\usepackage{unicode-math}",  # unicode math setup
            r"\DeclareSIUnit{\dBc}{dBc}",
            r"\DeclareSIUnit[per-mode=symbol]{\dBcHz}{\dBc\per\Hz}",
            r"\newcommand{\da}{\textsuperscript{$\dagger$}}"
            r"\setmathfont{xits-math.otf}",
            r"\setmainfont{DejaVu Serif}", # serif font via preamble
            ]
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
    matplotlib.rcParams['lines.markersize'] = 4


# building list of available measurement results
l_vdd = []
for i in os.listdir(pn_dir):
    m = re_pn.match(i)
    if m:
        circuit = m.group("circuit")
        if circuit == my_circuit:
            vdd   = m.group("vdd")
            if vdd not in l_vdd:
                l_vdd.append(vdd)
logger.info('Available measurement results: ' + ", ".join(l_vdd))
l_vdd_sim = [ i.replace('.','p') for i in d_sim.get(my_circuit).get('vcsv_cols').keys()]

for my_vdd in set().union(l_vdd, l_vdd_sim):    
    fig, ax = plt.subplots(figsize=(3.3914487339144874, 2.0960305886619515*0.85))
    # plotting the sim
    logger.info("VDD=" + my_vdd + " is being processed")
    if my_vdd in l_vdd_sim: # and False:
        d_sim_vdd = d_sim.get(my_circuit)
        logger.debug(f"d_sim_vdd: f{d_sim_vdd}")
        my_cols = [0]
        my_cols.append( d_sim_vdd.get('vcsv_cols').get(my_vdd) ) 
        logger.debug(f"VDD={my_vdd};\tUsed csv columns: {my_cols}")
        df_sim = pd.read_csv(pn_sim_dir + d_sim_vdd.get('f_sim'), header=None, skiprows=6, dtype=np.float64, usecols=my_cols, names=['f','pn'])
        logger.info(f"Adding simulation: {d_sim_vdd.get('f_sim')}")
        df_sim.f *= 1e-6
        ax.plot(df_sim.f, df_sim.pn, **style_sim)

        # x,pn10 = df_sim[df_sim.f==10].values[0]
        # ax.text(x-7.5,pn10-10,"$\SI{%.0f}{\dBcHz}$" % pn10, fontsize=7, color=style_sim.get("color"))
        # ax.scatter(x,pn10, marker='o',color=style_sim.get("color"),s=20)

        x,pn1 = df_sim[df_sim.f==1].values[0]
        ax.text(x-0.7,pn1-7,"$\SI{%.0f}{\dBcHz}" % pn1, fontsize=7, color=style_sim.get("color"))
        ax.scatter(x,pn1, marker='o',color=style_sim.get("color"),s=20)

        if annotate_vdd:
            ax.annotate(r"\SI{"+my_vdd.replace('p','.')+r"}{\volt}", xycoords="axes fraction", xy=(0.1,0.1), fontsize=7)
            ax.annotate(my_circuit, xycoords="axes fraction", xy=(0.1,0.9), fontsize=7)

    if my_vdd in l_vdd: # and my_vdd == "0p8":
        df_avg = pd.DataFrame()
        l_df = []
        for i in os.listdir(pn_dir):
            m = re_pn.match(i)
            if m:        
                # logger.debug("Matching file name: %s" % i)
                circuit = m.group("circuit")
                if circuit == my_circuit:
                    # logger.debug(f"{circuit} in {i}")
                    vtune   = m.group("vtune")
                    vdd     = m.group("vdd")
                    ib      = m.group("ib")
                    if "_sf" in circuit:
                        ibuf    = m.group("ibuf")
                    note    = m.group("note")
                    file_basename = pn_dir + "%s_%sV_%sV_%smA_pn" % (my_circuit, my_vdd, vtune, ib)
                    
                    if vdd == my_vdd:
                        f_csv = pn_dir + i
                        # logger.info(f"Processing file {f_csv}")
                        df = pd.read_csv(f_csv, skiprows=77, names=['f','pn'], delimiter=";", dtype=np.float64, header=None, usecols=[0,1], index_col=False )
                        df['f'] *= 1e-6
                        l_df.append(df)
                        if do_show_all:
                            df.plot(x='f', y='pn', logx=True, grid=True, ax=ax, label=note)
                        else:
                            # df.plot(x='f', y='pn', logx=True, grid=True, ax=ax, kind='scatter', alpha=0.005, color="k")
                            ax.plot(df.f, df.pn, alpha=0.05, color="k")
                        x,pn10 = df[df.f==10].values[0]
                        x,pn1 = df[df.f==1].values[0]
                        logger.debug(f"{os.path.basename(i)}\t%.1f %.1f" % (pn1, pn10))


        if annotate_vdd:
            ax.annotate(r"\SI{" + my_vdd.replace('p','.') +r"}{\volt}", xycoords="axes fraction", xy=(0.1,0.1), fontsize=7)
            ax.annotate(my_circuit, xycoords="axes fraction", xy=(0.1,0.9), fontsize=7)


        if not do_show_all:
            df_avg = pd.concat(l_df).groupby(level=0).mean().sort_values(by=['f'])
            df_avg.to_csv(file_basename + ".csv", index=False )

            ax.plot(df_avg.f, df_avg.pn, color="k") # , label="meas" )
            
            func = interp1d(df_avg.f, df_avg.pn, kind='cubic')
            x,pn1 = 1,func(1)
            ax.text(x-0.1, pn1+5,"$\SI{%.0f}{\dBcHz}" % pn1, fontsize=7)
            ax.scatter(x,pn1, marker='o',color='k',s=20)


    ax.set_xlabel(r"$\Delta f \left[\si{\MHz}\right]")
    ax.set_ylabel(r"$\mathcal{L}(f_0) \left[\si{\dBcHz} \right]$")
    ax.set_xscale("log")
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:n}'))
    ax.set_xlim(0.1,100)
    ax.set_ylim(-120,-50)
    if my_circuit == "nmos_60G_sf":
        # ax.set_xlim(0.3,100)
        pass
    # else:
    if do_show_all:
        ax.legend()
        suffix = "_all"
    else:
        suffix = ""

    for ext in [".pgf",".png"]:
    # for ext in [".png"]:
        file_basename = pn_dir + "%s_%sV_pn" % (my_circuit, my_vdd)
        fig.savefig(file_basename + suffix + ext, bbox_inches="tight", dpi=600)

plt.show()

