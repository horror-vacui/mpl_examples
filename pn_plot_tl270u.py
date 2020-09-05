import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.interpolate import interp1d
from scipy import optimize
import os.path
import re, os, logging

my_circuit =  "cmos_60G_tl270u"
do_show_all =False
# annotate_vdd = True
annotate_vdd = False
plot_sim = False

fig_style = {"figsize":(3.3914487339144874, 2.0960305886619515*0.8)}
# style_sim  = {"linestyle":"--", "color":"grey", "marker":"None", "label":"simulation"} # for plot

pn_dir      = "/home/zoltan/ccn/Measurements/MPW2215_VCO/Phase_noise/"
pn_sim_dir  = "/home/zoltan/ccn/Measurements/MPW2215_VCO/sim/"

re_pn = re.compile("(?P<circuit>.*)_vtune(?P<vtune>[0-9]+(.[0-9]+)?)V_vdd(?P<vdd>[0-9]p[0-9])V_ib(?P<ib>[0-9]+)mA_(?P<note>.*)")


logger = logging.getLogger("pn_plot")
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
d_sim = { 'f_sim'     : 'cmos_tl270u_pn_all_supply_20200419.vcsv',
                         'vcsv_cols' : {'1p0':7,'1p2':5,'1p4':3,'1p6':1}
                         }
d_color = { '1p0': '#0037ff', # blue
            '1p6': '#ff7300'  # its complementary color: orange
          }
d_txt_offset = { '1p0': (-0,5),
                 '1p6': (-0.8,-8)
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
        "axes.grid" :True,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction":"in",
        "ytick.direction":"in",
        "xtick.major.size":1.5,
        "ytick.major.size":1.5,
        "savefig.pad_inches":0,
        "savefig.bbox":"tight",
        "savefig.dpi":600,
        'lines.markersize':4,
        "pgf.texsystem": "lualatex",
        "pgf.preamble": [
            "\\usepackage{siunitx}",         
            "\\usepackage{amsmath}",         
            "\\usepackage{metalogo}",
            "\\usepackage{unicode-math}",  # unicode math setup
            r"\DeclareSIUnit{\dBc}{dBc}",
            r"\DeclareSIUnit[per-mode=symbol]{\dBcHz}{\dBc\per\Hz}",
            r"\newcommand{\da}{\textsuperscript{$\dagger$}}"
            # r"\setmathfont{xits-math.otf}",
            # r"\setmainfont{DejaVu Serif}", # serif font via preamble
            ]
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)
    matplotlib.rcParams['axes.unicode_minus'] = False


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
l_vdd_sim = [ i.replace('.','p') for i in d_sim.get('vcsv_cols').keys()]

fig, ax = plt.subplots(**fig_style)
for my_vdd in set().union(l_vdd, l_vdd_sim).intersection(d_color.keys()):    
    # plotting the sim
    logger.info("VDD=" + my_vdd + " is being processed")
    if my_vdd in l_vdd_sim and plot_sim: # and False:
        d_sim_vdd = d_sim.get(my_circuit)
        logger.debug(f"d_sim_vdd: f{d_sim_vdd}")
        my_cols = [0]
        my_cols.append( d_sim_vdd.get('vcsv_cols').get(my_vdd) ) 
        logger.debug(f"VDD={my_vdd};\tUsed csv columns: {my_cols}")
        df_sim = pd.read_csv(pn_sim_dir + d_sim_vdd.get('f_sim'), header=None, skiprows=6, dtype=np.float64, usecols=my_cols, names=['f','pn'])
        logger.info(f"Adding simulation: {pn_sim_dir + d_sim_vdd.get('f_sim')}")
        df_sim.f *= 1e-6
        ax.plot(df_sim.f, df_sim.pn, **style_sim)

        # x,pn10 = df_sim[df_sim.f==10].values[0]
        # ax.text(x-7.5,pn10-10,"$\SI{%.0f}{\dBcHz}$" % pn10, fontsize=7, color=style_sim.get("color"))
        # ax.scatter(x,pn10, marker='o',color=style_sim.get("color"),s=20)

        func = interp1d(df_sim.f, df_sim.pn, kind='cubic')
        x,pn1 = 1,func(1)
        # x,pn1 = df_sim[df_sim.f==1].values[0]
        ax.text(x-0.73,pn1-9,"$\SI{%.1f}{\dBcHz}" % pn1, fontsize=7, color=style_sim.get("color"))
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
                            df.plot(x='f', y='pn', logx=True, grid=True, ax=ax, label=note, color=d_color.get(my_vdd))
                        else:
                            # df.plot(x='f', y='pn', logx=True, grid=True, ax=ax, kind='scatter', alpha=0.005, color="k")
                            ax.plot(df.f, df.pn, alpha=0.05, color=d_color.get(my_vdd))
                        x,pn10 = df[df.f==10].values[0]
                        x,pn1 = df[df.f==1].values[0]
                        logger.debug(f"{os.path.basename(i)}\t%.1f %.1f" % (pn1, pn10))


        if annotate_vdd:
            ax.annotate(r"\SI{" + my_vdd.replace('p','.') +r"}{\volt}", xycoords="axes fraction", xy=(0.1,0.1), fontsize=7)
            ax.annotate(my_circuit, xycoords="axes fraction", xy=(0.1,0.9), fontsize=7)


        if not do_show_all:
            df_avg = pd.concat(l_df).groupby(level=0).mean().sort_values(by=['f'])
            df_avg.to_csv(file_basename + ".csv", index=False )
            ax.plot(df_avg.f, df_avg.pn, color=d_color.get(my_vdd), label=r'\SI{' +my_vdd.replace("p",".") + r'}{\volt}' )
            
            func = interp1d(df_avg.f, df_avg.pn, kind='cubic')
            x,pn1 = 1,func(1)
            off = d_txt_offset.get(my_vdd)
            ax.text(x+off[0], pn1+off[1],"$\SI{%.1f}{\dBcHz}" % pn1, fontsize=7, color=d_color.get(my_vdd))
            ax.scatter(x,pn1, marker='o',color=d_color.get(my_vdd),s=20)


ax.set_xlabel(r"$\Delta f \left[\si{\MHz}\right]",labelpad=0)
ax.set_ylabel(r"$\mathcal{L}(f_0) \left[\si{\dBcHz} \right]$",labelpad=1)
ax.set_xscale("log")
ax.grid(which='major', alpha=0.5)
ax.grid(which='minor', alpha=0.2)
ax.legend(frameon=False) #, title=r"V\textsubscript{DD}" )
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:n}'))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.set_xlim(0.1,100)
ax.set_ylim(-130,-50)
if do_show_all:
    ax.legend()
    suffix = "_all"
else:
    suffix = ""

for ext in [".pgf",".png"]:
# for ext in [".png"]:
    file_basename = pn_dir + "%s_pn" % (my_circuit)
    fig.savefig(file_basename + suffix + ext)

if not latex:
    plt.show()

