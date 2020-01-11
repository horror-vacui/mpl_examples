# This script takes two different files for the PSS (tran) average voltages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('/home/tibenszky/.config/matplotlib/pgf.mplstyle')

f_char = 
df_char = pd.read_csv("/mnt/home/documents/Measurements/MPW2221_TSPC_div/sim/etspc_inv_DC_char.vcsv", 
        header=None, skiprows=6, dtype=float)
df_avg = pd.read_csv("/mnt/home/documents/Measurements/MPW2221_TSPC_div/sim/etspc_tran_node_avg.vcsv",
        dtype=float, header=None, skiprows=6), 
        index_col=None, usecols=np.array([-1,0,2,4,6,8,10])+1,names=["vclk_dc","A0","B0","FB0","A3","B3","FB3"])

#####################################################################################################################################
# plotting setup
# color and linestyle definition
matplotlib.rcParams['lines.markersize'] = 3
d_plot = {
        'A' : ("^",'#1b9e77'),
        'B' : ("v",'#d95f02'),
        'FB': ("s","#7570b3")
        }
d_bb = {
        (3,-1) : ":",
        # (0,0)  : "--",
        (0,0)  : "-",
        }


IEEE_width = 516 #pt
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

s = set_size(IEEE_width, fraction=0.21, height_fraction=1)
fig, ax = plt.subplots(1,1,sharex=True, figsize=(s[0],s[1]))

#####################################################################################################################################
# plotting 
ax.plot(df_char[0],df_char[1],linestyle=d_bb.get((0,0)),color="black")
ax.plot(df_char[2],df_char[3],linestyle=d_bb.get((3,-1)),color="black")
ax.plot(df_char[0],df_char[0],linestyle=":", color="grey", alpha=0.5)
ax.grid(which='major', alpha=0.5)
ax.grid(which='minor', alpha=0.2)

fig2, ax2 = plt.subplots(1,1,sharex=True, figsize=(s[0],s[1]))
ax2.plot(df_avg['vclk_dc'],df_avg['A0'],  linestyle=d_bb.get((0,0)),  marker=d_plot.get('A')[0],  color=d_plot.get('A')[1] )   
ax2.plot(df_avg['vclk_dc'],df_avg['B0'],  linestyle=d_bb.get((0,0)),  marker=d_plot.get('B')[0] , color=d_plot.get('B')[1] )    
ax2.plot(df_avg['vclk_dc'],df_avg['FB0'], linestyle=d_bb.get((0,0)),  marker=d_plot.get('FB')[0], color=d_plot.get('FB')[1])    
ax2.plot(df_avg['vclk_dc'],df_avg['A3'],  linestyle=d_bb.get((3,-1)), marker=d_plot.get('A')[0] , color=d_plot.get('A')[1] )    
ax2.plot(df_avg['vclk_dc'],df_avg['B3'],  linestyle=d_bb.get((3,-1)), marker=d_plot.get('B')[0] , color=d_plot.get('B')[1] )    
ax2.plot(df_avg['vclk_dc'],df_avg['FB3'], linestyle=d_bb.get((3,-1)), marker=d_plot.get('FB')[0], color=d_plot.get('FB')[1])    

# Adding a subtle reference line
xlim = ax2.get_xlim()
x = np.linspace(xlim[0],xlim[1],num=50)
ax2.plot(x,x,linestyle=":", color="grey", alpha=0.5)

#####################################################################################################################################
# lets create now the handles. They are the description, text together with colored linestyles explaining what is what in the figure.
backbias_handle = []
backbias_handle.append(matplotlib.lines.Line2D([],[], color="black", linestyle=d_bb.get((0,0)), label="0V 0V"))
backbias_handle.append(matplotlib.lines.Line2D([],[], color="black", linestyle=d_bb.get((3,-1)), label="3V -1V"))

node_handle = []
node_handle.append(matplotlib.lines.Line2D([],[], marker=d_plot.get('A')[0], color=d_plot.get('A')[1], linestyle="-", label="A"))
node_handle.append(matplotlib.lines.Line2D([],[], marker=d_plot.get('B')[0], color=d_plot.get('B')[1], linestyle="-", label="B"))
node_handle.append(matplotlib.lines.Line2D([],[], marker=d_plot.get('FB')[0], color=d_plot.get('FB')[1], linestyle="-", label="FB"))

ax.annotate(r"V\textsubscript{BN} V\textsubscript{BP}:",xy=(-0.1,1.08),xycoords="axes fraction",fontsize=7)
leg_bb = ax.legend(handles = backbias_handle, frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(0.24,0.98,0.86,0.04), borderaxespad=0, ncol=2, loc='lower center', mode="expand", handlelength=1)

ax2.annotate(r"Nodes: ",xy=(-0.17,1.06),xycoords="axes fraction",fontsize=7)
leg_node = ax2.legend(handles = node_handle, frameon=False, fontsize=7, handletextpad=0.4, bbox_to_anchor=(0.12,0.98,0.9,0.04), borderaxespad=0, ncol=3, loc='lower center', mode="expand", handlelength=1.75)


#####################################################################################################################################
# Finishing up the plots
ax.set_ylabel(r"V\textsubscript{out} [V]")
ax.set_xlabel(r"V\textsubscript{in} [V]",labelpad=0)
ax2.set_ylabel(r"V\textsubscript{avg} [V]")
ax2.set_xlabel(r"V\textsubscript{in_avg} [V]",labelpad=0)
ax2.grid(which='major', alpha=0.5)
ax2.grid(which='minor', alpha=0.2)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.set_xticks(np.arange(0.2,0.7,0.1))
ax.set_xlim(0,0.8)
ax.set_ylim(0,0.8)
ax2.set_xlim(0.2,0.6)
ax2.set_ylim(0.2,0.7)

#####################################################################################################################################
# Saving the files. The final pgf file will be loaded into a latex document.
# For faster and esier checking how to plot looks like, we save it also into png format.
# Unfortunately the pgf backend has no interactive mode to see the plot.
for ext in ["png", "pgf"]:
    # fig.savefig("etspc_dc.%s" % ext, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    # fig2.savefig("etspc_pss_avg.%s" % ext, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    fig.savefig("etspc_dc.%s" % ext, bbox_inches='tight')
    fig2.savefig("etspc_pss_avg.%s" % ext, bbox_inches='tight')
