import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, pi, abs
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from functools import partial

outdir="/home/zoltan/publications/interferometer/trunk/pictures/"

do_plot_3d = False
do_plot_rhos = False

if False:
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

l_tl=0.384
l_tl = 405/2158
l_tl=0.5
# guided wavelength: 2158um @61GHz
# Z0 = 58.50 - j*1.94
# epsilon: 5.20
# L`=4.439e-7
# R`=11717.64
# C`=1.2989e-10
# G`=0.1208

l_ym = []
l_xm = []
l_l = np.linspace(0.01, 0.5, 50) 
x = np.linspace(0, 0.5, 50)
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
def pd_sum(n, x, l, r, rho):
    """ Sum of the peak detector outputs
        Note: Since x is swept, no phsae shift constant is necessary to be included

        Paramters:
        ----------
        x:  array
            location of the first tap point in electrical length;
        l:  float
            the total length of the measurment medium in electrical length
        n:  integer
            number of the equidistant tap points
        r:  float in [0,1]
            the ratio of the amplitudes
        rho: float in [0,2*pi)
            the phase difference between the waves
    """
    s = 0
    t = l / (n-1)
    for i in range(n):
        s += abs(cos( 2*pi*(i*t -x) )) - r*abs(cos(2*pi*(i*t -x)-rho))
    return s

def pd_sum1(n, x, l, r, rho):
    s = 0
    t = l / (n-1)
    for i in range(n):
        s += abs(cos( 2*pi*(i*t -x) )) 
    return s

def pd_sum2(n, x, l, r, rho):
    s = 0
    t = l / (n-1)
    for i in range(n):
        s += r*abs(cos(2*pi*(i*t -x)-rho))
    return s

def sum_minmax(n, x, l):
    """ Worst case of the standing wave for 'n' detectors. 

        Paramters:
        ----------
        x:  array
            location of the first tap point in electrical length;
        l:  float
            the total length of the measurment medium in electrical length
        n:  integer
            number of the equidistant tap points
        r:  float in [0,1]
            the ratio of the amplitudes
        rho: float in [0,2*pi)
            the phase difference between the waves
    """
    a_min = np.full(len(x),999)
    i_r = 0.5
    if True:
    # for i_r in np.linspace(0, 0.7, 10, endpoint=False):
        i_rho=0.0
        # for i_rho in np.linspace(0, 2*pi, 10, endpoint=False):
        if True:
            i = pd_sum(n=n,x=x,l=l,r=i_r, rho=i_rho)
            print(f"l={l:.3} n={n} r={i_r:.2} rho={i_rho:.3} min(sum)={i.min():.3}")
            a_min = np.minimum(i.min(),a_min)
    print("--------------------")
    print(a_min)
    return a_min

def my_formatter(x, pos):
    if x.is_integer():
        # return f"{x:.0}"
        return str(int(x))
    else:
        return f"{x:.1}"
###############################################

if do_plot_3d:
    fig, ax = plt.subplots()
    ax = fig.gca(projection='3d')
    n = 4
    X, Y = np.meshgrid(range(100),range(100), indexing='ij')
    z = np.zeros(X.shape)
    for r in X[:,0]:
        for rho in Y[0,:]:
            z[r,rho] = pd_sum(x=x, n=n, l=l_tl, r=r/8, rho=rho/4/2/pi).min()
    l_x = x[:]
    m = np.full(X.shape,999)
    for x in l_x:
        p = pd_sum(x=x, n=n, l=l_tl, r=X/len(X), rho=Y*2*pi/len(Y))
        m = np.minimum(m,p)



    # surf = ax.plot_surface(X,Y,z, cmap=cm.coolwarm)
    surf = ax.plot_surface(X/len(X),Y*2*pi/len(Y),m, cmap=cm.coolwarm)
    ax.set_xlabel(r"$r$", labelpad=2)
    ax.set_ylabel(r"$\rho$", labelpad=2)
    fig.colorbar(surf, shrink=0.75, aspect=5)
    plt.show()
    # ==> Conclusion: rho barely changes the results

if do_plot_rhos:
    n_plots = 8
    n_columns = 4
    fig, ax = plt.subplots(int(n_plots/n_columns), n_columns, figsize=(16,8))
    n=4
    cnt=0
    r=0.3
    rho=np.linspace(0,2*pi,100)
    l_y2 = []
    l_n = np.linspace(0.05,0.4,8,endpoint=True)
    for l in l_n:
        l_y = []
        for i in rho:
            l_y.append(pd_sum(n=n,x=x,l=l,r=r,rho=i).min())
        l_y2.append(l_y)

    for i in range(n_plots):
        row = int(i / n_columns)
        col = i % n_columns
        ax[row,col].plot(rho,l_y2[i],label=f"n={l_n[i]:.3}")
        ax[row,col].set_title(f"l={l_n[i]:.3}")

    for i in range(n_columns):
        ax[1,i].set_xlabel(r"$\rho$", labelpad=2)

    for i in range(int(n_plots/n_columns)):
        ax[i,0].set_ylabel(r"$\min \left \{ \Delta V_{\Sigma}(x) \right \}$", labelpad=2)

    # ax[0,0].plot(rho,l_y2[0],label=f"n={l_n[0]:.3}")
    # ax[0,0].set_title(f"l={l_n[0]:.3}")
    # ax[0,1].plot(rho,l_y2[1],label=f"n={l_n[1]:.3}")
    # ax[0,1].set_title(f"l={l_n[1]:.3}")
    # ax[1,0].plot(rho,l_y2[2],label=f"n={l_n[2]:.3}")
    # ax[1,0].set_title(f"l={l_n[2]:.3}")
    # ax[1,1].plot(rho,l_y2[3],label=f"n={l_n[3]:.3}")
    # ax[1,1].set_title(f"l={l_n[3]:.3}")
    fig.suptitle(f"r={r:.1}, n={n}")
    plt.tight_layout()
    # fig.savefig(outdir + f"sw_amp_diff_r{r:.1}_n{n}")
    plt.show()


####################################
# Plot the two part of the difference separately
# X axis: the electrical length of the tl
# assume r=0.1, plot multiple rho values, n=4
# fig, ax = plt.subplots(int(n_plots/n_columns), n_columns, figsize=(16,8))
# fig, ax = plt.subplots()
# n = 4
# r = 0.1
# x = np.linspace(0, 0.5, 50)
# ax.plot(x, pd_sum1(n=n, x=x, r=r, rho=0, l=l_tl), linestyle="--")
# for rho in np.linspace(0,pi,7):
#     ax.plot(x, abs(pd_sum(n=n, x=x, r=r, rho=rho, l=l_tl)), linestyle="-")
#     ax.plot(x, pd_sum2(n=n, x=x, r=r, rho=rho, l=l_tl), linestyle=":")
# plt.show()

l_tl = 0.5
r=0.0
rho=0.0
for r in np.arange(0,0.61,0.1):
    for rho in np.arange(0,pi+0.01,pi/20):
        for l_tl in np.arange(0.05,0.51,0.05):
            N=10
            fig, ax = plt.subplots()
            ax.set_xlim(0,0.5)
            ax.set_ylim(0,N)
            ax.set_xlabel("position of the SW maximum to the transmission line")
            ax.set_ylabel("SW & $V_{\Sigma}$ normalized to SW amplitude")
            fig.suptitle(f"T-line length: {l_tl}; r: {r:.0f} $\\rho$: {rho:.0f}")
            ax.grid(which='both')
            x=np.linspace(0, 0.5, 50)
            lines  = [ax.plot([],[],label='det No.{i}')[0] for i in range(N+2)] 
            lines2  = lines[:]
            for line in lines2:
                plt.setp(line, linestyle=":", alpha=0.7)
            lines.append(ax.plot([],[], 'k-', label="sum", alpha=0.7)[0])
            hlines = [ax.plot([min(x),max(x)],[0,0], linestyle=":", color='black',alpha=0.7)[0] for _ in range(2)]
            my_text  = ax.text(.35, 7.5, '')
            dV_text  = ax.text(.35, 7.0, '')
            patches = lines + [my_text]
            def init():
                for line in lines:
                    line.set_data([],[])
                my_text.set_text('')
                return patches

            def animate(n):
                m = n+2
                s = np.zeros(x.shape)
                for i in range(m):
                    yp =   abs(cos( 2*pi*(l_tl/(m-1)*i -x) ))
                    ym = r*abs(cos( 2*pi*(l_tl/(m-1)*i -x)-rho))
                    y = abs(yp - ym)
                    lines[i].set_data(x,y)
                    lines[i].set_label("det No. {i}")
                    # ax.plot(x,y,label=f"{n}")
                    s+=y
                for i in range(m,N+2):
                    lines[i].set_data([],[])
                lines[N+2].set_data(x,s)
                hlines[0].set_data([min(x),max(x)],[min(s),min(s)])
                hlines[1].set_data([min(x),max(x)],[max(s),max(s)])

                my_text.set_text(f"N={m}")
                dV_text.set_text("$\Delta V_{\Sigma}=%.2f$" % ( max(s)-min(s))) #,  200*(max(s)-min(s))/(max(s)+min(s)) )   )

            ani = FuncAnimation(fig, animate, interval=500, frames=N-2) 
            #, blit=True)

            # # Set up formatting for the movie files
            # Writer = matplotlib.animation.writers['ffmpeg']
            # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            # ani.save("/home/zoltan/publications/interferometer/trunk/pictures/det_sum_vs_Ndet_Xinput_phase_r0_rho0.mp4", writer=writer)
            ani.save(f"/home/zoltan/publications/interferometer/trunk/pictures/det_sum_vs_Ndet_Xinput_phase_lenTL{l_tl*100:.0f}_r{r:.0f}_rho{rho/pi*180:.0f}.gif", writer=matplotlib.animation.ImageMagickFileWriter())
            plt.close(fig)
# plt.show()

#ax.grid(True,which='minor', alpha=0.1, axis='both')
#ax.grid(True,which='major', alpha=0.7, axis='both')
#ax.set_xlabel(r"Electrical length of the transmission line ($l$)", labelpad=2)
#ax.set_ylabel(r"$\min \left ( V_{\Sigma}(x, \rho, x_0) \right )$", labelpad=2)
#ax.tick_params(axis='both', which='major', pad=2)
#ax.set_xlim(0,0.5)
#ax.set_ylim(bottom=0,top=6)
#ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))
#ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
## markers
#m = ['o','s','^','v','x']
#me = 50

#for i in range(2,7):
#    # smallest value for the sum of the peak detector outputs, for different line lengths (l_l)
#    # it is swept for different starting phases and the smallest value is selected
#    ymin = np.array([sum_minmax(i,x,l)[0] for l in l_l])
#    ax.plot(l_l, ymin, label=i, marker=m[i-2], markevery=(int(me*(i-1)/3), me))
## The two ranges are separated to plot the with the same color
#ax.set_prop_cycle(None)
#for i in range(2,7):
#    ymax = np.array([sum_minmax(i,x,l)[1] for l in l_l])
#    # ax.plot(l_l, ymax, marker=m[i-2], markevery=(int(me*(i-1)/3), me), linestyle="--")
#    ax.plot(l_l, ymax, markevery=(int(me*(i-1)/3), me), linestyle=":")

#legend_style = {"frameon":False,  "handletextpad":0.4, "borderaxespad":0, "ncol":5, "loc":'lower center', "mode":"expand", "handlelength":1}
#ax.legend(bbox_to_anchor=(0.1, 1.0, 0.9, 0.04), **legend_style)
#ax.annotate("$N$:", xy=(0.01,1.05), xycoords="axes fraction")
#ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(my_formatter))
#for ext in ("png", "pgf"):
#    fig.savefig(outdir + "interferometer_calc." + ext)
################################################
#if False:
#    n = 4
#    a=0
#    color = ["#7fc97f","#beaed4","#fdc086","#ffff99","#386cb0","#f0027f","#bf5b17"]
#    for l_tl in np.linspace(0.25,0.5,21):
#        for i in range(n):
#            plt.plot(x, abs(cos(2*pi*(i*l_tl/(n-1) - x))))
#        plt.plot(x, pd_sum(n,x,l_tl), label=l_tl)

#        plt.xlabel("SW peak location")
#        plt.ylabel("")
#        plt.grid()
#        plt.savefig(f"./interfero_n4_tl{l_tl:.2}sum.png")
