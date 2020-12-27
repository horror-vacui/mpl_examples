# skrf can return Spars at a given frequency!

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re,os,csv
import skrf as rf
from scipy.signal import savgol_filter
from quantiphy import Quantity

# SN_SN_RSC_..dB.s2p
r_fn = re.compile(r"PN_I67-A-GSG-100_SN_(?P<sn>.*).s2p")
# in_dir = "/mnt/MA/Anleitungen_Dokumentation/Labor/Messdaten/INF_A_Probes/"

# Data format: [<config>,<atten_setup>,<meas_atten>]
l_tmp = []
fig_all, ax_all = plt.subplots()
# for f in os.listdir(os.path.abspath(in_dir)):
for f in os.listdir(os.path.abspath('.')):
    m = r_fn.search(f)
    if m:
        sn = m.group("sn")
        # ff = in_dir + f
        # print(ff)
        print(f)
        nw = rf.Network(f)
        freq = nw.f * 1e-9
        s21  = nw.s[:,1,0]
        s21_filt = savgol_filter(20*np.log10(np.abs(s21)),11,3)
        l_tmp.append([config,atten,20*np.log10(np.abs(nw['61.25ghz'].s[:,1,0][0]))])
        ax_all.plot(freq,s21_filt, label=sn)


ax_all.grid(alpha=0.5)
ax_all.set_xlabel("freq [GHz]")
ax_all.set_ylabel("gain [dB]")
leg = fig_all.legend(title="I67 A GSG probes",bbox_to_anchor=(0.98,0.95))
plt.setp(leg.get_title(),multialignment='center')
fig_all.tight_layout()
fig_all.subplots_adjust(right=0.7)
fig_all.show()
fig_all.savefig("I67_A_GSG.pdf")
plt.show()

with open("I67_atten_61p25GHz.csv","w",newline='\n') as f_out:
    wr = csv.writer(f_out, delimiter=",")
    wr.writerows([i[1:] for i in l_tmp])


