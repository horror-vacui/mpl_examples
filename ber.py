""" Post-process a square wave signal waveform from oscilloscope dump
    Rhode & Schwarz RTO... 
    Dump: one csv file contains the recorrding details
    Another one the x,y coordinates
"""

import csv
import pandas as pd
import numpy as np
import argparse, logging, time, datetime, pathlib, subprocess
from importlib import reload

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=pathlib.Path, help="csv file of the dump.")
args = parser.parse_args()

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

reload(logging)
meas_date = time.strftime("%Y_%m_%d")      
t_script_start = datetime.datetime.now().replace(microsecond=0)
s_now = f"{t_script_start:%Y%m%d_%H%M}"
log_file = f"ber_{s_now}"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)

th_low  = 0.5
th_high = 1.3
th_mid  = 0.9
error_value = 666


# if the file contains only the y coordinates:
y_col = None
fdata = pathlib.Path(args.csv.with_suffix('.Wfm.csv'))
if fdata.exists():
    logging.debug(fdata)
    # RTO case
    logging.info("RTO case")
    fcsv = fdata
    csv_delim = ';'
    n_header = 0
    config = {}
    with open(args.csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=':', quotechar='|')
        for row in reader:
            tmp = row[1:-1]
            if isinstance(tmp, list) & len(tmp)==1:
                tmp = tmp[0]
            config[row[0]] = tmp
    # logging.debug(config)
    resolution = float(config['Resolution'])
    Ndata = int(config['RecordLength'])
    with open(fcsv,'r') as f:
        line = f.readline()
        y_col = len(line.split(csv_delim))
        logging.debug(line)
        logging.debug(line.split(csv_delim))
        logging.debug(f"y_col={y_col}")
else:
    # RTM 2004 case
    logging.info("RTM case")
    fcsv = args.csv
    csv_delim = ',' # for small R&S scope
    n_header = 1
    Ndata = line_count(fcsv) - n_header
    y_col = 1 # I know that it has two coloumns
    with open(fcsv,'r') as f:
        line = f.readline()
        line = f.readline() # 2nd row, no header
        t0 = float(line.split(csv_delim)[0])
        line = f.readline()
        t1 = float(line.split(csv_delim)[0])
        resolution = t1-t0
    logging.debug(f"y_col={y_col}")



freq = 640e3
Nper  = Ndata*resolution*freq 


# if False:
y = np.genfromtxt(fcsv,
        dtype=float, 
        delimiter=csv_delim,
        names=None, 
        usecols=y_col, # why was here a -1?
        skip_header= n_header,
        )

def last_valid_state(y, i):
    if y[i] > th_high:
        valid_state = 1
    elif y[i]< th_low:
        valid_state = 0
    else:
        valid_state = last_valid_state(y, i-1)
    # print(f"y={y[i]} logic high? {y[i]>th_high};\tstate: {valid_state}")
    return valid_state

# initial state
if y[0] > th_mid:
    last_state = 1
elif y[0] < th_low:
    last_state = 0
else:
    last_state = error_value
    raise ValueError('initial state is unknown')

# Let's count the edges. It seems easier than measuring the time difference between the edges
# actually I should do both
cnt_up = 0
cnt_dn = 0
for i in range(1,Ndata):
    # print(i)
    state = last_valid_state(y,i)
    # print(state)
    if not state == last_state:
        if state ==1:
            cnt_up  += 1
            last_state = 1
        elif state ==0:
            cnt_dn += 1
            last_state = 0
        else:
            print(y[i-5:i+1])
            raise ValueError("invalid state")

print(f"There was {cnt_up} rising and {cnt_dn} folling edges in the dataset. Expected value: {Nper}")

