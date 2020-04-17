import csv
import itertools

def vcsv_cols(fn_vcsv):
    """ Takes a file object of a vcsv file and returns the column names.
        
        Colunm names are stored in the second row
        """
    with open(fn_vcsv, 'r') as f:
        # row2 = next(itertools.islice(csv.reader(f, delimiter=','), 1, 2))
        row2 = next(itertools.islice(csv.reader(f), 1, 2))  # list of ';Yname'
        row2b = [i.split(';') for i in row2]    # list of ['', 'Yname]] 
        rows = [item for sublist in row2b for item in sublist] # flat list
    return [x if x != '' else 'X' for x in rows]


if __name__ == "__main__":
    # filename_vcsv = "~/publications/vco_cmos_sf/bin/varactor_Q_TR_vs_L_2.vcsv"
    filename_vcsv = "/home/zoltan/publications/vco_cmos_sf/bin/varactor_Q_TR_vs_L_2.vcsv"
    print(vcsv_cols(filename_vcsv))


