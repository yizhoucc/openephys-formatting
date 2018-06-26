import OpenEphys as op
import numpy as np
import os
import encode
from mlpy import writemda16i
import sys



cwd=os.getcwd()
sys.path.append(cwd)
sys.path.append('~/repos/mountainlab/old/packages/mountainsort/packages/pyms')
dump = []


for i in range(64):
    i+=1
    data=[]
    data=op.load('2044_2018-06-19_13-42-08/100_CH%s.continuous' %i)
    print('loaded %s'%i,'start cutting')
    # cutting
    timestamps = data['timestamps'][0:30*60*30000]


    #writing

    # encode, write to another openephys
    # encode.writecontinous(filepath='cut_CH%s.continuous'%i,timestamps=timestamps, header=data['header'])

# accumulate to N*M, to mda
dump.append(timestamps) #append to row, need to check if need transpose
#convert to mda,
writemda16i(dump, 'raw30min.mda')