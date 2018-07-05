import OpenEphys as op
import numpy as np
import os
# import encode
# from mlpy import writemda16i
import sys
import filter
import matplotlib.pyplot as plt
import threshold
import math
import spikedetekt
import waveform
cwd=os.getcwd()
sys.path.append(cwd)
sys.path.append('~/repos/mountainlab/old/packages/mountainsort/packages/pyms')
# sys.path.append('D:\\repos\\klusta\\klusta\\traces')




def load(path, nchan,cutmin):
    ndpoints=cutmin*60*30000
    dim=int(math.floor((ndpoints/1024)))
    raw=np.empty((dim*1024, nchan))

    for i in range(nchan):
        i+=1
        # dump=[]
        data = op.loadContinuous(str(path)+'\\100_CH%s.continuous' %i,stop_record=ndpoints/1024)
        print('loaded %s'%i,'start cutting')
        # cutting
        timestamps = data['data']#[0:cutmin*60*30000]
        raw[:,i-1] = (timestamps)

    return raw

        #writing

        # encode, write to another openephys
        # encode.writecontinous(filepath='cut_CH%s.continuous'%i,timestamps=timestamps, header=data['header'])

    # accumulate to N*M, to mda
    # dump.append(timestamps) #append to row, need to check if need transpose
    # #convert to mda,
    # writemda16i(dump, 'raw30min.mda')
def fandw(raw):
    fil = filter.Filter(rate=30000., low=500., high=9000., order=4)
    traces_f = fil(raw)

    super_threshold_indices = traces_f > 150
    traces_f[super_threshold_indices] = 0
    super_threshold_indices = traces_f < -150
    traces_f[super_threshold_indices] = 0
    return traces_f


def plot(x):
    plt.plot(x)
    plt.show()

def stdfactor(data):
    std=np.std(data)
    return std

def thresh(whittened):

    # thresholdarray = threshold.compute_threshold(whittened, single_threshold=False)

    # sort = threshold.Thresholder(mode='negative')
    #
    # thresh = threshold.compute_threshold(whittened,single_threshold=True,std_factor=stdfactor(whittened))
    #
    # crossing = sort(whittened,{'weak':5,'strong':20})

    # weak = sort.detect(threshholded,'weak')
    # strong = sort.detect(threshholded,'strong')


    sigma_n_ch1 = np.median(np.absolute((whittened[:,0]) / 0.6745))

    sigma_n_ch2 = np.median(np.absolute((whittened[:,1]) / 0.6745))
    Thr_ch1 = 5 * sigma_n_ch1
    Thr_ch2 = 5 * sigma_n_ch2

    super_threshold_indices_ch1 = whittened > Thr_ch1

    super_threshold_indices_ch2 = whittened > Thr_ch2



    # flood = threshold.FloodFillDetector()
    # detection = flood(weak_crossings=weak, strong_crossings=strong)
    return super_threshold_indices_ch1, super_threshold_indices_ch2




def wave(components):
    # w = waveform.Waveform()
    waveform.extract_waveform(component=components)




    # testing
'''
import loader
raw= loader.load('D:\\2044_2018-05-31_12-50-24', 2,1)
raw = loader.fandw(raw)

d = loader.thresh(raw)
     
    
loader.plot(raw)

'''
# std=loader.stdfactor(raw)