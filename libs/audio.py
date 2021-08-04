import scipy.io.wavfile as wf
import numpy as np
import os
import pdb

def read_wav(fname,normalize=True):
    samp_rate, samps_int16 = wf.read(fname)
    pdb.set_trace()

class chunkSplit(object):
    def __init__(self,chunk_time,least_time,fs):
        self.chunk_time = chunk_time
        self.least_time = least_time
        self.fs = fs
        self.normalize = normalize

    def Split(self,Mixed,ref_1,ref_2):
        samp_rate, samps = read_wav(Mixed,normalize=self.normlaize)


