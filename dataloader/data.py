import numpy as np
import os
import torch
import torch.utils.data as data
import pdb
import pickle
from pathlib import Path
from scipy import signal
class AudioDataset(data.Dataset):
    
    def __init__(self,mode,pickle_dir,**STFT_args):
        super(AudioDataset, self).__init__()
        self.fs = STFT_args['fs']
        self.window = STFT_args['window']
        self.nperseg = STFT_args['length']
        self.noverlap = STFT_args['overlap']
        
        self.pickle_dir = list(Path(pickle_dir).glob('**/**/**/**/*.pickle'))
        # # check chunked audio signal
        # MAX_INT16 = np.iinfo(np.int16).max
        # test=  ref2 * MAX_INT16
        # test = test.astype(np.int16)
        # wf.write('sample_ref2.wav',16000,test)
    
    def STFT(self,time_sig):
        '''
        input : [T,Nch]
        output : [Nch,F,T]
        '''
        assert time_sig.shape[0] > time_sig.shape[1], "Please check the STFT input dimension, input = [T,Nch] "
        num_ch = time_sig.shape[1]
        for num_ch in range(num_ch):
            # scipy.signal.stft : output : [F range, T range, FxT components]
            _,_,stft_ch = signal.stft(time_sig[:,num_ch],fs=self.fs,window=self.window,nperseg=self.nperseg,noverlap=self.noverlap)
            # output : [FxT]
            stft_ch = np.expand_dims(stft_ch,axis=0)
            if num_ch == 0:
                stft_chcat = stft_ch
            else:
                stft_chcat = np.append(stft_chcat,stft_ch,axis=0)
        return stft_chcat


    def __getitem__(self,index):
        # STFT
        with open(self.pickle_dir[index], 'rb') as f:
            data_infos = pickle.load(f)
        mix = data_infos['mix']
        mix_stft = self.STFT(mix)
        ref1 = data_infos['ref1']
        ref1_stft = self.STFT(ref1)
        ref2 = data_infos['ref2']
        ref2_stft = self.STFT(ref2)

        # numpy to torch & reshpae [C,F,T] ->[C,T,F]

        mix_stft = torch.permute( torch.from_numpy(mix_stft),[0,2,1])
        ref1_stft = torch.permute( torch.from_numpy(ref1_stft), [0,2,1])
        ref2_stft = torch.permute( torch.from_numpy(ref2_stft), [0,2,1])
        
        return mix_stft, ref1_stft, ref2_stft

    
    def __len__(self):
        return len(self.pickle_dir)
