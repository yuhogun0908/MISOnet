import numpy as np
import os
import torch
import torch.utils.data as data
import pdb
import pickle
from pathlib import Path
from scipy import signal
import librosa
import scipy

class AudioDataset(data.Dataset):
    
    def __init__(self,mode,num_spks,pickle_dir,**STFT_args):
        super(AudioDataset, self).__init__()
        self.fs = STFT_args['fs']
        self.window = STFT_args['window']
        self.nperseg = STFT_args['length']
        self.noverlap = STFT_args['overlap']
        self.num_spks = num_spks
        self.pickle_dir = list(Path(pickle_dir).glob('**/**/**/**/*.pickle'))
        self.mode = mode

        hann_win = scipy.signal.get_window('hann', self.nperseg)
        self.scale = np.sqrt(1.0 / hann_win.sum()**2)
        # self.pickle_dir = self.pickle_dir[0:10]

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
        f.close()

        mix = data_infos['mix']
        mix_stft = self.STFT(mix)
        mix_stft = mix_stft/self.scale # scale equality between scipy stft and matlab stft

        assert self.num_spks+1 == len(data_infos), "[ERROR] Check the number of speakers"
        ref_stft = [[] for spk_idx in range(self.num_spks)]
        for spk_idx in range(self.num_spks):
            ref_sig = data_infos['ref'+str(spk_idx+1)]
            if len(ref_sig.shape) == 1:
                ref_sig = np.expand_dims(ref_sig,axis=1)
            ref_stft[spk_idx] = torch.permute(torch.from_numpy(self.STFT(ref_sig)),[0,2,1])
            ref_stft[spk_idx] = ref_stft[spk_idx]/self.scale # scale equality between scipy stft and matlab stft
       
        # numpy to torch & reshpae [C,F,T] ->[C,T,F]

        mix_stft = torch.permute( torch.from_numpy(mix_stft),[0,2,1])

        if self.mode == 'Beamforming':
            # hann_win = scipy.signal.get_window('hann', self.nperseg)
            # scale = np.sqrt(1.0 / hann_win.sum()**2)
            # mix, fs = librosa.load('./TEST1/mix.wav', mono = False, sr= 8000)
            # mix = mix.T
            # mix_stft = self.STFT(mix)
            # mix_stft = mix_stft/scale
            # s1, fs = librosa.load('./TEST1/s1.wav', mono = False, sr= 8000)
            # s1 = s1.T
            # S1 = self.STFT(s1)
            # S1 = S1/scale
            # s2, fs = librosa.load('./TEST1/s2.wav', mono = False, sr= 8000)
            # s2 = s2.T
            # S2 = self.STFT(s2)
            # S2 = S2/scale
            # BeamOutDir = str(self.pickle_dir[index]).replace('CleanMix','Beamforming')

            # return mix_stft, ref_stft, S1, S2, BeamOutDir

            saveDir = str(self.pickle_dir[index]).replace('CleanMix','Beamforming')
            return mix_stft, ref_stft, saveDir

        elif self.mode == 'Enhance':
            #BeamOutDir = str(self.pickle_dir[index]).replace('CleanMix','Beamforming')
            BeamOutDir = '/home/data/DBhogun/SMS_WSJ_DB/train/Beamforming/16449_4abc020w_49bc0305_0.pickle'
            with open (BeamOutDir, 'rb') as f:
                beam_infos = pickle.load(f)
                
                s1_bf = beam_infos['S1_BF']
                s2_bf = beam_infos['S2_BF']
            f.close()

            # Temp Code
            hann_win = scipy.signal.get_window('hann', self.nperseg)
            scale = np.sqrt(1.0 / hann_win.sum()**2)
            mix, fs = librosa.load('./TEST1/mix.wav', mono = False, sr= 8000)
            mix = mix.T
            mix_stft = self.STFT(mix)
            mix_stft = torch.from_numpy(mix_stft/scale)
            mix_stft = torch.permute(mix_stft,[0,2,1])
            s1, fs = librosa.load('./TEST1/s1.wav', mono = False, sr= 8000)
            s1 = s1.T
            S1 = self.STFT(s1)
            S1 = torch.from_numpy(S1/scale)
            S1 = torch.permute(S1,[0,2,1])
            s2, fs = librosa.load('./TEST1/s2.wav', mono = False, sr= 8000)
            s2 = s2.T
            S2 = self.STFT(s2)
            S2 = torch.from_numpy(S2/scale)
            S2 = torch.permute(S2,[0,2,1])
            return mix_stft, ref_stft, s1_bf, s2_bf, S1, S2
            
            # return mix_stft, ref_stft, s1_bf, s2_bf

        else:
            return mix_stft, ref_stft

    
    def __len__(self):
        return len(self.pickle_dir)
