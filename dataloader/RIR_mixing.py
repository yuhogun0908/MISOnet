import pickle
import pdb
import os
import pathlib
import scipy.io.wavfile as wf
import numpy as np
import torch.nn.functional as F 
from tqdm import tqdm
from itertools import combinations
MAX_INT16 = np.iinfo(np.int16).max
# nbits = 16

def read_wav(fname,normalize=False,Direct=True):
    samps_cat = np.array([])
    if isinstance(fname,pathlib.Path):
            fname = str(fname)  
    if Direct:
        for idx in range(len(fname)):
            samp_rate, samps_int16 = wf.read(fname[idx])
            # l_wave = librosa.core.load 와 s_wave=wf.read()/ 2**(16-1)과 같음  
            samps = samps_int16.astype(np.float)
            if normalize:
                # samps = samps / MAX_INT16
                samps = samps / 2**(nbits-1)
            if idx == 0:
                samps_cat = np.hstack((samps_cat, samps))
            else:
                samps_cat = np.vstack((samps_cat, samps))
        return samp_rate, samps_cat.T
    else:
        samp_rate, samps_int16 = wf.read(fname)
        #samp_rate 이 원하는 fs와 다를경우 resampling하는 코드 추가하기
        samps = samps_int16.astype(np.float)
        if normalize:
            # samps = samps / MAX_INT16
            samps = samps / 2**(nbits-1)
        return samp_rate, samps

class chunkSplit(object):
    def __init__(self,num_spks,chunk_time,least_time,fs,normalize):
        self.chunk_time = chunk_time
        self.least_time = least_time
        self.fs = fs
        self.normalize = normalize
        self.num_spks = num_spks
    def Readwave(self,scp_dict):
        samps_ = {}
        samp_rate, samp_mix = read_wav(scp_dict['mix'],normalize=self.normalize,Direct=False)
        samp = {'mix':samp_mix}
        samps_.update(samp)
        for idx in range(self.num_spks):
            _,samp_ref = read_wav(scp_dict['ref'+str(idx+1)],normalize=self.normalize,Direct=False)
            samp ={'ref'+str(idx+1):samp_ref}
            samps_.update(samp)
        return samps_
    def Split(self,save_dir,samp_dict):
        split_idx = 0
        chunk_size= self.chunk_time*self.fs
        least_size = self.least_time*self.fs
        mix = samp_dict['mix']
        ref = [samp_dict['ref'+str(spk_idx+1)] for spk_idx in range(self.num_spks)]
        
        perm = list(combinations([x for x in range(self.num_spks)],2))
        perm = [list(x) for x in perm]
        
        for perm_element in range(len(perm)):
            assert  ref[perm[perm_element][0]].shape == ref[perm[perm_element][1]].shape, "[Shape Error] 'Length' or 'The number of ch' between reference waves is not equal"
        assert mix.shape == ref[0].shape, "[Shape Error] 'Length' or 'The number of ch' between mix and reference wavs is not equal"

        length = mix.shape[0]
        num_ch = mix.shape[1]

        if length < least_size:
            return
        if length > least_size and length < chunk_size:
            ref_ = [[] for spk_idx in range(self.num_spks)]
            split_samp = {}
            start = 0
            gap = chunk_size-length
            mix_ = np.pad(mix, ((0,gap),(0,0)), constant_values= 0)
            for spk_idx in range(self.num_spks):
                ref_[spk_idx] = np.pad(ref[spk_idx], ((0,gap),(0,0)), constant_values=0)
                split_samp['ref'+str(spk_idx+1)] = ref_[spk_idx]
            split_samp['mix'] = mix_
            # ref1_ = np.pad(ref1, ((0,gap),(0,0)), constant_values=0)
            # ref2_ = np.pad(ref2, ((0,gap),(0,0)), constant_values=0)
            # split_samp = {'mix':mix_ ,'ref1':ref1_, 'ref2':ref2_}
            with open(save_dir + '_'+ str(split_idx)+'.pickle', 'wb') as f:
                    pickle.dump(split_samp,f)       
        if length > chunk_size:
            ref_ = [[] for spk_idx in range(self.num_spks)]
            start = 0
            while True:
                split_samp = {}
                if start+chunk_size > length:
                    break
                mix_ = mix[start:start+chunk_size,:]
                for spk_idx in range(self.num_spks):
                    ref_[spk_idx] = ref[spk_idx][start:start+chunk_size,:]
                    split_samp['ref'+str(spk_idx+1)]= ref_[spk_idx]
                split_samp['mix'] = mix_
                # split_samp ={'mix':mix_ ,'ref1':ref1_, 'ref2':ref2_}
                with open(save_dir + '_' + str(split_idx)+'.pickle', 'wb') as f:
                    pickle.dump(split_samp,f)   

                # verify chunk audio signal
                # test=  mix_ * MAX_INT16
                # test = mix_ * 2**(nbits-1)
                # test = test.astype(np.int16)
                # wf.write('sample.wav',self.fs,test)
                # overlap 2s
                start += least_size
                split_idx += 1

class AudioSave(object):
    '''
    Class that reads wav format files and saves pickle format
    Input :
        scp_path(str) : scp file address
        sample rate(int, optional), default 16kHz
        chunk size(int)  : split audio size (time(s)*sample rate, default: 640000(4s) )
    Output :
        split audio(list)
    '''
    def __init__(self,mode,num_ch,num_spks,scp_path,wave_path, fs=16000, chunk_time=4, least_time=2):
        super(AudioSave, self).__init__()        
        self.mode = mode
        #train은 mix할때 하나의 source 기준으로 2번 mix
        #dev와 test는 1번 mix
        if self.mode == 'Train':
            self.num_dup = 1
        else:
            self.num_dup = 1
        self.wave_path = wave_path
        self.fs = fs
        self.splitter = chunkSplit(num_spks,chunk_time,least_time,fs,normalize=True)
        self.num_spks = num_spks
        self.num_ch = num_ch

        if mode == 'Train':
            tr_list = 'audio_si_tr.lst'
            with open(scp_path + tr_list) as f:
                self.lines = f.read().splitlines()
        elif mode == 'Development':
            self.lines = []
            dt_list = ['audio_si_dt5b.lst', 'audio_si_dt5a.lst']
            for dt in dt_list:
                with open(scp_path + dt) as f:
                    self.lines.extend(f.read().splitlines())
        elif mode == 'Evaluation':
            self.lines = []
            et_list = ['audio_si_et_1.lst', 'audio_si_et_2.lst']
            for et in et_list:
                with open(scp_path + et) as f:
                    self.lines.extend(f.read().splitlines())


        self.array_types = os.listdir(wave_path)
        # self.All_split_samps_dict = {}

    def save(self,array_info,save_pickle_dir):
        '''
            split audio with chunk size and least size
        '''
        if not os.path.exists(save_pickle_dir):
            os.makedirs(save_pickle_dir)

        for array_idx in tqdm(array_info):
            array = [s for s in self.array_types if array_idx in s]
            p_dir = self.wave_path + array[0]
            for key in tqdm(self.lines):
                if not os.path.exists(os.path.join(save_pickle_dir+array_idx)+str(pathlib.Path(key).parent.absolute())):
                    os.makedirs(os.path.join(save_pickle_dir+array_idx)+str(pathlib.Path(key).parent.absolute()))
                wave_path = pathlib.Path(p_dir + key)
                p_wave_path = wave_path.parent.absolute()
                # os.listdir(p)
                
                for f_idx in range(self.num_dup):
                    temp_dict = {}
                    paths_dict = {}
                    for spk_idx in range(self.num_spks):
                        temp_dict['ref'+str(spk_idx+1)] = pathlib.Path(str(wave_path) + '_Direct' + str(spk_idx+1) + '.wav')

                    temp_mix = pathlib.Path(str(wave_path) + '_Mixed.wav')
                    temp_dict['mix'] = temp_mix
                    paths_dict.update(temp_dict)

                    samp_dict= self.splitter.Readwave(paths_dict)
                    save_dir  = os.path.join(save_pickle_dir+array_idx)+key+'_'+str(f_idx)
                    self.splitter.Split(save_dir,samp_dict)              
        
def main_rirmixing(mode,num_ch, fs, chunk_time, least_time,num_spks,scp_path,wave_path,save_pickle_dir):
    REVERB_SAVE = AudioSave(mode,num_ch,num_spks,scp_path,wave_path,fs,chunk_time,least_time)
    array_info = ['no_reverb']
    REVERB_SAVE.save(array_info,save_pickle_dir)