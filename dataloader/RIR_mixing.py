import pickle
import pdb
import os
from pathlib import Path
import scipy.io.wavfile as wf
import numpy as np
import torch.nn.functional as F 
from tqdm import tqdm

# MAX_INT16 = np.iinfo(np.int16).max
nbits = 16

def read_wav(fname,normalize=False):
    samps_cat = np.array([])
    if len(fname) != 1:
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
        samp_rate, samps_int16 = wf.read(fname[0])
        #samp_rate 이 원하는 fs와 다를경우 resampling하는 코드 추가하기
        samps = samps_int16.astype(np.float)
        if normalize:
            # samps = samps / MAX_INT16
            samps = samps / 2**(nbits-1)
        return samp_rate, samps

class chunkSplit(object):
    def __init__(self,chunk_time,least_time,fs,normalize):
        self.chunk_time = chunk_time
        self.least_time = least_time
        self.fs = fs
        self.normalize = normalize
    def Readwave(self,num_spks,scp_dict):
        samps_ = {}
        samp_rate, samp_mix = read_wav(scp_dict['mix'],normalize=self.normalize)
        samp = {'mix':samp_mix}
        samps_.update(samp)
        for idx in range(num_spks):
            _,samp_ref = read_wav(scp_dict['ref'+str(idx+1)],normalize=self.normalize)
            samp ={'ref'+str(idx+1):samp_ref}
            samps_.update(samp)
        return samps_
    def Split(self,save_dir,samp_dict):
        split_idx = 0
        chunk_size= self.chunk_time*self.fs
        least_size = self.least_time*self.fs
        mix = samp_dict['mix']
        ref1 = samp_dict['ref1']
        ref2 = samp_dict['ref2']
        assert (mix.shape == ref1.shape) or (ref1.shape == ref2.shape), "length or the number of ch of mix,reference wave is not equal"
        length = mix.shape[0]
        num_ch = mix.shape[1]
        if length < least_size:
            return
        if length > least_size and length < chunk_size:
            start = 0
            gap = chunk_size-length
            mix_ = np.pad(mix, ((0,gap),(0,0)), constant_values= 0)
            ref1_ = np.pad(ref1, ((0,gap),(0,0)), constant_values=0)
            ref2_ = np.pad(ref2, ((0,gap),(0,0)), constant_values=0)
            split_samp = {'mix':mix_ ,'ref1':ref1_, 'ref2':ref2_}
            with open(save_dir + '_'+ str(split_idx)+'.pickle', 'wb') as f:
                    pickle.dump(split_samp,f)       

        if length > chunk_size:
            start = 0
            while True:
                if start+chunk_size > length:
                    break
                mix_ = mix[start:start+chunk_size,:]
                ref1_ = ref1[start:start+chunk_size,:]
                ref2_ = ref2[start:start+chunk_size,:]

                split_samp ={'mix':mix_ ,'ref1':ref1_, 'ref2':ref2_}
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
    def __init__(self,mode,num_spks,scp_path,wave_path, fs=16000, chunk_time=4, least_time=2):
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
        self.splitter = chunkSplit(chunk_time,least_time,fs,normalize=True)
        self.num_spks = num_spks

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
                if not os.path.exists(os.path.join(save_pickle_dir+array_idx)+str(Path(key).parent.absolute())):
                    os.makedirs(os.path.join(save_pickle_dir+array_idx)+str(Path(key).parent.absolute()))
                wave_path = Path(p_dir + key)
                p_wave_path = wave_path.parent.absolute()
                # os.listdir(p)
                for f_idx in range(self.num_dup):
                    temp_direct1 = []
                    temp_direct2 = []
                    paths_dict = {}
                    temp_direct1.append(Path(str(wave_path) + '_Direct1.wav'))
                    temp_direct2.append(Path(str(wave_path) + '_Direct2.wav'))    
                    temp_mix = Path(str(wave_path) + '_Mixed.wav')
                    temp_dict = {'mix': [temp_mix],'ref1': temp_direct1, 'ref2':temp_direct2}
                    paths_dict.update(temp_dict)

                    samp_dict= self.splitter.Readwave(self.num_spks,paths_dict)
                    save_dir  = os.path.join(save_pickle_dir+array_idx)+key+'_'+str(f_idx)
                    self.splitter.Split(save_dir,samp_dict)              
        
def main_rirmixing(mode,fs, chunk_time, least_time,num_spks,scp_path,wave_path,save_pickle_dir):
    REVERB_SAVE = AudioSave(mode,num_spks,scp_path,wave_path,fs,chunk_time,least_time)
    array_info = ['no_reverb']
    REVERB_SAVE.save(array_info,save_pickle_dir)