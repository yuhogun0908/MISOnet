import glob
import pickle
import pdb
import os
import scipy.io.wavfile as wf
import numpy as np
import torch.nn.functional as F 
from tqdm import tqdm
from itertools import combinations
from pathlib import Path
import librosa
from multiprocessing import Pool, cpu_count

MAX_INT16 = np.iinfo(np.int16).max
# nbits = 16

def read_wav(filedir,samplingrate):
    if isinstance(filedir,Path):
        filedir = str(filedir)
    # samp_rate, samps_int16 = wf.read(filedir)
    # samps = samps_int16.astype(np.float)
    # if normalize:
        # samps = samps / 2**(nbits-1)
        # samps = samps / MAX_INT16
    wav, fs = librosa.load(filedir, mono = False, sr= samplingrate)
    wav = wav.T

    return wav



def chunkSplit(num_spks,num_ch, chunk_time,least_time, fs, CleanMixsaveDir, earlysaveDir, tailsaveDir,noisesaveDir, wavpath_dict, fname, normalize=True):

    fname = fname.rstrip('.wav')
    ''' read wave '''
    mix = read_wav(wavpath_dict['mix'], fs) # mix
    ref1 = read_wav(wavpath_dict['ref1'], fs)
    ref2 = read_wav(wavpath_dict['ref2'], fs)
    ref = [ref1, ref2]
    # early1 = read_wav(wavpath_dict['early1'], fs)
    # early2 = read_wav(wavpath_dict['early2'], fs)
    # early = [early1, early2]
    # tail1 = read_wav(wavpath_dict['tail1'], fs)
    # tail2 = read_wav(wavpath_dict['tail2'], fs)
    # tail = [tail1, tail2]
    # noise = read_wav(wavpath_dict['noise'], fs)

    ''' chunk wave '''

    split_idx = 0
    chunk_size = int(chunk_time * fs)
    least_size = int(least_time * fs)

    length = mix.shape[0]
    num_ch = mix.shape[1]


    ref_pad = [[] for spk_idx in range(num_spks)]
    # early_pad = [[] for spk_idx in range(num_spks)]
    # tail_pad = [[]for spk_idx in range(num_spks)]

    if length < least_size:
        return
    if length > least_size and length < chunk_size:        
        start = 0
        split_cleanmix = {}
        # split_early = {}
        # split_tail = {}
        # split_noise = {}



        gap = chunk_size-length

        mix_pad = np.pad(mix, ((0,gap),(0,0)), constant_values= 0)
        split_cleanmix['mix'] = mix_pad
        
        # noise_pad = np.pad(noise, ((0, gap), (0,0)), constant_values=0) 
        # split_noise['noise'] = noise_pad
  
        for spk_idx in range(num_spks):
            ref_pad[spk_idx] = np.pad(ref[spk_idx], ((0,gap),(0,0)), constant_values=0)
            split_cleanmix['ref'+str(spk_idx+1)] = ref_pad[spk_idx]
            
            # early_pad[spk_idx] = np.pad(early[spk_idx], (0,gap), constant_values=0)
            # split_early['early'+str(spk_idx+1)] = early_pad[spk_idx]
            
            # tail_pad[spk_idx] = np.pad(tail[spk_idx], (0,gap), constant_values=0)
            # split_tail['tail'+str(spk_idx+1)] = tail_pad[spk_idx]
            
        with open(os.path.join(CleanMixsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
            pickle.dump(split_cleanmix,f)       
        # with open(os.path.join(earlysaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
            # pickle.dump(split_early,f)   
        # with open(os.path.join(tailsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
            # pickle.dump(split_tail,f)   
        # with open(os.path.join(noisesaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
            # pickle.dump(split_noise,f)

    if length > chunk_size:
        start = 0
        ref_split = [[] for spk_idx in range(num_spks)]
        # early_split = [[] for spk_idx in range(num_spks)]
        # tail_split = [[] for spk_idx in range(num_spks)]
        
        while True:
            split_cleanmix = {}
            # split_early = {}
            # split_tail = {}
            # split_noise = {}

            if start+chunk_size > length:
                break
            mix_split = mix[start:start+chunk_size,:]
            split_cleanmix['mix'] = mix_split
            
            # noise_split = noise[start:start+chunk_size,:]
            # split_noise['noise'] = noise_split
            
            for spk_idx in range(num_spks):
                ref_split[spk_idx] = ref[spk_idx][start:start+chunk_size,:]
                split_cleanmix['ref'+str(spk_idx+1)]= ref_split[spk_idx]

                # early_split[spk_idx] = early[spk_idx][start:start+chunk_size,:]
                # split_early['early'+str(spk_idx+1)]= early_split[spk_idx]

                # tail_split[spk_idx] = tail[spk_idx][start:start+chunk_size,:]
                # split_tail['early'+str(spk_idx+1)]= tail_split[spk_idx]
            
            with open(os.path.join(CleanMixsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                pickle.dump(split_cleanmix,f)       
            # with open(os.path.join(earlysaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                # pickle.dump(split_early,f)   
            # with open(os.path.join(tailsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                # pickle.dump(split_tail,f)   
            # with open(os.path.join(noisesaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                # pickle.dump(split_noise,f)
            
            # verify chunk audio signal
            # test=  mix_split * MAX_INT16
            # test = mix_split * 2**(nbits-1)
            # test = test.astype(np.int16)
            # wf.write('sample.wav',self.fs,test)
            # overlap 2s

            start += least_size
            split_idx += 1
            



class main_smswsj:
    def __init__(self, num_spks, num_ch, chunk_time, least_time, fs, rootDir,saveRootDir, cleanDir, mixDir, earlyDir, tailDir, noiseDir, trFile, devFile, testFile):
        mode = {'train' : trFile,'dev' : devFile, 'test':testFile}
        # mode = {'dev' : devFile, 'test':testFile}
        for mode_idx in mode:
            self.CleanMixsaveDir = os.path.join(saveRootDir,mode_idx,'CleanMix')
            Path(self.CleanMixsaveDir).mkdir(exist_ok = True, parents=True)
            
            self.mixrootDir = os.path.join(rootDir, mixDir, mode[mode_idx])
            
            self.cleanrootDir = os.path.join(rootDir, cleanDir, mode[mode_idx])
            
            self.earlyrootDir = os.path.join(rootDir, earlyDir, mode[mode_idx])
            self.earlysaveDir = os.path.join(saveRootDir,mode_idx,earlyDir) 
            Path(self.earlysaveDir).mkdir(exist_ok = True, parents=True)
            
            self.tailrootDir = os.path.join(rootDir, tailDir, mode[mode_idx])
            self.tailsaveDir = os.path.join(saveRootDir,mode_idx,tailDir)
            Path(self.tailsaveDir).mkdir(exist_ok = True, parents=True)
            
            self.noiserootDir = os.path.join(rootDir, noiseDir, mode[mode_idx])
            self.noisesaveDir = os.path.join(saveRootDir,mode_idx,noiseDir)
            Path(self.noisesaveDir).mkdir(exist_ok = True, parents=True)

            self.mixfiles = os.listdir(self.mixrootDir)
            
            self.num_spks =num_spks; self.num_ch=num_ch; self.chunk_time = chunk_time; self.least_time = least_time; self.fs = fs

            ''' multi process ''' 
            print('[Extraction] Cropping & Save')
            cpu_num = cpu_count()

            arr = list(range(len(self.mixfiles)))
            with Pool(cpu_num) as p:
                r = list(tqdm(p.imap(self.process,arr), total= len(arr), ascii=True, desc = 'Extraction : Cropping & Save'))


    def process(self,idx):
        file_ =  self.mixfiles[idx]       
        # for file_ in tqdm(mixfiles):
        wavpath_dict = {}
        for spk_idx in range(2):  # SMS_WSJ dataset has two spk
            ref_name = file_.replace('.wav','_{}.wav'.format(spk_idx))
            ref_path = os.path.join(self.cleanrootDir,ref_name)
            wavpath_dict['ref'+str(spk_idx+1)] = Path(str(ref_path))

            early_path = os.path.join(self.earlyrootDir,ref_name)
            wavpath_dict['early'+str(spk_idx+1)] = Path(str(early_path))
            tail_path = os.path.join(self.tailrootDir, ref_name)
            wavpath_dict['tail' + str(spk_idx+1)] = Path(str(tail_path))

        mix_path = os.path.join(self.mixrootDir, file_)
        wavpath_dict['mix'] = mix_path
        
        noise_path = os.path.join(self.noiserootDir, file_)
        wavpath_dict['noise'] = noise_path
        chunkSplit(self.num_spks,self.num_ch, self.chunk_time,self.least_time, self.fs, self.CleanMixsaveDir, self.earlysaveDir, self.tailsaveDir,self.noisesaveDir, wavpath_dict,file_,normalize=True)

        
        



        



        # noise wav가 clean, mix 신호와 길이가 같은지 확인하기

