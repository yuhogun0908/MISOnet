import glob
import pickle
import pdb
import os
from typing_extensions import ParamSpec
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

def chunkSplit(save_flag, num_spks, num_ch, chunk_time, least_time, fs, CleanMixsaveDir, earlysaveDir, tailsaveDir, noisesaveDir, MISO1saveDir, BeamformingsaveDir, wavpath_dict, fname, normalize=True):

    fname = fname.replace('.wav', '')
    ''' read wave '''
    if save_flag['mix']:
        mix = read_wav(wavpath_dict['mix'], fs) # mix
        length, _ = mix.shape
    if save_flag['clean']:
        ref_pad = [[] for _ in range(num_spks)]
        ref1 = read_wav(wavpath_dict['ref_1'], fs)
        ref2 = read_wav(wavpath_dict['ref_2'], fs)
        length, _ = ref1.shape
        ref = [ref1, ref2]
    if save_flag['early']:
        early_pad = [[] for _ in range(num_spks)]
        early1 = read_wav(wavpath_dict['early_1'], fs)
        early2 = read_wav(wavpath_dict['early_2'], fs)
        length, _ = early1.shape
        early = [early1, early2]
    if save_flag['tail']:
        tail_pad = [[]for _ in range(num_spks)]
        tail1 = read_wav(wavpath_dict['tail_1'], fs)
        tail2 = read_wav(wavpath_dict['tail_2'], fs)
        length, _ = tail1.shape
        tail = [tail1, tail2]
    if save_flag['noise']:
        noise = read_wav(wavpath_dict['noise'], fs)
        length, _ = noise.shape
    if save_flag['MISO1']:
        MISO1_pad = [[] for _ in range(num_spks)]
        MISO1_1 = read_wav(wavpath_dict['MISO1_1'],fs)
        MISO1_2 = read_wav(wavpath_dict['MISO1_2'],fs)
        length, _ = MISO1_1.shape
        MISO1 = [MISO1_1, MISO1_2]
    if save_flag['Beamforming']:
        Beamforming_pad = [[] for _ in range(num_spks)]
        # 1 ch일때 shape 확인 -> 최종 [T,1]로 나와야 됨.
        Beamforming_1 = read_wav(wavpath_dict['Beamforming_1'],fs)
        Beamforming_2 = read_wav(wavpath_dict['Beamforming_2'],fs)

        if len(Beamforming_1.shape) == 1:
            Beamforming_1 = np.expand_dims(Beamforming_1, axis=1)
        if len(Beamforming_2.shape) == 1:
            Beamforming_2 = np.expand_dims(Beamforming_2, axis=1)
 
        length, _ = Beamforming_1.shape
        Beamforming = [Beamforming_1, Beamforming_2]

    ''' chunk wave '''
    split_idx = 0
    chunk_size = int(chunk_time * fs)
    least_size = int(least_time * fs)



    if length < least_size:
        return
    if length > least_size and length < chunk_size:        
        start = 0
        gap = chunk_size-length
        if save_flag['mix'] and save_flag['clean']:
            split_cleanmix = {}
            mix_pad = np.pad(mix, ((0,gap),(0,0)), constant_values= 0)
            split_cleanmix['mix'] = mix_pad
            for spk_idx in range(num_spks):
                ref_pad[spk_idx] = np.pad(ref[spk_idx], ((0,gap),(0,0)), constant_values=0)
                assert ref_pad[spk_idx].shape[0] == chunk_size, ('ref{} length is not equall the chunk size'.format(spk_idx))
                split_cleanmix['ref'+str(spk_idx+1)] = ref_pad[spk_idx]

            with open(os.path.join(CleanMixsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                pickle.dump(split_cleanmix,f) 
        if save_flag['early']:
            split_early = {}
            for spk_idx in range(num_spks):
                early_pad[spk_idx] = np.pad(early[spk_idx], (0,gap), constant_values=0)
                assert early_pad[spk_idx].shape[0] == chunk_size, ('early{} length is not equall the chunk size'.format(spk_idx))
                split_early['early'+str(spk_idx+1)] = early_pad[spk_idx]

            with open(os.path.join(earlysaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                pickle.dump(split_early,f)
        if save_flag['tail']:
            split_tail = {}
            for spk_idx in range(num_spks):             
                tail_pad[spk_idx] = np.pad(tail[spk_idx], (0,gap), constant_values=0)
                assert tail_pad[spk_idx].shape[0] == chunk_size, ('tail{} length is not equall the chunk size'.format(spk_idx))
                split_tail['tail'+str(spk_idx+1)] = tail_pad[spk_idx]

            with open(os.path.join(tailsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                pickle.dump(split_tail,f)   
        if save_flag['noise']:
            split_noise = {}
            noise_pad = np.pad(noise, ((0, gap), (0,0)), constant_values=0) 
            assert noise_pad.shape[0] == chunk_size, ('noise length is not equall the chunk size')
            split_noise['noise'] = noise_pad
            
            with open(os.path.join(noisesaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                pickle.dump(split_noise,f)
        if save_flag['MISO1']:
            split_MISO1 = {}
            for spk_idx in range(num_spks):
                MISO1_pad[spk_idx] = np.pad(MISO1[spk_idx], ((0,gap), (0,0)),constant_values= 0)
                assert MISO1_pad[spk_idx].shape[0] == chunk_size, ('MISO1_{} length is not equall the chunk size'.format(spk_idx))
                split_MISO1['MISO1_'+str(spk_idx+1)] = MISO1_pad[spk_idx]
                
            with open(os.path.join(MISO1saveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                pickle.dump(split_MISO1,f)
        if save_flag['Beamforming']:
            split_Beamforming = {}
            for spk_idx in range(num_spks):
                Beamforming_pad[spk_idx] = np.pad(Beamforming[spk_idx],((0,gap), (0,0)), constant_values=0)
                assert Beamforming_pad[spk_idx].shape[0] == chunk_size, ('Beamforming_{} length is not equall the chunk size'.format(spk_idx))
                split_Beamforming['Beamforming_'+str(spk_idx+1)] = Beamforming_pad[spk_idx]

            with open(os.path.join(BeamformingsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                pickle.dump(split_Beamforming,f)

    if length > chunk_size:
        start = 0
        if save_flag['clean']:
            ref_split = [[] for _ in range(num_spks)]
        if save_flag['early']:
            early_split = [[] for _ in range(num_spks)]
        if save_flag['tail']:
            tail_split = [[] for _ in range(num_spks)]
        if save_flag['MISO1']:
            MISO1_split = [[] for _ in range(num_spks)]
        if save_flag['Beamforming']:
            Beamforming_split = [[] for _ in range(num_spks)]

        while True:
            if start+chunk_size > length:
                break
            
            if save_flag['mix'] and save_flag['clean']:
                split_cleanmix = {}
                mix_split = mix[start:start+chunk_size,:]
                split_cleanmix['mix'] = mix_split
                for spk_idx in range(num_spks):
                    ref_split[spk_idx] = ref[spk_idx][start:start+chunk_size,:]
                    assert ref_split[spk_idx].shape[0] == chunk_size, ('ref{} length is not equall the chunk size'.format(spk_idx))
                    split_cleanmix['ref'+str(spk_idx+1)]= ref_split[spk_idx]
                with open(os.path.join(CleanMixsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                    pickle.dump(split_cleanmix,f)    

            if save_flag['early']:
                split_early = {}
                for spk_idx in range(num_spks):
                    early_split[spk_idx] = early[spk_idx][start:start+chunk_size,:]
                    assert early_split[spk_idx].shape[0] == chunk_size, ('early{} length is not equall the chunk size'.format(spk_idx))
                    split_early['early'+str(spk_idx+1)]= early_split[spk_idx]
                with open(os.path.join(earlysaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                    pickle.dump(split_early,f)   

            if save_flag['tail']:
                split_tail = {}
                for spk_idx in range(num_spks):
                    tail_split[spk_idx] = tail[spk_idx][start:start+chunk_size,:]
                    assert tail_pad[spk_idx].shape[0] == chunk_size, ('tail{} length is not equall the chunk size'.format(spk_idx))
                    split_tail['early'+str(spk_idx+1)]= tail_split[spk_idx]
                with open(os.path.join(tailsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                    pickle.dump(split_tail,f)
                    
            if save_flag['noise']:
                split_noise = {}
                noise_split = noise[start:start+chunk_size,:]
                assert noise_split.shape[0] == chunk_size, ('noise length is not equall the chunk size')
                split_noise['noise'] = noise_split
                with open(os.path.join(noisesaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                    pickle.dump(split_noise,f)
                    
            if save_flag['MISO1']:
                split_MISO1 = {}
                for spk_idx in range(num_spks):
                    MISO1_split[spk_idx] = MISO1[spk_idx][start:start+chunk_size,:]
                    assert MISO1_split[spk_idx].shape[0] == chunk_size, ('MISO1_{} length is not equall the chunk size'.format(spk_idx))
                    split_MISO1['MISO1_'+str(spk_idx+1)]= MISO1_split[spk_idx]
                with open(os.path.join(MISO1saveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                    pickle.dump(split_MISO1,f)  

            if save_flag['Beamforming']:
                split_Beamforming = {}
                for spk_idx in range(num_spks):
                    Beamforming_split[spk_idx] = Beamforming[spk_idx][start:start+chunk_size,:]
                    assert Beamforming_split[spk_idx].shape[0] == chunk_size, ('Beamforming_{} length is not equall the chunk size'.format(spk_idx))
                    split_Beamforming['Beamforming_'+str(spk_idx+1)]= Beamforming_split[spk_idx]
                with open(os.path.join(BeamformingsaveDir,fname + '_'+ str(split_idx)+'.pickle'), 'wb') as f:
                    pickle.dump(split_Beamforming,f)  

            
            # verify chunk audio signal
            # test=  mix_split * MAX_INT16
            # test = mix_split * 2**(nbits-1)
            # test = test.astype(np.int16)
            # wf.write('sample.wav',self.fs,test)
            # overlap 2s

            start += least_size
            split_idx += 1
            

class main_smswsj:
    def __init__(self, save_flag,  num_spks, num_ch, chunk_time, least_time, fs, rootDir,saveRootDir, cleanDir, mixDir, earlyDir, tailDir, noiseDir, MISO1Dir, BeamformingDir, trFile, devFile, testFile):
        # mode = {'train' : trFile,'dev' : devFile, 'test':testFile}
        mode = {'train' : trFile,'dev' : devFile}
        # mode = {'dev' : devFile, 'test':testFile}
        self.save_flag = save_flag
        for mode_idx in mode:

            self.CleanMixsaveDir = os.path.join(saveRootDir,mode_idx,'CleanMix')
            if self.save_flag['clean']:
                Path(self.CleanMixsaveDir).mkdir(exist_ok = True, parents=True)
            
            self.mixrootDir = os.path.join(rootDir, mixDir, mode[mode_idx])
            self.cleanrootDir = os.path.join(rootDir, cleanDir, mode[mode_idx])
            
            self.earlyrootDir = os.path.join(rootDir, earlyDir, mode[mode_idx])
            self.earlysaveDir = os.path.join(saveRootDir,mode_idx,earlyDir) 
            if self.save_flag['early']:
                Path(self.earlysaveDir).mkdir(exist_ok = True, parents=True)
            
            self.tailrootDir = os.path.join(rootDir, tailDir, mode[mode_idx])
            self.tailsaveDir = os.path.join(saveRootDir,mode_idx,tailDir)
            if self.save_flag['tail']:
                Path(self.tailsaveDir).mkdir(exist_ok = True, parents=True)
            
            self.noiserootDir = os.path.join(rootDir, noiseDir, mode[mode_idx])
            self.noisesaveDir = os.path.join(saveRootDir,mode_idx,noiseDir)
            if self.save_flag['noise']:
                Path(self.noisesaveDir).mkdir(exist_ok = True, parents=True)

            self.MISO1rootDir = os.path.join(rootDir, MISO1Dir, mode[mode_idx])
            self.MISO1saveDir = os.path.join(saveRootDir,mode_idx,MISO1Dir)
            if self.save_flag['MISO1']:
                Path(self.MISO1saveDir).mkdir(exist_ok=True, parents=True)

            self.BeamformingrootDir = os.path.join(rootDir, BeamformingDir, mode[mode_idx])
            self.BeamformingsaveDir = os.path.join(saveRootDir, mode_idx, BeamformingDir)
            if self.save_flag['Beamforming']:
                Path(self.BeamformingsaveDir).mkdir(exist_ok=True, parents=True)
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
        wavpath_dict = {}
        for spk_idx in range(2):  # SMS_WSJ dataset has two spk
            ref_name = file_.replace('.wav','_{}.wav'.format(spk_idx))
            ref_path = os.path.join(self.cleanrootDir,ref_name)
            wavpath_dict['ref_'+str(spk_idx+1)] = Path(str(ref_path))

            early_path = os.path.join(self.earlyrootDir,ref_name)
            wavpath_dict['early_'+str(spk_idx+1)] = Path(str(early_path))
            
            tail_path = os.path.join(self.tailrootDir, ref_name)
            wavpath_dict['tail_' + str(spk_idx+1)] = Path(str(tail_path))

            MISO1_path = os.path.join(self.MISO1rootDir, ref_name)
            wavpath_dict['MISO1_'+str(spk_idx+1)] = Path(str(MISO1_path))

            Beamforming_path = os.path.join(self.BeamformingrootDir, ref_name)
            wavpath_dict['Beamforming_'+str(spk_idx+1)] = Path(str(Beamforming_path))

        mix_path = os.path.join(self.mixrootDir, file_)
        wavpath_dict['mix'] = mix_path
        
        noise_path = os.path.join(self.noiserootDir, file_)
        wavpath_dict['noise'] = noise_path

        chunkSplit(self.save_flag, self.num_spks,self.num_ch, self.chunk_time,self.least_time,\
                self.fs, self.CleanMixsaveDir, self.earlysaveDir, self.tailsaveDir, self.noisesaveDir,\
                self.MISO1saveDir, self.BeamformingsaveDir, wavpath_dict, file_ ,normalize=True)

        
        



        



        # noise wav가 clean, mix 신호와 길이가 같은지 확인하기

