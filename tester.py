import torch
import time
from tqdm import tqdm
from criterion import loss_uPIT
from torch.utils.tensorboard import SummaryWriter
import pdb
from pathlib import Path
import os
from scipy import signal
import numpy as np
import scipy.io.wavfile as wf
from scipy.io import savemat

#import criterion import cal_loss


class Tester(object):
    def __init__(self,loader,model,config,device):
        self.loader = loader
        self.config = config
        self.model = model
        self.device = device
        self._load()

    def _load(self):
        if self.config['tester']['model_load'][0]:
            print("Loading checkpoint model %s" % self.config['tester']['model_load'][1])
            package = torch.load(self.config['tester']['model_load'][1],map_location="cuda:"+str(self.device))
            self.model.load_state_dict(package['model_state_dict'])

    def test(self):
        print('Testing...')
        self.model.eval()
        self._run_one_epoch()

    def _run_one_epoch(self):
        for idx, (data) in enumerate(self.loader):
            """
            Input : [B,Mic,T,F]
            """
            mix_stft, ref1_stft, ref2_stft = data
            mix_stft = mix_stft.cuda(self.device) #[1,8,168,257]
            ref1_stft = ref1_stft #[1,8,168,257]
            ref2_stft = ref2_stft #[1,8,168,257]
            B, Mic, T, F = mix_stft.size()
            # for ref_mic in range(Mic):
            """
            Select the reference microphone by circular shifting the microphone
            [Yq,...,YP,Y1,...,Yq-1]
            """
            # mix_stft = torch.roll(mix_stft,-ref_mic,dims=1)
            estimate_sources = self.model(mix_stft) #[B,Spk,T,F]

            for ref_mic in range(Mic):
                """
                Select the reference microphone by circular shifting the microphone
                [Yq,...,YP,Y1,...,Yq-1]
                """
                mix_stft = torch.roll(mix_stft,-ref_mic,dims=1)
                estimate_sources = self.model(mix_stft) #[B,Spk,T,F]
                pdb.set_trace()
                mix_t_1ch = self.ISTFT(mix_stft[:,ref_mic,:,:].cpu().detach().numpy(),'_mix_mic'+str(ref_mic)) 
                ref1_t_1ch = self.ISTFT(ref1_stft[:,ref_mic,:,:].cpu().detach().numpy(),'_ref1_mic'+str(ref_mic))
                ref2_t_1ch = self.ISTFT(ref2_stft[:,ref_mic,:,:].cpu().detach().numpy(),'_ref2_mic'+str(ref_mic))
                e_s1 = estimate_sources[:,0,:,:].cpu().detach().numpy()
                e_s2 = estimate_sources[:,1,:,:].cpu().detach().numpy()
                e_s1_t = self.ISTFT(e_s1,'_e1_mic'+str(ref_mic))
                e_s2_t = self.ISTFT(e_s2,'_e2_mic'+str(ref_mic))
            # pdb.set_trace()

            # mdic = {'mix' : mix_stft[:,0,:,:].cpu().detach().numpy(), 'ref1': ref1_stft[:,0,:,:].cpu().detach().numpy(),
            #              'ref2': ref2_stft[:,0,:,:].cpu().detach().numpy(), 'e_s1' : estimate_sources[:,0,:,:].cpu().detach().numpy(),
            #              'e_s2': estimate_sources[:,1,:,:].cpu().detach().numpy()}
            # savemat("test.mat", mdic)
                

            #Source Alignment across Microphones module should be implemented
            # when beamforming is performed after training.

    def ISTFT(self,FT_sig,index): 
        '''
        input : [C,F,T]
        output : [T,C]
        '''
        if FT_sig.shape[1] != self.config['ISTFT']['length']+1:
            FT_sig = np.transpose(FT_sig,(0,2,1)) # [C,T,F] -> [C,F,T]
        fs = self.config['ISTFT']['fs']; window = self.config['ISTFT']['window']; nperseg=self.config['ISTFT']['length']; noverlap=self.config['ISTFT']['overlap']
        _, t_sig = signal.istft(FT_sig,fs=fs, window=window, nperseg=nperseg, noverlap=noverlap) #[C,F,T] -> [T,C]
        

        MAX_INT16 = np.iinfo(np.int16).max
        t_sig=  t_sig * (MAX_INT16-1)
        t_sig = t_sig.astype(np.int16)
        # t_sig = t_sig/np.max(abs(t_sig))
        wf.write('sample'+index+'.wav',self.config['ISTFT']['fs'],t_sig.T)

        return t_sig





