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
from itertools import permutations
from numpy.linalg import solve
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

class AudioDataset(data.Dataset):
    
    def __init__(self,trainMode, functionMode, num_spks, num_ch, pickle_dir,model,device,cudaUse,check_audio,**STFT_args):
        super(AudioDataset, self).__init__()
        self.trainMode = trainMode
        self.functionMode = functionMode
        self.model = model
        self.fs = STFT_args['fs']
        self.window = STFT_args['window']
        self.nperseg = STFT_args['length']
        self.noverlap = STFT_args['overlap']
        self.num_spks = num_spks
        self.num_ch = num_ch
        self.device = device
        self.cudaUse = cudaUse
        self.pickle_dir = list(Path(pickle_dir).glob('**/**/**/**/*.pickle'))
        hann_win = scipy.signal.get_window('hann', self.nperseg)
        self.scale = np.sqrt(1.0 / hann_win.sum()**2)
        self.check_audio = check_audio
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

                    
        if self.functionMode == 'Separate':
            """
                Output :
                        mix_stft : [Mic,T,F]
                        ref_stft : [Mic,T,F]
            """
            return mix_stft, ref_stft

        elif self.functionMode == 'Beamforming':
            """
            Output :
                    mix_stft : [Mic,T,F]
                    ref_stft : [Mic,T,F]
            """
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

            BeamOutSaveDir = str(self.pickle_dir[index]).replace('CleanMix','Beamforming')
            MISO1OutSaveDir = str(self.pickle_dir[index]).replace('CleanMix','MISO1')

            return mix_stft, ref_stft, BeamOutSaveDir, MISO1OutSaveDir

        elif 'Enhance' in self.functionMode:
            """
            Output :
                    mix_stft : [Mic,T,F]
                    ref_stft_1ch, list, [Mic,T,F]
                    MISO1_stft, list, [Mic,T,F]
                    Beamform_stft, list, [Mic,T,F]
            """
                
            if len(mix_stft.shape)==3:
                mix_stft = torch.unsqueeze(mix_stft,dim=0)
            if self.cudaUse:
                mix_stft = mix_stft.cuda(self.device)
            ref_stft_1ch = [[] for _ in range(self.num_spks)]
            for spk_idx in range(self.num_spks):
                if len(ref_stft[spk_idx].shape) == 3:
                    ref_stft[spk_idx] = torch.unsqueeze(ref_stft[spk_idx], dim=0)
                ref_stft_1ch[spk_idx] = ref_stft[spk_idx][:,0,:,:]            # select reference mic channel
                ref_stft_1ch[spk_idx] = torch.unsqueeze(ref_stft_1ch[spk_idx], dim=1)
            
            B, Mic, T, F = mix_stft.size()
            
            if self.functionMode == 'Enhance_Load_MISO1_Output' or self.functionMode == 'Enhance_Load_MISO1_MVDR_Output':

                MISO1OutSaveDir = str(self.pickle_dir[index]).replace('CleanMix','MISO1')
                MISO1_stft = [[] for _ in range(self.num_spks)]
                spk_name = {0:'_s1.wav', 1:'_s2.wav'}
                # Load MISO1 Output
                for spk_idx in range(self.num_spks):
                    MISO1_sig, fs = librosa.load(MISO1OutSaveDir.replace('.pickle',spk_name[spk_idx]), mono= False, sr= 8000)
                    if MISO1_sig.shape[1] != self.num_ch:
                        MISO1_sig = MISO1_sig.T
                    assert fs == self.fs, 'Check sampling rate'
                    if len(MISO1_sig.shape) == 1:
                        MISO1_sig = np.expand_dims(MISO1_sig, axis=1)
                    MISO1_stft[spk_idx] = torch.permute(torch.from_numpy(self.STFT(MISO1_sig)),[0,2,1])
                    MISO1_stft[spk_idx] = MISO1_stft[spk_idx]/self.scale
                
                MISO1_spk1 = torch.unsqueeze(MISO1_stft[0],dim=0)
                MISO1_spk2 = torch.unsqueeze(MISO1_stft[1],dim=0)

            else:

                """
                MISO1 Inference
                Input : mix_stft, size : [Mic,T,F]
                Output : MISO1_stft, list, size : [reference Mic channel,T,F]
                """

                MISO1_spk1 = torch.empty(B, Mic,T,F, dtype=torch.complex64)
                MISO1_spk2 = torch.empty(B, Mic,T,F, dtype=torch.complex64)

                with torch.no_grad():
                    pdb.set_trace()
                    MISO1_ch0 = self.model(mix_stft)
                                
                MISO1_spk1[:,0,:,:] = MISO1_ch0[:,0,:,:]
                MISO1_spk2[:,0,:,:] = MISO1_ch0[:,1,:,:]

                # circular shift the microphones at run time for the prediction of each microphone signal
                # If the microphones are arranged uniformly on a circle
                # Select the reference microphone by circular shifting the microphone
                # [Yq, ... , Yp, ... , Yq-1]

                s_MISO1_ch0 = torch.unsqueeze(MISO1_ch0, dim=2)
                s_Mag_MISO1_ch0 = torch.abs(torch.sqrt(s_MISO1_ch0.real**2 + s_MISO1_ch0.imag**2)) #[B,Spks,1,T,F]
                with torch.no_grad():
                    for ref_micIdx in range(1,Mic):
                        mix_stft_chShift = torch.roll(mix_stft,-ref_micIdx, dims=1)
                        MISO1_chShift = self.model(mix_stft_chShift)

                        # Permutation Invariant Alignmnet
                        # calculate magnitude distance between ref mic and non ref mic
                        s_MISO1_chShift = torch.unsqueeze(MISO1_chShift, dim=1) #[B,1,Spks,T,F]
                        s_Mag_MISO1_chShift = torch.sum(torch.abs(s_Mag_MISO1_ch0 - abs(s_MISO1_chShift)),[3,4]) #[B,Spks,Spks,T,F]
                        perms = MISO1_chShift.new_tensor(list(permutations(range(self.num_spks))), dtype=torch.long)
                        index_ = torch.unsqueeze(perms, dim=2)
                        perms_one_hot = MISO1_chShift.new_zeros((*perms.size(), self.num_spks), dtype=torch.float).scatter_(2,index_,1)
                        batchwise_distance = torch.einsum('bij,pij->bp', [s_Mag_MISO1_chShift, perms_one_hot])
                        min_distance_idx = torch.argmin(batchwise_distance,dim=1)

                        for batch_idx in range(B):
                            if min_distance_idx[batch_idx] == 1:
                                MISO1_spk1[batch_idx,ref_micIdx,:,:] = MISO1_chShift[batch_idx,1,:,:]
                                MISO1_spk2[batch_idx,ref_micIdx,:,:] = MISO1_chShift[batch_idx,0,:,:]
                            else:
                                MISO1_spk1[batch_idx,ref_micIdx,:,:] = MISO1_chShift[batch_idx,0,:,:]
                                MISO1_spk2[batch_idx,ref_micIdx,:,:] = MISO1_chShift[batch_idx,1,:,:]     

                MISO1_stft = [[] for _ in range(self.num_spks)]   # select reference mic channel
                MISO1_stft[0] = MISO1_spk1
                MISO1_stft[1] = MISO1_spk2

            """
            Source Alignment between Clean reference signal and MISO1 signal 
            calculate magnitude distance between ref mic(ch0) and target signal(reference mic : ch0) 
            """

            ref1 = torch.empty(B,1,T,F, dtype=torch.complex64)
            ref2 = torch.empty(B,1,T,F, dtype=torch.complex64)

            for spk_idx in range(self.num_spks):
                if spk_idx == 0 :
                    ref_ = ref_stft_1ch[spk_idx]
                else:
                    ref_ = torch.cat((ref_,ref_stft_1ch[spk_idx]), dim=1)
            
            s_MISO1 = torch.unsqueeze(torch.stack((MISO1_spk1[:,0,...], MISO1_spk2[:,0,...]), dim=1) ,dim=2) #[B,Spks,1,T,F]
            magnitude_MISO1 = torch.abs(torch.sqrt(s_MISO1.real**2 + s_MISO1.imag**2)) #[B,Spks,1,T,F]
            
            s_ref = torch.unsqueeze(ref_, dim=1)
            magnitude_distance = torch.sum(torch.abs(magnitude_MISO1 - abs(s_ref)),[3,4])
            perms = ref_.new_tensor(list(permutations(range(self.num_spks))), dtype=torch.long) #[[0,1],[1,0]]
            index_ = torch.unsqueeze(perms, dim=2)
            perms_one_hot = ref_.new_zeros((*perms.size(), self.num_spks), dtype=torch.float).scatter_(2,index_,1)
            batchwise_distance = torch.einsum('bij,pij->bp',[magnitude_distance, perms_one_hot])
            min_distance_idx = torch.argmin(batchwise_distance, dim=1)

            for ii in range(B):
                if min_distance_idx[ii] == 1:
                    ref1[ii,...] = ref_[ii,1,:,:]
                    ref2[ii,...] = ref_[ii,0,:,:]
                else:
                    ref1[ii,...] = ref_[ii,0,:,:]
                    ref2[ii,...] = ref_[ii,1,:,:]


            ref_stft_1ch[0] = ref1
            ref_stft_1ch[1] = ref2
        
            for spk_idx in range(self.num_spks):
                ref_stft_1ch[spk_idx] = torch.squeeze(ref_stft_1ch[spk_idx],dim=0)
            
            """
                Apply MVDR Beamforming
                    Output :
                            Beamform_stft, [reference Mic channel, T, F]
            """
            if self.functionMode == 'Enhance_Load_MVDR_Output' or self.functionMode == 'Enhance_Load_MISO1_MVDR_Output':

                BeamformSaveDir = str(self.pickle_dir[index]).replace('CleanMix','Beamforming')
                Beamform_stft = [[] for _ in range(self.num_spks)]
                spk_name = {0:'_s1.wav', 1:'_s2.wav'}
                # Load MISO1 Output
                for spk_idx in range(self.num_spks):
                    Beamform_sig, fs = librosa.load(BeamformSaveDir.replace('.pickle',spk_name[spk_idx]), mono= False, sr= 8000)
                    if len(Beamform_sig.shape) == 1:
                        Beamform_sig = np.expand_dims(Beamform_sig, axis=1)
                    assert fs == self.fs, 'Check sampling rate'
                    Beamform_stft[spk_idx] = torch.permute(torch.from_numpy(self.STFT(Beamform_sig)),[0,2,1])
                    Beamform_stft[spk_idx] = Beamform_stft[spk_idx]/self.scale
            
            else:
                s1_ch6 = torch.permute(MISO1_spk1,[0,3,1,2]) # [B, F, Ch, T]
                s1_ch6 = s1_ch6.detach().cpu().numpy()
                s2_ch6 = torch.permute(MISO1_spk2,[0,3,1,2])
                s2_ch6 = s2_ch6.detach().cpu().numpy()
                numpy_mix_stft = torch.permute(mix_stft,[0,3,1,2])
                numpy_mix_stft = numpy_mix_stft.detach().cpu().numpy()

                # Apply small Diagonal matrix to prevent matrix inversion error
                eye = np.eye(Mic)
                eye = eye.reshape(1,1,Mic,Mic)
                delta = 1e-6* np.tile(eye,[B,F,1,1])

                ''' Source 1 '''
                s1_SCMs = self.get_spatial_covariance_matrix(s1_ch6,normalize=True) # target covariance matrix, size : [B,F,C,C]
                s1_SCMs = 0.5 * (s1_SCMs + np.conj(s1_SCMs.swapaxes(-1,-2))) # verify hermitian symmetric  
                
                ''' Noise Spatial Covariance ''' 
                s1_noise_signal = numpy_mix_stft - s1_ch6
                # s1_noise_signal = mix_stft  #MPDR
                s1_SCMn = self.get_spatial_covariance_matrix(s1_noise_signal,normalize = True) # noise covariance matrix, size : [B,F,C,C]
                # s1_SCMn = self.condition_covariance(s1_SCMn, 1e-6)
                # s1_SCMn /= np.trace(s1_SCMn, axis1=-2, axis2= -1)[...,None, None]
                s1_SCMn = 0.5 * (s1_SCMn + np.conj(s1_SCMn.swapaxes(-1,-2))) # verify hermitian symmetric

                ''' Get Steering vector : Eigen-decomposition '''
                shape = s1_SCMs.shape
                s1_steering = np.empty(shape[:-1], dtype=np.complex)

                # s1_SCMs += delta
                s1_SCMs = np.reshape(s1_SCMs, (-1,) + shape[-2:]) 
                eigenvals, eigenvecs = np.linalg.eigh(s1_SCMs)
                # Find max eigenvals
                vals = np.argmax(eigenvals, axis=-1)
                # Select eigenvec for max eigenval
                s1_steering = np.array([eigenvecs[i,:,vals[i]] for i in range(eigenvals.shape[0])])
                # s1_steering = np.array([eigenvecs[i,:,vals[i]] * np.sqrt(Mic/np.linalg.norm(eigenvecs[i,:,vals[i]])) for i in range(eigenvals.shape[0])]) # [B*F,Ch,Ch]
                s1_steering = np.reshape(s1_steering, shape[:-1]) # [B,F,Ch]
                s1_SCMs = np.reshape(s1_SCMs, shape)
                
                ''' steering normalize with respect to the reference microphone '''
                # ver 1 
                s1_steering = s1_steering / np.expand_dims(s1_steering[:,:,0], axis=2)
                for b_idx in range(0,B):
                    for f_idx in range(0,F):
                        # s1_steering[b_idx,f_idx,:] = s1_steering[b_idx,f_idx,:] / s1_steering[b_idx,f_idx,0]
                        s1_steering[b_idx,f_idx,:] = s1_steering[b_idx,f_idx,:] * np.sqrt(Mic/(np.linalg.norm(s1_steering[b_idx,f_idx,:])))
                
                # ver 2
                # s1_steering = self.normalize(s1_steering)

                s1_steering = self.PhaseCorrection(s1_steering)
                s1_beamformer = self.get_mvdr_beamformer(s1_steering, s1_SCMn, delta)
                # s1_beamformer = self.blind_analytic_normalization(s1_beamformer,s1_SCMn)
                s1_bf = self.apply_beamformer(s1_beamformer,numpy_mix_stft)
                s1_bf = torch.permute(torch.from_numpy(s1_bf), [0,2,1])
                
                ####################
                ''' source 2 '''
                ####################
                s2_SCMs = self.get_spatial_covariance_matrix(s2_ch6, normalize = True) # target covariance matrix, size : [B,F,C,C]
                s2_SCMs = 0.5 * (s2_SCMs + np.conj(s2_SCMs.swapaxes(-1,-2))) # verify hermitian symmetric  

                ''' Noise Spatial Covariance '''
                s2_noise_signal = numpy_mix_stft - s2_ch6
                # s2_noise_signal = mix_stft #MPDR
                s2_SCMn = self.get_spatial_covariance_matrix(s2_noise_signal, normalize = True) # noise covariance matrix, size : [B,F,C,C]
                # s2_SCMn = self.condition_covariance(s2_SCMn, 1e-6)
                # s2_SCMn /= np.trace(s2_SCMn, axis1=-2, axis2= -1)[...,None, None]
                s2_SCMn = 0.5 * (s2_SCMn + np.conj(s2_SCMn.swapaxes(-1,-2))) # verify hermitian symmetric

                ########  Get Steering vector : Eigen-decomposition ########
                shape = s2_SCMs.shape
                s2_steering = np.empty(shape[:-1], dtype=np.complex)

                # s2_SCMs += delta
                s2_SCMs = np.reshape(s2_SCMs, (-1,) + shape[-2:])
                eigenvals, eigenvecs = np.linalg.eigh(s2_SCMs) # eigenvals size : [B*F,Ch]
                # Find max eigenvals
                vals = np.argmax(eigenvals, axis=-1)
                # Select eigenvec for max eigenval
                s2_steering = np.array([eigenvecs[i,:,vals[i]] for i in range(eigenvals.shape[0])])
                # s2_steering = np.array([eigenvecs[i,:,vals[i]] * np.sqrt(Mic/np.linalg.norm(eigenvecs[i,:,vals[i]])) for i in range(eigenvals.shape[0])]) # [B*F,Ch,Ch]
                s2_steering = np.reshape(s2_steering, shape[:-1])
                s2_SCMs = np.reshape(s2_SCMs, shape)

                ''' steering normalize with respect to the reference microphone '''
                # ver 1
                s2_steering = s2_steering / np.expand_dims(s2_steering[:,:,0], axis=2)
                for b_idx in range(0,B):
                    for f_idx in range(0,F):
                        # s2_steering[b_idx,f_idx,:] = s2_steering[b_idx,f_idx,:] / s2_steering[b_idx,f_idx,0]
                        s2_steering[b_idx,f_idx,:] = s2_steering[b_idx,f_idx,:] * np.sqrt(Mic/(np.linalg.norm(s2_steering[b_idx,f_idx,:])))
                # ver 2
                # s2_steering = self.normalize(s2_steering)

                s2_steering = self.PhaseCorrection(s2_steering)
                s2_beamformer = self.get_mvdr_beamformer(s2_steering, s2_SCMn, delta)
                # s2_beamformer = self.blind_analytic_normalization(s2_beamformer,s2_SCMn)
                s2_bf = self.apply_beamformer(s2_beamformer,numpy_mix_stft)
                s2_bf = torch.permute(torch.from_numpy(s2_bf), [0,2,1])

                Beamform_stft = [[] for _ in range(self.num_spks)]


                Beamform_stft[0] = s1_bf
                Beamform_stft[1] = s2_bf
                
                
            if len(mix_stft.shape)== 4:
                mix_stft = torch.squeeze(mix_stft)
                mix_stft = mix_stft.detach().cpu()
            for spk_idx in range(self.num_spks):
                if len(MISO1_stft[spk_idx].shape)== 4:
                    MISO1_stft[spk_idx] = torch.squeeze(MISO1_stft[spk_idx])

            if self.check_audio:
                
                ''' Check the result of MISO1 '''
                self.save_audio(np.transpose(mix_stft, [0,2,1]), 'mix')
                for spk_idx in range(self.num_spks):
                    self.save_audio(np.transpose(ref_stft_1ch[spk_idx], [0,2,1]), 'ref_s{}'.format(spk_idx))
                    self.save_audio(np.transpose(MISO1_stft[spk_idx], [0,2,1]), 'MISO1_s{}'.format(spk_idx))
                    self.save_audio(np.transpose(Beamform_stft[spk_idx], [0,2,1]), 'Beamform_s{}'.format(spk_idx))

                pdb.set_trace()
                
            return mix_stft, ref_stft_1ch, MISO1_stft, Beamform_stft
            
        else:
            assert -1, '[Error] Choose correct train mode'        

    def save_audio(self,signal, wavname):
        '''
            Input:
                    signal : [Ch,F,T]
                    wavename : str, wav name to save
        '''

        hann_win = scipy.signal.get_window(self.window, self.nperseg)
        scale = np.sqrt(1.0 / hann_win.sum()**2)
        MAX_INT16 = np.iinfo(np.int16).max


        signal = signal * scale
        t_sig = self.ISTFT(signal)
        t_sig=  t_sig * MAX_INT16
        t_sig = t_sig.astype(np.int16)
        sf.write('{}.wav'.format(wavname),t_sig.T, self.fs,'PCM_24')

    def ISTFT(self,FT_sig): 

        '''
        input : [F,T]
        output : [T,C]
        '''
        # if FT_sig.shape[1] != self.config['ISTFT']['length']+1:
            # FT_sig = np.transpose(FT_sig,(0,1)) # [C,T,F] -> [C,F,T]

        _, t_sig = signal.istft(FT_sig,fs=self.fs, window=self.window, nperseg=self.nperseg, noverlap=self.noverlap) #[C,F,T] -> [T,C]

        return t_sig

    def get_spatial_covariance_matrix(self,observation,normalize):
        '''
        Input : 
            observation : complex 
                            size : [B,F,C,T]
        Return :
                R       : double
                            size : [B,F,C,C]
        '''
        B,F,C,T = observation.shape
        R = np.einsum('...dt,...et-> ...de', observation, observation.conj())
        if normalize:
            normalization = np.sum(np.ones((B,F,1,T)),axis=-1, keepdims=True)
            R /= normalization
        return R
    
    def PhaseCorrection(self,W): #Matlab과 동일
        """
        Phase correction to reduce distortions due to phase inconsistencies.
        Input:
                W : steering vector
                    size : [B,F,Ch]
        """
        w = W.copy()
        B, F, Ch = w.shape
        for b_idx in range(0,B):
            for f in range(1, F):
                w[b_idx,f, :] *= np.exp(-1j*np.angle(
                    np.sum(w[b_idx,f, :] * w[b_idx,f-1, :].conj(), axis=-1, keepdims=True)))
        return w
    
    def condition_covariance(self,x,gamma):
        """see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
        B,F,_,_ = x.shape
        for b_idx in range(0,B):
            scale = gamma * np.trace(x[b_idx,...]) / x[b_idx,...].shape[-1]
            scaled_eye = np.eye(x.shape[-1]) * scale
            x[b_idx,...] = (x[b_idx,...]+scaled_eye) / (1+gamma)
        return x
    
    def normalize(self,vector):
        B,F,Ch = vector.shape
        for b_idx in range(0,B):
            for ii in range(0,F):   
                weight = np.matmul(np.conjugate(vector[b_idx,ii,:]).reshape(1,-1), vector[b_idx,ii,:])
                vector[b_idx,ii,:] = (vector[b_idx,ii,:] / weight) 
        return vector     

    def blind_analytic_normalization(self,vector, noise_psd_matrix, eps=0):
        """Reduces distortions in beamformed ouptput.
            
        :param vector: Beamforming vector
            with shape (..., sensors)
        :param noise_psd_matrix:
            with shape (..., sensors, sensors)
        :return: Scaled Deamforming vector
            with shape (..., sensors)
        """
        nominator = np.einsum(
            '...a,...ab,...bc,...c->...',
            vector.conj(), noise_psd_matrix, noise_psd_matrix, vector
        )
        nominator = np.abs(np.sqrt(nominator))

        denominator = np.einsum(
            '...a,...ab,...b->...', vector.conj(), noise_psd_matrix, vector
        )
        denominator = np.abs(denominator)

        normalization = nominator / (denominator + eps)
        return vector * normalization[..., np.newaxis]


    def get_mvdr_beamformer(self, steering_vector, R_noise, delta):
        """
        Returns the MVDR beamformers vector

        Input :
            steering_vector : Acoustic transfer function vector
                                shape : [B, F, Ch]
                R_noise     : Noise spatial covariance matrix
                                shape : [B, F, Ch, Ch]
        """
        R_noise += delta
        numer = solve(R_noise, steering_vector)
        denom = np.einsum('...d,...d->...', steering_vector.conj(), numer)
        beamformer = numer / np.expand_dims(denom, axis=-1)
        return beamformer

    def apply_beamformer(self, beamformer, mixture):
        return np.einsum('...a,...at->...t',beamformer.conj(), mixture)            


    
    def __len__(self):
        return len(self.pickle_dir)



class AudioDataset_Test(data.Dataset):

    def __init__(self,sms_wsj_observation_dir,speech_source_dir, chunk_time, ref_ch, **STFT_args):
        
        self.wav_dir = list(Path(sms_wsj_observation_dir).glob('**/*.wav'))
        self.speech_source_dir = speech_source_dir

        # STFT parameters
        self.fs = STFT_args['fs']
        self.window = STFT_args['window']
        self.nperseg = STFT_args['length']
        self.noverlap = STFT_args['overlap']
        hann_win = scipy.signal.get_window('hann', self.nperseg)
        self.scale = np.sqrt(1.0 / hann_win.sum()**2)
        self.MAX_INT16 = np.iinfo(np.int16).max

        self.chunck_size = int(chunk_time * self.fs)
        self.ref_ch = ref_ch
        
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

        wav_observe = self.read_wav(self.wav_dir[index], self.fs)
        wav_name = str(self.wav_dir[index]).split('/')[-1].replace('.wav','')
        wav_clean_s0 = self.read_wav(os.path.join(self.speech_source_dir, wav_name+'_0.wav'),self.fs)
        wav_clean_s1 = self.read_wav(os.path.join(self.speech_source_dir, wav_name+'_1.wav'),self.fs)
        
        
        split_idx = 0
        length, _ = wav_observe.shape
        
        split_observe_dict = {} 
        split_clean_s0_dict = {}
        split_clean_s1_dict = {}
        if length < self.chunck_size:
            gap = self.chunck_size - length
            split_observe_wav = np.pad(wav_observe, ((0,gap), (0,0)), constant_values=0)
            split_observe_stft = self.STFT(split_observe_wav)
            split_observe_stft = torch.permute(torch.from_numpy(split_observe_stft/self.scale), [0,2,1])
            split_observe_dict[str(split_idx)] = split_observe_stft

            split_clean_s0_wav = np.pad(wav_clean_s0, ((0,gap), (0,0)), constant_values=0)
            split_clean_s0_stft = self.STFT(split_clean_s0_wav)
            split_clean_s0_stft = torch.permute(torch.from_numpy(split_clean_s0_stft/self.scale), [0,2,1])
            split_clean_s0_dict[str(split_idx)] = split_clean_s0_stft

            split_clean_s1_wav = np.pad(wav_clean_s1, ((0,gap), (0,0)), constant_values=0)
            split_clean_s1_stft = self.STFT(split_clean_s1_wav)
            split_clean_s1_stft = torch.permute(torch.from_numpy(split_clean_s1_stft/self.scale), [0,2,1])
            split_clean_s1_dict[str(split_idx)] = split_clean_s1_stft

        elif length > self.chunck_size:
            start = 0
            while True:
                if start + self.chunck_size > length:
                    gap = self.chunck_size - (length - start)
                    split_observe_wav = np.pad(wav_observe[start:,:], ((0,gap), (0,0)), constant_values=0)
                    split_clean_s0_wav = np.pad(wav_clean_s0[start:,:], ((0,gap), (0,0)), constant_values=0)
                    split_clean_s1_wav = np.pad(wav_clean_s1[start:,:], ((0,gap), (0,0)), constant_values=0)

                else:
                    split_observe_wav = wav_observe[start:start+self.chunck_size,:]
                    split_clean_s0_wav = wav_clean_s0[start:start+self.chunck_size,:]
                    split_clean_s1_wav = wav_clean_s1[start:start+self.chunck_size,:]

                assert split_observe_wav.shape[0] == self.chunck_size, ('observed wav length does not match chunk_size')
                assert split_clean_s0_wav.shape[0] == self.chunck_size, ('clean s0 wav length does not match chunk_size')
                assert split_clean_s1_wav.shape[0] == self.chunck_size, ('clean s1 wav length does not match chunk_size')
                
                split_observe_stft = self.STFT(split_observe_wav)
                split_observe_stft = torch.permute(torch.from_numpy(split_observe_stft/self.scale),[0,2,1])
                split_observe_dict[str(split_idx)] = split_observe_stft

                split_clean_s0_stft = self.STFT(split_clean_s0_wav)
                split_clean_s0_stft = torch.permute(torch.from_numpy(split_clean_s0_stft/self.scale),[0,2,1])
                split_clean_s0_dict[str(split_idx)] = split_clean_s0_stft

                split_clean_s1_stft = self.STFT(split_clean_s1_wav)
                split_clean_s1_stft = torch.permute(torch.from_numpy(split_clean_s1_stft/self.scale),[0,2,1])
                split_clean_s1_dict[str(split_idx)] = split_clean_s1_stft

                if start + self.chunck_size > length:
                    break
                else:
                    start += self.chunck_size
                    split_idx += 1
        return split_observe_dict, split_clean_s0_dict, split_clean_s1_dict, gap, wav_name

    def __len__(self):
        return len(self.wav_dir)

    # nbits = 16

    def read_wav(self,filedir,samplingrate):
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


