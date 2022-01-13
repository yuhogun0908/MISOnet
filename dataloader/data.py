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
    
    def __init__(self,trainMode, functionMode, num_spks, num_ch, num_ch_utilize, pickle_dir,ref_ch, MISO1_pickle_Rootdir, Beamforming_pickle_Rootdir, model,device,cudaUse,check_audio,**STFT_args):
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
        self.num_ch_utilize = num_ch_utilize
        self.device = device
        self.cudaUse = cudaUse
        self.pickle_dir = list(Path(pickle_dir).glob('**/**/**/**/*.pickle'))
        self.MISO1_pickle_Rootdir = MISO1_pickle_Rootdir
        self.Beamforming_pickle_Rootdir = Beamforming_pickle_Rootdir
        self.ref_ch = ref_ch
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

        wav_name = str(self.pickle_dir[index]).split('/')[-1]
        with open(self.pickle_dir[index], 'rb') as f:
            data_infos = pickle.load(f)
        f.close()

        mix = data_infos['mix']
        mix_stft = self.STFT(mix)
        mix_stft = mix_stft/self.scale # scale equality between scipy stft and matlab stft
        mix_stft = torch.permute( torch.from_numpy(mix_stft),[0,2,1]) # numpy to torch & reshape [M,F,T] ->[M,T,F]
        M, T, F = mix_stft.shape
        mix_stft = mix_stft[0:M:M//self.num_ch_utilize,...]

        
        assert self.num_spks+1 == len(data_infos), "[ERROR] Check the number of speakers"
        ref_stft = [[] for spk_idx in range(self.num_spks)]
        for spk_idx in range(self.num_spks):
            ref_sig = data_infos['ref'+str(spk_idx+1)]
            if len(ref_sig.shape) == 1:
                ref_sig = np.expand_dims(ref_sig,axis=1)
            ref_temp = torch.permute(torch.from_numpy(self.STFT(ref_sig)),[0,2,1]) # numpy to torch & reshape [M,F,T] -> [M,T,F]
            ref_temp = ref_temp/self.scale # scale equality between scipy stft and matlab stft
            ref_stft[spk_idx] = ref_temp[0:M:M//self.num_ch_utilize,...]
    
                    
        if self.functionMode == 'Separate':
            """
                Output :
                        mix_stft : [Mic,T,F]
                        ref_stft : [Mic,T,F]
            """
            return mix_stft ,ref_stft

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
                    ref_stft[spk_idx] = torch.unsqueeze(ref_stft[spk_idx], dim=0) #[1,M,T,F]
                ref_stft_1ch[spk_idx] = ref_stft[spk_idx][:,self.ref_ch,:,:]            # select reference mic channel
                ref_stft_1ch[spk_idx] = torch.unsqueeze(ref_stft_1ch[spk_idx], dim=1) #[1,1,T,F]
            
            B, Mic, T, F = mix_stft.size()
            

            """
                MISO1 Inference
                Input : mix_stft, size : [Mic,T,F]
                Output : MISO1_stft, list, size : [reference Mic channel,T,F]
            """

            if self.functionMode == 'Enhance_Load_MISO1_Output' or self.functionMode == 'Enhance_Load_MISO1_MVDR_Output':
                

                with open(os.path.join(self.MISO1_pickle_Rootdir, wav_name), 'rb') as f:
                    MISO1_infos = pickle.load(f)
                f.close()

                MISO1_stft = [[] for _ in range(self.num_spks)]
                for spk_idx in range(self.num_spks):                
                    MISO1_stft[spk_idx] = torch.permute(torch.from_numpy(self.STFT(MISO1_infos['MISO1_{}'.format(spk_idx+1)])),[0,2,1])
                    MISO1_stft[spk_idx] = MISO1_stft[spk_idx]/self.scale
                    if len(MISO1_stft[spk_idx].shape) == 3:
                        MISO1_stft[spk_idx] = torch.unsqueeze(MISO1_stft[spk_idx], dim=0) # B,C,T,F

            else:
                MISO1_stft = self.MISO1_Inference(mix_stft, ref_ch=self.ref_ch)
                if self.cudaUse:
                    mix_stft = mix_stft.detach().cpu()
                    for spk_idx in range(self.num_spks):
                        MISO1_stft[spk_idx] = MISO1_stft[spk_idx].detach().cpu()

            """
            Source Alignment between Clean reference signal and MISO1 signal 
            calculate magnitude distance between ref mic(ch0) and target signal(reference mic : ch0) 
            """

            for spk_idx in range(self.num_spks):
                if spk_idx == 0 :
                    ref_ = ref_stft_1ch[spk_idx]
                    s_MISO1 = MISO1_stft[spk_idx][:,self.ref_ch,:,:] # [B,T,F]
                else:
                    ref_ = torch.cat((ref_,ref_stft_1ch[spk_idx]), dim=1)
                    s_MISO1 = torch.stack((s_MISO1, MISO1_stft[spk_idx][:,self.ref_ch,:,:]), dim=1)

            s_MISO1_ = torch.unsqueeze(s_MISO1,dim=2) #[B,Spks,1,T,F]
            magnitude_MISO1 = torch.abs(torch.sqrt(s_MISO1_.real**2 + s_MISO1_.imag**2)) #[B,Spks,1,T,F]
            
            s_ref = torch.unsqueeze(ref_, dim=1)
            magnitude_distance = torch.sum(torch.abs(magnitude_MISO1 - abs(s_ref)),[3,4])
            perms = ref_.new_tensor(list(permutations(range(self.num_spks))), dtype=torch.long) #[[0,1],[1,0]]
            index_ = torch.unsqueeze(perms, dim=2)
            perms_one_hot = ref_.new_zeros((*perms.size(), self.num_spks), dtype=torch.float).scatter_(2,index_,1)
            batchwise_distance = torch.einsum('bij,pij->bp',[magnitude_distance, perms_one_hot])
            min_distance_idx = torch.argmin(batchwise_distance, dim=1)

            for batch_idx in range(B):
                align_index = torch.argmax(perms_one_hot[min_distance_idx[batch_idx]], dim=1)
                for spk_idx in range(self.num_spks):
                    target_index = align_index[spk_idx]
                    ref_stft_1ch[spk_idx] = torch.unsqueeze(ref_[batch_idx,target_index,...],dim=0) 
                    
            """
                Apply MVDR Beamforming
                    Output :
                            Beamform_stft, [reference Mic channel, T, F]
            """

            if self.functionMode == 'Enhance_Load_MVDR_Output' or self.functionMode == 'Enhance_Load_MISO1_MVDR_Output':

                with open(os.path.join(self.Beamforming_pickle_Rootdir, wav_name), 'rb') as f:
                    Beamform_infos = pickle.load(f)
                f.close()

                Beamform_stft = [[] for _ in range(self.num_spks)]
                for spk_idx in range(self.num_spks):                
                    Beamform_stft[spk_idx] = torch.permute(torch.from_numpy(self.STFT(Beamform_infos['Beamforming_{}'.format(spk_idx+1)])),[0,2,1])
                    Beamform_stft[spk_idx] = Beamform_stft[spk_idx]/self.scale #[1, T, F]
            
            else:

                Beamform_stft = [[] for _ in range(self.num_spks)]
                for spk_idx in range(self.num_spks):
                    source = torch.permute(MISO1_stft[spk_idx],[0,3,1,2]).numpy() # [B, F, M, T]
                    observe = torch.permute(mix_stft,[0,3,1,2]).detach().cpu() # [B, F, M, T]
                    Beamform_stft[spk_idx] = self.Apply_Beamforming(source, observe)                
                
            if len(mix_stft.shape)== 4:
                mix_stft = torch.squeeze(mix_stft) #[M,T,F]
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


    def MISO1_Inference(self,mix_stft,ref_ch=0):
        """
        Input:
            mix_stft : observe STFT, size - [B, Mic, T, F]
        Output:
            MISO1_stft : list of separated source, - [B, reference Mic, T, F]

            1. circular shift the microphone array at run time for the prediction of each microphone signal
               If the microphones are arranged uniformly on a circle, Select the reference microphone by circular shifting the microphone. e.g reference mic q -> [Yq, Yq+1, ..., Yp, Y1, ..., Yq-1]
            2. Using Permutation Invariance Alignmnet method to match between clean target signal and estimated signal
        """
        B, M, T, F = mix_stft.size()

        MISO1_stft = [torch.empty(B,M,T,F, dtype=torch.complex64) for _ in range(self.num_spks)]
        
        Mic_array = [x for x in range(M)]
        Mic_array = np.roll(Mic_array, -ref_ch)  # [ref_ch, ref_ch+1, ..., 0, 1, ..., ref_ch-1]
        # print('Mic_array : ', Mic_array)

        with torch.no_grad():
            mix_stft_refCh = torch.roll(mix_stft,-ref_ch, dims=1)
            MISO1_refCh = self.model(mix_stft_refCh)

        for spk_idx in range(self.num_spks):
            MISO1_stft[spk_idx][:,ref_ch,...] = MISO1_refCh[:,spk_idx,...]
            
        # MISO1_spk1[:,ref_ch,...] = MISO1_refCh[:,0,...]
        # MISO1_spk2[:,ref_ch,...] = MISO1_refCh[:,1,...]

        s_MISO1_refCh = torch.unsqueeze(MISO1_refCh, dim=2)
        s_Magnitude_refCh = torch.abs(torch.sqrt(s_MISO1_refCh.real**2 + s_MISO1_refCh.imag**2)) # [B,Spks,1,T,F]
        
        with torch.no_grad():
            for shiftIdx in Mic_array[1:]:
                # print('shift Micnumber', shiftIdx)
                
                mix_stft_shift = torch.roll(mix_stft,-shiftIdx, dims=1)
                MISO1_chShift = self.model(mix_stft_shift)

                s_MISO1_chShift = torch.unsqueeze(MISO1_chShift, dim=1) #[B,1,Spks,T,F]
                s_magnitude_chShift = torch.sum(torch.abs(s_Magnitude_refCh - abs(s_MISO1_chShift)),[3,4]) #[B,Spks,Spks,T,F]
                perms = MISO1_chShift.new_tensor(list(permutations(range(self.num_spks))), dtype=torch.long)
                index_ = torch.unsqueeze(perms, dim=2)
                perms_one_hot = MISO1_chShift.new_zeros((*perms.size(), self.num_spks), dtype=torch.float).scatter_(2,index_,1)
                batchwise_distance = torch.einsum('bij,pij->bp', [s_magnitude_chShift, perms_one_hot])
                min_distance_idx = torch.argmin(batchwise_distance,dim=1)
                
                for batch_idx in range(B):              
                    align_index = torch.argmax(perms_one_hot[min_distance_idx[batch_idx]],dim=1)
                    for spk_idx in range(self.num_spks):
                        target_index = align_index[spk_idx]     
                        MISO1_stft[spk_idx][:,shiftIdx,...] = MISO1_chShift[batch_idx,target_index,...]
        

        return MISO1_stft     

    def Apply_Beamforming(self, source_stft, mix_stft, epsi=1e-6):
        """
        Input :
            mix_stft : observe STFT, size - [B, F, Ch, T], np.ndarray
            source_stft : estimated source STFT, size - [B, F, Ch, T], np.ndarray
        Output :    
            Beamform_stft : MVDR Beamforming output, size - [B, 1, T, F], np.ndarray
        
            1. estimate target steering using EigenValue decomposition
            2. get source, noise Spatial Covariance Matrix,  S = 1/T * xx_h
            3. MVDR Beamformer
        """
        B, F, M, T = source_stft.shape

        # Apply small Diagonal matrix to prevent matrix inversion error
        eye = np.eye(M)
        eye = eye.reshape(1,1,M,M)
        delta = epsi * np.tile(eye,[B,F,1,1])

        ''' Source '''
        source_SCM = self.get_spatial_covariance_matrix(source_stft,normalize=True) # target covariance matrix, size : [B,F,C,C]
        source_SCM = 0.5 * (source_SCM + np.conj(source_SCM.swapaxes(-1,-2))) # verify hermitian symmetric  
        
        ''' Noise Spatial Covariance ''' 
        noise_signal = mix_stft - source_stft
        # s1_noise_signal = mix_stft  #MPDR
        noise_SCM = self.get_spatial_covariance_matrix(noise_signal,normalize = True) # noise covariance matrix, size : [B,F,C,C]
        # s1_SCMn = self.condition_covariance(s1_SCMn, 1e-6)
        # s1_SCMn /= np.trace(s1_SCMn, axis1=-2, axis2= -1)[...,None, None]
        noise_SCM = 0.5 * (noise_SCM + np.conj(noise_SCM.swapaxes(-1,-2))) # verify hermitian symmetric

        ''' Get Steering vector : Eigen-decomposition '''
        shape = source_SCM.shape
        source_steering = np.empty(shape[:-1], dtype=np.complex)

        # s1_SCMs += delta
        source_SCM = np.reshape(source_SCM, (-1,) + shape[-2:]) 
        eigenvals, eigenvecs = np.linalg.eigh(source_SCM)
        # Find max eigenvals
        vals = np.argmax(eigenvals, axis=-1)
        # Select eigenvec for max eigenval
        source_steering = np.array([eigenvecs[i,:,vals[i]] for i in range(eigenvals.shape[0])])
        # s1_steering = np.array([eigenvecs[i,:,vals[i]] * np.sqrt(Mic/np.linalg.norm(eigenvecs[i,:,vals[i]])) for i in range(eigenvals.shape[0])]) # [B*F,Ch,Ch]
        source_steering = np.reshape(source_steering, shape[:-1]) # [B,F,Ch]
        source_SCM = np.reshape(source_SCM, shape)
        
        ''' steering normalize with respect to the reference microphone '''
        # ver 1 
        source_steering = source_steering / np.expand_dims(source_steering[:,:,0], axis=2)
        for b_idx in range(0,B):
            for f_idx in range(0,F):
                # s1_steering[b_idx,f_idx,:] = s1_steering[b_idx,f_idx,:] / s1_steering[b_idx,f_idx,0]
                source_steering[b_idx,f_idx,:] = source_steering[b_idx,f_idx,:] * np.sqrt(M/(np.linalg.norm(source_steering[b_idx,f_idx,:])))
        
        # ver 2
        # s1_steering = self.normalize(s1_steering)

        source_steering = self.PhaseCorrection(source_steering)
        beamformer = self.get_mvdr_beamformer(source_steering, noise_SCM, delta)
        # s1_beamformer = self.blind_analytic_normalization(s1_beamformer,s1_SCMn)
        source_bf = self.apply_beamformer(beamformer,mix_stft)
        source_bf = torch.permute(torch.from_numpy(source_bf), [0,2,1])
        
        return source_bf


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

    def __init__(self,sms_wsj_observation_dir,speech_source_dir, chunk_time, ref_ch, num_ch_utilize, **STFT_args):
        
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
        self.num_ch_utilize = num_ch_utilize
        
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
            M, T, F = split_observe_stft.shape
            split_observe_dict[str(split_idx)] = split_observe_stft[0:M:M//self.num_ch_utilize,...]

            split_clean_s0_wav = np.pad(wav_clean_s0, ((0,gap), (0,0)), constant_values=0)
            split_clean_s0_stft = self.STFT(split_clean_s0_wav)
            split_clean_s0_stft = torch.permute(torch.from_numpy(split_clean_s0_stft/self.scale), [0,2,1])
            M, T, F = split_clean_s0_stft.shape
            split_clean_s0_dict[str(split_idx)] = split_clean_s0_stft[0:M:M//self.num_ch_utilize]

            split_clean_s1_wav = np.pad(wav_clean_s1, ((0,gap), (0,0)), constant_values=0)
            split_clean_s1_stft = self.STFT(split_clean_s1_wav)
            split_clean_s1_stft = torch.permute(torch.from_numpy(split_clean_s1_stft/self.scale), [0,2,1])
            M, T, F = split_clean_s1_stft.shape
            split_clean_s1_dict[str(split_idx)] = split_clean_s1_stft[0:M:M//self.num_ch_utilize]

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
                M, T, F = split_observe_stft.shape
                split_observe_dict[str(split_idx)] = split_observe_stft[0:M:M//self.num_ch_utilize]

                split_clean_s0_stft = self.STFT(split_clean_s0_wav)
                split_clean_s0_stft = torch.permute(torch.from_numpy(split_clean_s0_stft/self.scale),[0,2,1])
                M, T, F = split_clean_s0_stft.shape
                split_clean_s0_dict[str(split_idx)] = split_clean_s0_stft[0:M:M//self.num_ch_utilize]

                split_clean_s1_stft = self.STFT(split_clean_s1_wav)
                split_clean_s1_stft = torch.permute(torch.from_numpy(split_clean_s1_stft/self.scale),[0,2,1])
                M, T, F = split_clean_s1_stft.shape
                split_clean_s1_dict[str(split_idx)] = split_clean_s1_stft[0:M:M//self.num_ch_utilize]

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


