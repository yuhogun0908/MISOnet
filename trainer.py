import torch
import time
from tqdm import tqdm
from criterion import loss_uPIT, loss_uPIT_v1, loss_Enhance
from torch.utils.tensorboard import SummaryWriter
import pdb
from pathlib import Path
import os
#import criterion import cal_loss
from utils.writer import MyWriter
import numpy as np
import math
from itertools import permutations
from scipy import signal
import soundfile as sf
from numpy.linalg import solve
from scipy.linalg import eig
from scipy.linalg import eigh
import scipy 
from pathlib import Path
import pickle
<<<<<<< HEAD
from tqdm import tqdm
import random
=======
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d

class Trainer_Separate(object):
    def __init__(self,dataset, num_spks, tr_loader,dt_loader,model, optimizer,scheduler,config,device,log_path):
        self.num_spks = num_spks
        self.tr_loader = tr_loader
        self.dt_loader = dt_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model = model
        self.dataset = dataset
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("model parameters :{}".format(params))
        self.device = device
        self.log_path = log_path
        #from visdom import Visdom
        self.tr_avg_loss = torch.Tensor(config['trainer_sp']['epochs'])
        self.val_avg_loss = torch.Tensor(config['trainer_sp']['epochs'])
        self.save_folder = config['trainer_sp']['save_folder']
        self.model_path = config['trainer_sp']['model_path']
        Path(self.save_folder).mkdir(exist_ok=True, parents=True)
        # self.num_params = sum(
            # [param.nelement() for param in nnet.parameters()]) / 10.0**6
        self.model_load = config['trainer_sp']['model_load']
        self._reset()
        self.audiowritter = SummaryWriter(self.log_path+'_audio')
        self.writter = MyWriter(self.config, self.log_path)
        self.clip_norm = config['trainer_sp']['clipping']
    def _reset(self):
        # model load, tr&val loss, optimizer, 시작 epoch 추가

        if self.config['trainer_sp']['model_load'][0]:
            print("Loading checkpoint model %s" % self.config['trainer_sp']['model_load'][1])
            package = torch.load(self.config['trainer_sp']['model_load'][1],map_location= "cuda:"+str(self.device))
            self.model.load_state_dict(package['model_state_dict'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.start_epoch = int(package.get('epoch',1))
            self.tr_avg_loss[:self.start_epoch] = package['tr_avg_loss'][:self.start_epoch]
            self.val_avg_loss[:self.start_epoch] = package['val_avg_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0

        self.prev_val_loss = float("inf")
        self.best_val_loss_epoch = float("inf")
        self.val_no_impv = 0
        self.halving = False

    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.config['trainer_sp']['epochs'])):
<<<<<<< HEAD
            print('Separate Training Start ...')
=======
            print('Training...')
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d
            start = time.time()
            self.model.train()
            tr_avg_loss_epoch = self._run_one_epoch(epoch,training=True)
            
            print('-'* 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(epoch+1, time.time()-start,tr_avg_loss_epoch))
            print('_'* 85)
            # writter.add_scalar('data/Train_Loss', tr_avg_loss_epoch, epoch)
            self.writter.log_value(tr_avg_loss_epoch,epoch,'data/Train_Loss')


            #save model per 10 epochs
            if self.config['trainer_sp']['check_point'][0]:
                if epoch % self.config['trainer_sp']['check_point'][1] == 0 :
                    file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch+1))
                    state_dict = {
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch' : epoch+1,
                        'tr_avg_loss' : self.tr_avg_loss,
                        'val_avg_loss' : self.val_avg_loss
                    }
                    torch.save(state_dict, file_path)
                    print('Saving checkpoint model to %s' % file_path)
            print('validation...')
            self.model.eval()
            val_avg_loss_epoch = self._run_one_epoch(epoch,training=False)
            print('-'* 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s |'
                  'Valid Loss {2:.3f}'.format(epoch+1, time.time()-start,val_avg_loss_epoch))
            print('_'* 85)
            # writter.add_scalar('data/Validation_Loss', val_avg_loss_epoch, epoch)
            self.writter.log_value(val_avg_loss_epoch,epoch,'data/Validation_Loss')

            # scheduler
            # if self.config['trainer']['half_lr']:
            #     if val_avg_loss_epoch >= self.prev_val_loss:
            #         self.val_no_impv += 1
            #         if self.val_no_impv >= 3:
            #             self.halving = True
            #         assert(self.val_no_impv <10 and not(self.config['trainer']['early_stop'])), "No improvement for 10 epochs, ealry stopping"
            # if self.halving:
            #     optim_state = self.optimizer.state_dict()
            #     optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2.0
            #     self.optimizer.load_state_dict(optim_state)
            #     print('Learning rate adjusted to :{lr:.6f}'.format(lr=optim_stae['param_groups'][0]['lr']))
            #     self.halving = False
            # self.prev_val_loss = val_avg_loss_epoch

            # save best model
            self.tr_avg_loss[epoch] = tr_avg_loss_epoch
            self.val_avg_loss[epoch] = val_avg_loss_epoch
            if val_avg_loss_epoch < self.best_val_loss_epoch:
                self.best_val_loss_epoch = val_avg_loss_epoch
                file_path = os.path.join(self.save_folder, self.model_path)
                state_dict = {
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch' : epoch+1,
                        'tr_avg_loss' : self.tr_avg_loss,
                        'val_avg_loss' : self.val_avg_loss
                    }
                torch.save(state_dict, file_path)
                print("Find better validated model, saving to %s" % file_path)

            self.scheduler.step(val_avg_loss_epoch)
        # writter.close()

    def _run_one_epoch(self, epoch, training= True):
        start = time.time()
        total_loss = 0
        loss = 0
        data_loader = self.tr_loader if training else self.dt_loader
        for idx, (data) in enumerate(data_loader):
            
            """
            Input : [B,Mic,T,F]
            """
            mix_stft, ref_stft = data
            mix_stft = mix_stft.cuda(self.device)
            # ref1_stft = ref1_stft.cuda(self.device)
            # ref2_stft = ref2_stft.cuda(self.device)
            B, Mic, T, F = mix_stft.size()
            estimate_sources = self.model(mix_stft) #[B,Spk,T,F]

            # reference mic 1 : Train
            ref_stft_1ch = [ [] for _ in range(self.num_spks)]
            for spk_idx in range(self.num_spks):
                #[B,1,T,F]
                ref_stft_1ch[spk_idx] = ref_stft[spk_idx][:,0,:,:].cuda(self.device)

            # ref1_stft_1ch = ref1_stft[:,0,:,:] #[B,1,T,F]
            # ref2_stft_1ch = ref2_stft[:,0,:,:] #[B,1,T,F]


            _, Spks, _, _ = estimate_sources.size()
            assert Spks == self.num_spks, '[ERROR] please check the number of speakers'

            # Original Loss
            loss = loss_uPIT(self.num_spks, estimate_sources, ref_stft_1ch)
            
            # Loss에 maximum(abs(estimate - target), zeros) 추가
            # alpha = (epoch+1) * 0.03
            # zeros = torch.zeros(B,Spks,Spks,T,F).cuda(self.device)
            # loss = loss_uPIT_v1(self.num_spks,estimate_sources,ref_stft_1ch,zeros,alpha)

            if not training and idx == 0:
                # [B,1,T,F] -> [T,F]
                mix_test = np.transpose(mix_stft[0,0,:,:].cpu().detach().numpy(),[1,0])
                ref_test = [ np.transpose(ref_sig[0,0,:,:].cpu().detach().numpy(),[1,0]) for ref_sig in ref_stft_1ch]
                # ref1_test = np.transpose(ref1_stft_1ch[0,:,:].cpu().detach().numpy(),[1,0])
                # ref2_test = np.transpose(ref2_stft_1ch[0,:,:].cpu().detach().numpy(),[1,0])
                estimate_test = [ np.transpose(estim_sig.cpu().detach().numpy(),[1,0]) for estim_sig in estimate_sources[0,:,:,:]]
                # estim1_test = np.transpose(estimate_sources[0,0,:,:].cpu().detach().numpy(),[1,0])
                # estim2_test = np.transpose(estimate_sources[0,1,:,:].cpu().detach().numpy(),[1,0])

                self.writter.log_spec(mix_test,'mix',epoch+1)
                for spk_idx in range(self.num_spks):
                    self.writter.log_spec(ref_test[spk_idx],'clean'+str(spk_idx+1),epoch+1)
                    # self.writter.log_spec(ref2_test,'clean2',epoch+1)
                    self.writter.log_spec(estimate_test[spk_idx],'estim'+str(spk_idx+1),epoch+1)
                    # self.writter.log_spec(estim2_test,'estim2',epoch+1) 
                mix, clean, estim = self.writter.log_audio(self.num_spks,mix_test,ref_test,estimate_test,epoch+1)

                self.audiowritter.add_audio('mix', mix/max(abs(mix)), epoch+1, self.config[self.dataset]['fs'])
                for spk_idx in range(self.num_spks):
                    self.audiowritter.add_audio('clean'+str(spk_idx+1), clean[spk_idx]/max(abs(clean[spk_idx])), epoch+1, self.config[self.dataset]['fs'])
                    self.audiowritter.add_audio('estim'+str(spk_idx+1), estim[spk_idx]/max(abs(estim[spk_idx])), epoch+1, self.config[self.dataset]['fs'])

            #Source Alignment across Microphones module should be implemented
            # when beamforming is performed after training.
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                #gradient threshold to clip
                if self.clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.config['trainer_sp']['max_norm'])
                self.optimizer.step()

            total_loss += loss.item()
            
            if idx % self.config['trainer_sp']['print_freq'] == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} |'
                        'Current Loss {3:6f} | {4:.1f} ms/batch'.format(
                            epoch + 1, idx+1, total_loss / (idx+1), 
                            loss.item(), 1000*(time.time()-start)/(idx+1)),
                            flush = True)

        return total_loss /(idx+1)

class Trainer_Beamforming(object):
    def __init__(self, dataset, num_spks, tr_loader, dt_loader, model_sep, config, device,log_path):
        self.num_spks = num_spks
        self.tr_loader = tr_loader
        self.dt_loader = dt_loader
        self.config = config
        self.model_sep = model_sep
        self.dataset = dataset
        self.Spks = config[dataset]['num_spks']
        self.device = device
        #MISO1 separation model load
<<<<<<< HEAD
        self.MISO1_path = config['trainer_beamform']['MISO1_path']
        self._load()
        self.check_MISO1 = config['trainer_beamform']['check_output']
        self.saveOutput = config['trainer_beamform']['save_output']
=======
        self.MISO1_path = config['trainer_en']['MISO1_path']
        self._load()
        self.check_MISO1 = config['check_MISO1']
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d
        self.log_path = log_path
        self.writter = MyWriter(self.config, self.log_path)


    def _load(self):
        ''' Load pretrained MISO1 '''
        print('Loading MISO_1 model %s'% self.MISO1_path)
        package = torch.load(self.MISO1_path, map_location="cuda:"+str(self.device))
        self.model_sep.load_state_dict(package['model_state_dict'])

    def train(self):
<<<<<<< HEAD
        print('Beaforming Start...')
=======
        print('Training...')
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d
        start = time.time()
        tr_avg_loss_epoch = self._run_one_epoch(0,training=True)
           
        print('validation...')
        val_avg_loss_epoch = self._run_one_epoch(0,training=False)
            
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
    
    
    def _run_one_epoch(self, epoch, training= True):
        self.model_sep.eval()

        start = time.time()
        data_loader = self.tr_loader if training else self.dt_loader

<<<<<<< HEAD
        for idx, (data) in tqdm(enumerate(data_loader)):
=======
        for idx, (data) in enumerate(data_loader):
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d
            
            """
            Input : 
                mix_stft : two source mixture with reverb, white noise 
                            size : [B,Mic,T,F]
                ref_stft : direct path source
                            size : [B,Mic,T,F]
                BeamOutDir: directory to save beamforming output
                            type : pickle 
            """
            #################################################################################
<<<<<<< HEAD
            mix_stft, ref_stft, BeamOutDir, MISO1OutDir = data
=======
            mix_stft, ref_stft, BeamOutDir = data
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d
            mix_stft = mix_stft.cuda(self.device)
            # reference mic 1 : Train
            ref_stft_1ch = [ [] for _ in range(self.num_spks)]
            for spk_idx in range(self.num_spks):
                #[B,1,T,F]
                ref_stft_1ch[spk_idx] = ref_stft[spk_idx][:,0,:,:].cuda(self.device)
                ref_stft_1ch[spk_idx] = torch.unsqueeze(ref_stft_1ch[spk_idx], dim=1)

            for spk_idx in range(self.num_spks):
                ref_stft[spk_idx] = ref_stft[spk_idx].cuda(self.device)
            B, Mic, T, F = mix_stft.size()

            eye = np.eye(Mic)
            eye = eye.reshape(1,1,Mic,Mic)
            delta = 1e-6* np.tile(eye,[B,F,1,1])

            ''' Source Alignment across Microphones'''
            s1_ch6 = torch.empty(B,Mic,T,F, dtype=torch.complex64)
            s2_ch6 = torch.empty(B,Mic,T,F, dtype=torch.complex64)
            ref1 = torch.empty(B,1,T,F, dtype=torch.complex64)
            ref2 = torch.empty(B,1,T,F, dtype=torch.complex64)

            with torch.set_grad_enabled(False):
                ref_sources = self.model_sep(mix_stft) #[B,Spk,T,F]
    
            #######################################
            ''' Source-wise Alignment ''' 
            ## Source wise alignment 기능 정상 작동
            #######################################
            s1_ch6[:,0,:,:] = ref_sources[:,0,:,:] 
            s2_ch6[:,0,:,:] = ref_sources[:,1,:,:]


            s_ref_sources = torch.unsqueeze(ref_sources,dim=2) #[B,Spks,1,T,F]
            ref_magnitude = torch.abs(torch.sqrt(s_ref_sources.real**2 + s_ref_sources.imag**2)) #[B,Spks,1,T,F]
            with torch.set_grad_enabled(False):
                for ref_mic in range(1,Mic):
                    # Select the reference microphone by circular shifting the microphone
                    # [Yq, ... , Yp, ... , Yq-1]
                    
                    s_mix_stft = torch.roll(mix_stft,-ref_mic, dims=1)
                    estimate_sources = self.model_sep(s_mix_stft) #[B,Spks,T,F]
                    # calculate magnitude distance between ref mic and non ref mic
                    s_estimate_sources = torch.unsqueeze(estimate_sources,dim=1) #[B,1,Spks,T,F]
                    magnitude_distance = torch.sum(torch.abs(ref_magnitude - abs(s_estimate_sources)),[3,4]) #[B,Spks,Spks,T,F]
                    
                    perms = estimate_sources.new_tensor(list(permutations(range(self.Spks))), dtype=torch.long) #[[0,1],[1,0]]
                    index = torch.unsqueeze(perms, dim=2)
                    perms_one_hot = estimate_sources.new_zeros((*perms.size(), self.Spks), dtype=torch.float).scatter_(2,index,1)

                    batchwise_distance = torch.einsum('bij,pij->bp',[magnitude_distance, perms_one_hot])
                    min_distance_idx = torch.argmin(batchwise_distance,dim=1)

                    for ii in range(B):
                        if min_distance_idx[ii] == 1:
                            s1_ch6[ii,ref_mic,:,:] = estimate_sources[ii,1,:,:]
                            s2_ch6[ii,ref_mic,:,:] = estimate_sources[ii,0,:,:]
                        else:
                            s1_ch6[ii,ref_mic,:,:] = estimate_sources[ii,0,:,:]
                            s2_ch6[ii,ref_mic,:,:] = estimate_sources[ii,1,:,:]

                # calculate magnitude distance between ref mic and target signal

                for spk_idx in range(self.num_spks):
                    if spk_idx == 0 :
                        ref_ = ref_stft_1ch[spk_idx]
                    else:
                        ref_ = torch.cat((ref_,ref_stft_1ch[spk_idx]), dim=1)

<<<<<<< HEAD
                s_estimate_sources = torch.unsqueeze(ref_, dim=1)
                magnitude_distance = torch.sum(torch.abs(ref_magnitude - abs(s_estimate_sources)),[3,4])
=======
                s_ref = torch.unsqueeze(ref_, dim=1)
                magnitude_distance = torch.sum(torch.abs(ref_magnitude - abs(s_ref)),[3,4])
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d
                perms = ref_.new_tensor(list(permutations(range(self.Spks))), dtype=torch.long) #[[0,1],[1,0]]
                index = torch.unsqueeze(perms, dim=2)
                perms_one_hot = ref_.new_zeros((*perms.size(), self.Spks), dtype=torch.float).scatter_(2,index,1)
                batchwise_distance = torch.einsum('bij,pij->bp',[magnitude_distance, perms_one_hot])
                min_distance_idx = torch.argmin(batchwise_distance, dim=1)

                for ii in range(B):
                    if min_distance_idx[ii] == 1:
                        ref1[ii,...] = ref_[ii,1,:,:]
                        ref2[ii,...] = ref_[ii,0,:,:]
                    else:
                        ref1[ii,...] = ref_[ii,0,:,:]
                        ref2[ii,...] = ref_[ii,1,:,:]
                ref1 = ref1.cuda(self.device)
                ref2 = ref2.cuda(self.device)

                
            ''' MVDR Beamforming '''
            """
                This code is based on nn_gev, https://github.com/fgnt/nn-gev/blob/master/fgnt
            """
            # torch to numpy
            # s1_ch6 : [B,Ch,T,F]
            s1_ch6 = torch.permute(s1_ch6,[0,3,1,2]) # [B, F, Ch, T]
            s1_ch6 = s1_ch6.detach().cpu().numpy()
            s2_ch6 = torch.permute(s2_ch6,[0,3,1,2])
            s2_ch6 = s2_ch6.detach().cpu().numpy()
            mix_stft = torch.permute(mix_stft,[0,3,1,2])
            mix_stft = mix_stft.detach().cpu().numpy()
            #######################################################################################


            # ############# Test Sample Data to verify ##############
            # mix_stft, ref_stft, s1_ch6, s2_ch6, BeamOutDir = data
            # B, Mic, F, T = mix_stft.size()
            # # [B, Ch, F, T]
            # s1_ch6 = torch.permute(s1_ch6,[0,2,1,3])
            # s1_ch6 = s1_ch6.detach().cpu().numpy()
            # s2_ch6 = torch.permute(s2_ch6,[0,2,1,3])
            # s2_ch6 = s2_ch6.detach().cpu().numpy()
            # mix_stft = torch.permute(mix_stft,[0,2,1,3])
            # mix_stft = mix_stft.detach().cpu().numpy()
            # ref_stft_1ch = [ [] for _ in range(self.num_spks)]
            # for spk_idx in range(self.num_spks):
            #     #[B,1,T,F]
            #     ref_stft_1ch[spk_idx] = ref_stft[spk_idx][:,0,:,:].cuda(self.device)
            #     ref_stft_1ch[spk_idx] = torch.unsqueeze(ref_stft_1ch[spk_idx], dim=1)

            # for spk_idx in range(self.num_spks):
            #     ref_stft[spk_idx] = ref_stft[spk_idx].cuda(self.device)
            # eye = np.eye(Mic)
            # eye = eye.reshape(1,1,Mic,Mic)
            # delta = 1e-6* np.tile(eye,[B,F,1,1])
            # ########################################################

            ####################
            ''' Source 1 '''
            ####################
            s1_SCMs = self.get_spatial_covariance_matrix(s1_ch6,normalize=True) # target covariance matrix, size : [B,F,C,C]
            s1_SCMs = 0.5 * (s1_SCMs + np.conj(s1_SCMs.swapaxes(-1,-2))) # verify hermitian symmetric  
            
            ''' Noise Spatial Covariance ''' 
            s1_noise_signal = mix_stft - s1_ch6
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
            s1_beamout = self.apply_beamformer(s1_beamformer,mix_stft)
            
            ####################
            ''' source 2 '''
            ####################
            s2_SCMs = self.get_spatial_covariance_matrix(s2_ch6, normalize = True) # target covariance matrix, size : [B,F,C,C]
            s2_SCMs = 0.5 * (s2_SCMs + np.conj(s2_SCMs.swapaxes(-1,-2))) # verify hermitian symmetric  

            ''' Noise Spatial Covariance '''
            s2_noise_signal = mix_stft - s2_ch6
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
            s2_beamout = self.apply_beamformer(s2_beamformer,mix_stft)

<<<<<<< HEAD
            if self.saveOutput:
                hann_win = scipy.signal.get_window('hann', self.config['ISTFT']['length'])
                scale = np.sqrt(1.0 / hann_win.sum()**2)
                MAX_INT16 = np.iinfo(np.int16).max

                # Save MISO1 Output to pickle format
                for fileIdx in range(0,len(MISO1OutDir)):
                    fileRoot = MISO1OutDir[fileIdx]
                    if fileIdx == 0:
                        parentRoot = Path(MISO1OutDir[fileIdx]).parent.absolute()
                        parentRoot.mkdir(exist_ok = True, parents = True)
                    
                    s1_MISO1_ = np.transpose(s1_ch6,[0,2,1,3])
                    s1_MISO1_ = s1_MISO1_ * scale
                    s1_t_sig = self.ISTFT(s1_MISO1_[fileIdx,...])
                    s1_t_sig = s1_t_sig * MAX_INT16
                    s1_t_sig = s1_t_sig.astype(np.int16)

                    s2_MISO1_ = np.transpose(s2_ch6,[0,2,1,3])
                    s2_MISO1_ = s2_MISO1_ * scale
                    s2_t_sig = self.ISTFT(s2_MISO1_[fileIdx,...])
                    s2_t_sig = s2_t_sig * MAX_INT16
                    s2_t_sig = s2_t_sig.astype(np.int16)

                    sf.write(str(fileRoot).replace('.pickle', '_s1.wav'),s1_t_sig.T, self.config['ISTFT']['fs'],'PCM_24')
                    sf.write(str(fileRoot).replace('.pickle', '_s2.wav'),s2_t_sig.T, self.config['ISTFT']['fs'],'PCM_24')
                    
                    # s_dict = {'s1_MISO1': s1_t_sig, 's2_MISO1': s2_t_sig}
                    # with open(str(fileRoot), 'wb') as f:
                        # pickle.dump(s_dict, f)
                # f.close()
            
                # Save MVDR beamformer Ouput to pickle format
                for fileIdx in range(0,len(BeamOutDir)):
                    fileRoot = BeamOutDir[fileIdx]
                    if fileIdx == 0:
                        parentRoot = Path(BeamOutDir[fileIdx]).parent.absolute()
                        parentRoot.mkdir(exist_ok = True, parents = True)
                    
                    s1_beamout_ = s1_beamout * scale
                    s1_bf_t_sig = self.ISTFT(s1_beamout_[fileIdx,...])
                    s1_bf_t_sig = s1_bf_t_sig * MAX_INT16
                    s1_bf_t_sig = s1_bf_t_sig.astype(np.int16)

                    s2_beamout_ = s2_beamout * scale
                    s2_bf_t_sig = self.ISTFT(s2_beamout_[fileIdx,...])
                    s2_bf_t_sig = s2_bf_t_sig * MAX_INT16
                    s2_bf_t_sig = s2_bf_t_sig.astype(np.int16)

                    sf.write(str(fileRoot).replace('.pickle', '_s1.wav'),s1_bf_t_sig.T, self.config['ISTFT']['fs'],'PCM_24')
                    sf.write(str(fileRoot).replace('.pickle', '_s2.wav'),s2_bf_t_sig.T, self.config['ISTFT']['fs'],'PCM_24')

                    pdb.set_trace()
                
                #     s_dict = {'s1_BF': s1_t_sig, 's2_BF': s2_t_sig}
                #     with open(str(fileRoot), 'wb') as f:
                #         pickle.dump(s_dict, f)
                # f.close()

                print('Save Output {} / {} ----- {}%'.format(idx, len(data_loader), (idx//len(data_loader)*100) ))

            if self.check_MISO1:

                ''' Check the result of MISO1 '''                
                mix_stft = mix_stft * scale
                t_sig = self.ISTFT(mix_stft[0,:,:,:].swapaxes(0,1))
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * MAX_INT16
                t_sig = t_sig.astype(np.int16)
                sf.write('mix.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')

                s1_MISO1 =  np.transpose(s1_ch6,[0,2,1,3])         
                s1_MISO1 = s1_MISO1 * scale
                t_sig = self.ISTFT(s1_MISO1[0,0,...])
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * MAX_INT16
                t_sig = t_sig.astype(np.int16)
                sf.write('MISO1.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')

                s2_MISO1 =  np.transpose(s2_ch6,[0,2,1,3])         
                s2_MISO1 = s2_MISO1 * scale
                t_sig = self.ISTFT(s2_MISO1[0,0,...])
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * MAX_INT16
                t_sig = t_sig.astype(np.int16)
                sf.write('MISO2.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')

                s1_beamout = s1_beamout * scale
                t_sig = self.ISTFT(s1_beamout[0,...])
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * MAX_INT16
                t_sig = t_sig.astype(np.int16)
                sf.write('e1.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')
                
                s2_beamout = s2_beamout * scale
                t_sig = self.ISTFT(s2_beamout[0,...])
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * MAX_INT16
                t_sig = t_sig.astype(np.int16)
                sf.write('e2.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')

                s1_clean = ref_stft_1ch[0]
                s1_clean = torch.permute(s1_clean,[0,1,3,2]).detach().cpu().numpy()         
                s1_clean = s1_clean * scale
                t_sig = self.ISTFT(s1_clean[0,...])
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * MAX_INT16
                t_sig = t_sig.astype(np.int16)
                sf.write('clean1.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')

                s2_clean = ref_stft_1ch[1]
                s2_clean = torch.permute(s2_clean,[0,1,3,2]).detach().cpu().numpy()         
                s2_clean = s2_clean * scale
                t_sig = self.ISTFT(s2_clean[0,...])
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * MAX_INT16
                t_sig = t_sig.astype(np.int16)
                sf.write('clean2.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')

                pdb.set_trace()
                
        print('Beaforming Finished')

        return 

    def ISTFT(self,FT_sig): 

        '''
        input : [F,T]
        output : [T,C]
        '''
        # if FT_sig.shape[1] != self.config['ISTFT']['length']+1:
            # FT_sig = np.transpose(FT_sig,(0,1)) # [C,T,F] -> [C,F,T]

        fs = self.config['ISTFT']['fs']; window = self.config['ISTFT']['window']; nperseg=self.config['ISTFT']['length']; noverlap=self.config['ISTFT']['overlap']
        _, t_sig = signal.istft(FT_sig,fs=fs, window=window, nperseg=nperseg, noverlap=noverlap) #[C,F,T] -> [T,C]

        return t_sig


class Trainer_Enhance(object):
    def __init__(self,dataset,enhanceModelType, num_spks, tr_loader,dt_loader, model, optimizer,scheduler,config,device,log_path):
        self.num_spks = num_spks
        self.tr_loader = tr_loader
        self.dt_loader = dt_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model = model
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("model parameters :{}".format(params))
        self.dataset = dataset
        self.Spks = config[dataset]['num_spks']
        self.enhanceModelType = enhanceModelType
        self.device = device
        self.log_path = log_path

        #from visdom import Visdom
        self.tr_avg_loss = torch.Tensor(config['trainer_en']['epochs'])
        self.val_avg_loss = torch.Tensor(config['trainer_en']['epochs'])
        self.save_folder = config['trainer_en']['save_folder']
        self.model_path = config['trainer_en']['model_path']
        Path(self.save_folder).mkdir(exist_ok=True, parents=True)
        # self.num_params = sum(
            # [param.nelement() for param in nnet.parameters()]) / 10.0**6
        self.model_load = config['trainer_en']['model_load']
        self._reset()
        self.audiowritter = SummaryWriter(self.log_path+'_audio')
        self.writter = MyWriter(self.config, self.log_path)
        self.clip_norm = config['trainer_en']['clipping']
        
    def _reset(self):
        # model load, tr&val loss, optimizer, 시작 epoch 추가

        if self.config['trainer_en']['model_load'][0]:
            print("Loading checkpoint model %s" % self.config['trainer_en']['model_load'][1])
            package = torch.load(self.config['trainer_en']['model_load'][1],map_location= "cuda:"+str(self.device))
            self.model.load_state_dict(package['model_state_dict'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.start_epoch = int(package.get('epoch',1))
            self.tr_avg_loss[:self.start_epoch] = package['tr_avg_loss'][:self.start_epoch]
            self.val_avg_loss[:self.start_epoch] = package['val_avg_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0

        self.prev_val_loss = float("inf")
        self.best_val_loss_epoch = float("inf")
        self.val_no_impv = 0
        self.halving = False

  
    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.config['trainer_en']['epochs'])):
            print('Enhance Training Start...')
            start = time.time()
            self.model.train()
            tr_avg_loss_epoch = self._run_one_epoch(epoch,training=True)
            
            print('-'* 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(epoch+1, time.time()-start,tr_avg_loss_epoch))
            print('_'* 85)
            # writter.add_scalar('data/Train_Loss', tr_avg_loss_epoch, epoch)
            self.writter.log_value(tr_avg_loss_epoch,epoch,'data/Train_Loss')

            #save model per 10 epochs
            if self.config['trainer_en']['check_point'][0]:
                if epoch % self.config['trainer_en']['check_point'][1] == 0 :
                    file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch+1))
                    state_dict = {
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch' : epoch+1,
                        'tr_avg_loss' : self.tr_avg_loss,
                        'val_avg_loss' : self.val_avg_loss
                    }
                    torch.save(state_dict, file_path)
                    print('Saving checkpoint model to %s' % file_path)
            print('validation...')
            self.model.eval()
            
            val_avg_loss_epoch = self._run_one_epoch(epoch,training=False)
            print('-'* 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s |'
                  'Valid Loss {2:.3f}'.format(epoch+1, time.time()-start,val_avg_loss_epoch))
            print('_'* 85)
            # writter.add_scalar('data/Validation_Loss', val_avg_loss_epoch, epoch)
            self.writter.log_value(val_avg_loss_epoch,epoch,'data/Validation_Loss')

            # scheduler
            # if self.config['trainer']['half_lr']:
            #     if val_avg_loss_epoch >= self.prev_val_loss:
            #         self.val_no_impv += 1
            #         if self.val_no_impv >= 3:
            #             self.halving = True
            #         assert(self.val_no_impv <10 and not(self.config['trainer']['early_stop'])), "No improvement for 10 epochs, ealry stopping"
            # if self.halving:
            #     optim_state = self.optimizer.state_dict()
            #     optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2.0
            #     self.optimizer.load_state_dict(optim_state)
            #     print('Learning rate adjusted to :{lr:.6f}'.format(lr=optim_stae['param_groups'][0]['lr']))
            #     self.halving = False
            # self.prev_val_loss = val_avg_loss_epoch

            # save best model
            self.tr_avg_loss[epoch] = tr_avg_loss_epoch
            self.val_avg_loss[epoch] = val_avg_loss_epoch
            if val_avg_loss_epoch < self.best_val_loss_epoch:
                self.best_val_loss_epoch = val_avg_loss_epoch
                file_path = os.path.join(self.save_folder, self.model_path)
                state_dict = {
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch' : epoch+1,
                        'tr_avg_loss' : self.tr_avg_loss,
                        'val_avg_loss' : self.val_avg_loss
                    }
                torch.save(state_dict, file_path)
                print("Find better validated model, saving to %s" % file_path)

            self.scheduler.step(val_avg_loss_epoch)
        # writter.close()
    
    def _run_one_epoch(self, epoch, training= True):
        start = time.time()
        total_loss = 0
        loss = 0
        data_loader = self.tr_loader if training else self.dt_loader
        
        for idx, (data) in enumerate(data_loader):
            
            """
            MISO2 or MISO3 Train

            Input : 
                    mix_stft      :  [B, Mic, T, F]
                    ref_stft : list, [2], ref
            Output :
                    s1_MISO1, s2_MISO2 :  [B, Mic, T, F]
                       s1_bf, s2_bf    :  [B, Mic, T, F]
                        Enhance_out    :  [B, Sources, T, F] 
            
            """

            mix_stft, ref_stft, MISO1_stft, Beamform_stft = data
            mix_stft = mix_stft.cuda(self.device)
        
            
            ref1 = torch.unsqueeze(ref_stft[0][:,0,:,:], dim=1).cuda(self.device)
            ref2 = torch.unsqueeze(ref_stft[1][:,0,:,:], dim=1).cuda(self.device)
            Reference_sources = torch.cat((ref1.detach().cpu(), ref2.detach().cpu()), dim=1)
            
            s1_bf = Beamform_stft[0].cuda(self.device)
            s2_bf = Beamform_stft[1].cuda(self.device)
            Beamform_sources = torch.cat((s1_bf.detach().cpu(), s2_bf.detach().cpu()), dim=1)

            MISO1_spk1= torch.unsqueeze(MISO1_stft[0][:,0,:,:],dim=1).cuda(self.device)
            MISO1_spk2= torch.unsqueeze(MISO1_stft[1][:,0,:,:],dim=1).cuda(self.device)
            MISO1_sources = torch.cat((MISO1_spk1.detach().cpu(), MISO1_spk2.detach().cpu()), dim=1)


            if self.enhanceModelType == 'MISO3':
                ''' MISO3 Enhancement train '''
                if not training:
                    with torch.no_grad():
                        estimate_sources_MISO3_s1 = self.model(mix_stft, s1_bf, MISO1_spk1)
                else:
                    estimate_sources_MISO3_s1 = self.model(mix_stft, s1_bf, MISO1_spk1)
                    
                loss_s1 = loss_Enhance(estimate_sources_MISO3_s1, ref1)
                if training:
                    self.optimizer.zero_grad()
                    loss_s1.backward()
                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['trainer_sp']['max_norm'])
                    self.optimizer.step()
                total_loss += loss_s1.item()/2


                if not training:
                    with torch.no_grad():
                        estimate_sources_MISO3_s2 = self.model(mix_stft, s2_bf, MISO1_spk2)
                else:
                    estimate_sources_MISO3_s2 = self.model(mix_stft, s1_bf, MISO1_spk2)
                    
                loss_s2 = loss_Enhance(estimate_sources_MISO3_s2, ref2)
                if training:
                    self.optimizer.zero_grad()
                    loss_s2.backward()
                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['trainer_sp']['max_norm'])
                    self.optimizer.step()
                total_loss += loss_s2.item()/2

            else:
                ''' MISO2 Enhancement train '''
                if not training:
                    with torch.no_grad():
                        estimate_sources_MISO2 = self.model(mix_stft, s1_bf, s2_bf, MISO1_spk1, MISO1_spk2)
                else:
                    estimate_sources_MISO2 = self.model(mix_stft, s1_bf, s2_bf, MISO1_spk1, MISO1_spk2)

                loss = loss_uPIT(self.num_spks, estimate_sources_MISO2, torch.cat((ref1,ref2), dim=1))
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['trainer_sp']['max_norm'])
                    self.optimizer.step()
                total_loss += loss.item()/2


            if not training and idx == 0:
                if self.enhanceModelType == 'MISO3':
                    Enhance_out = torch.cat((estimate_sources_MISO3_s1.detach().cpu(), estimate_sources_MISO3_s2.detach().cpu()), dim=1) # B Ch T F
                else:
                    Enhance_out = estimate_sources_MISO2

                # [B,1,T,F] -> [T,F]
                batch_idx = random.randint(0,mix_stft.shape[0]-1)
                mix_test = np.transpose(mix_stft[batch_idx,0,:,:].cpu().detach().numpy(),[1,0])
                ref_test = [ np.transpose(ref_sig.numpy(),[1,0]) for ref_sig in Reference_sources[batch_idx,...]]
                MISO1_test = [ np.transpose(MISO1_sig.numpy(),[1,0]) for MISO1_sig in MISO1_sources[batch_idx,...]]
                Beamform_test = [np.transpose(Beamform_sig.numpy(), [1,0]) for Beamform_sig in Beamform_sources[batch_idx,...]]
                Enhance_test = [ np.transpose(estim_sig.numpy(),[1,0]) for estim_sig in Enhance_out[batch_idx,:,:,:]]
                

                self.writter.log_spec(mix_test,'mix',epoch+1)
                for spk_idx in range(self.num_spks):
                    self.writter.log_spec(ref_test[spk_idx],'clean_'+str(spk_idx+1),epoch+1)
                    self.writter.log_spec(MISO1_test[spk_idx],'MISO1_'+ str(spk_idx+1), epoch+1)
                    self.writter.log_spec(Beamform_test[spk_idx],'Beamform_'+str(spk_idx+1), epoch+1)
                    if self.enhanceModelType == 'MISO3':
                        self.writter.log_spec(Enhance_test[spk_idx],'MISO3_'+str(spk_idx+1),epoch+1)
                    else:
                        self.writter.log_spec(Enhance_test[spk_idx],'MISO2_'+str(spk_idx+1),epoch+1)
                    
                mix, clean, separate, beamform, enhance  = self.writter.log_audio_v2(self.num_spks, mix_test, ref_test, MISO1_test, Beamform_test, Enhance_test, epoch+1)
                

                self.audiowritter.add_audio('mix', mix/max(abs(mix)), epoch+1, self.config[self.dataset]['fs'])
                for spk_idx in range(self.num_spks):
                    self.audiowritter.add_audio('clean'+str(spk_idx+1), clean[spk_idx]/max(abs(clean[spk_idx])), epoch+1, self.config[self.dataset]['fs'])
                    self.audiowritter.add_audio('MISO1_'+str(spk_idx+1), separate[spk_idx]/max(abs(separate[spk_idx])), epoch+1, self.config[self.dataset]['fs'])
                    self.audiowritter.add_audio('Beamform_'+str(spk_idx+1), beamform[spk_idx]/max(abs(beamform[spk_idx])), epoch+1, self.config[self.dataset]['fs'])
                    if self.enhanceModelType == 'MISO3':
                        self.audiowritter.add_audio('MISO3_'+str(spk_idx+1), enhance[spk_idx]/max(abs(enhance[spk_idx])), epoch+1, self.config[self.dataset]['fs'])
                    else:
                        self.audiowritter.add_audio('MISO2_'+str(spk_idx+1), enhance[spk_idx]/max(abs(enhance[spk_idx])), epoch+1, self.config[self.dataset]['fs'])

            if training :    
                if idx % self.config['trainer_en']['print_freq'] == 0:
                    print('[Train] Epoch {0} | Iter {1} | Average Loss {2:.3f} |'
                            'Current Loss {3:6f} | {4:.1f} ms/batch'.format(
                                epoch + 1, idx+1, total_loss / (idx+1), 
                                loss_s1.item()/2 + loss_s2.item()/2, 1000*(time.time()-start)/(idx+1)),
                                flush = True)
            else:
                if idx % self.config['trainer_en']['print_freq'] == 0:
                    print('[Evaluation] Epoch {0} | Iter {1} | Average Loss {2:.3f} |'
                            'Current Loss {3:6f} | {4:.1f} ms/batch'.format(
                                epoch + 1, idx+1, total_loss / (idx+1), 
                                loss_s1.item()/2 + loss_s2.item()/2, 1000*(time.time()-start)/(idx+1)),
                                flush = True)

        return total_loss /(idx+1)
=======
            if self.check_MISO1:

                ''' Check the result of MISO1 '''
                hann_win = scipy.signal.get_window('hann', 256)
                scale = np.sqrt(1.0 / hann_win.sum()**2)
                import torchaudio
                import soundfile as sf
                mix_stft = mix_stft * scale
                t_sig = self.ISTFT(mix_stft[0,:,:,:].swapaxes(0,1))
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * 2**15
                t_sig = t_sig.astype(np.int16)
                sf.write('mix.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')


                s1_beamout = s1_beamout * scale
                t_sig = self.ISTFT(s1_beamout[0,...])
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * 2**15
                t_sig = t_sig.astype(np.int16)
                sf.write('e1.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')
                
                s2_beamout = s2_beamout * scale
                t_sig = self.ISTFT(s2_beamout[0,...])
                MAX_INT16 = np.iinfo(np.int16).max
                t_sig=  t_sig * 2**15
                t_sig = t_sig.astype(np.int16)
                sf.write('e2.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')
                
                # b_idx = 0
                # mix_test = mix_stft[b_idx,:,0,:]
                # ref_test = [ np.transpose(ref_sig[b_idx,0,:,:].cpu().detach().numpy(),[1,0]) for ref_sig in ref_stft_1ch]
                # estimate_test = [ np.transpose(estim_sig.cpu().detach().numpy(),[1,0]) for estim_sig in estimate_sources[b_idx,:,:,:]]
                # estimate_test[0] = s1_beamout[b_idx,:,:]
                # estimate_test[1] = s2_beamout[b_idx,:,:]
                # mix, clean, estim = self.writter.log_audio(self.num_spks,mix_test,ref_test,estimate_test,epoch+1)
                # import soundfile as sf
                # sf.write('clean1.nwav',clean[0],8000,'PCM_24')
                # sf.write('clean2.wav',clean[1],8000,'PCM_24')
                # sf.write('mix.wav',mix,8000,'PCM_24')
                # sf.write('estim1.wav',estim[0],8000,'PCM_24')
                # sf.write('estim2.wav',estim[1],8000,'PCM_24')
                
                # estimate_test = [ np.transpose(estim_sig.cpu().detach().numpy(),[1,0]) for estim_sig in estimate_sources[b_idx,:,:,:]]
                # mix, clean, estim = self.writter.log_audio(self.num_spks,mix_test,ref_test,estimate_test,epoch+1)
                # sf.write('MISO1_1.wav',estim[0],8000,'PCM_24')
                # sf.write('MISO1_2.wav',estim[1],8000,'PCM_24')

            s1_beamout = torch.permute(torch.from_numpy(s1_beamout), [0,2,1])
            s2_beamout = torch.permute(torch.from_numpy(s2_beamout), [0,2,1])

            ''' Save Beamforming output '''
            for fileIdx in range(0,len(BeamOutDir)):
                fileRoot = BeamOutDir[fileIdx]
                if fileIdx == 0:
                    ParentRoot = Path(BeamOutDir[fileIdx]).parent.absolute()
                    ParentRoot.mkdir(exist_ok = True, parents = True)
                s_dict = {'S1_BF': s1_beamout[fileIdx,...], 'S2_BF': s2_beamout[fileIdx,...]}
                with open(str(fileRoot), 'wb') as f:
                    pickle.dump(s_dict, f)
            f.close()

            pdb.set_trace()

        return 

    def ISTFT(self,FT_sig): 

        '''
        input : [F,T]
        output : [T,C]
        '''
        # if FT_sig.shape[1] != self.config['ISTFT']['length']+1:
            # FT_sig = np.transpose(FT_sig,(0,1)) # [C,T,F] -> [C,F,T]

        fs = self.config['ISTFT']['fs']; window = self.config['ISTFT']['window']; nperseg=self.config['ISTFT']['length']; noverlap=self.config['ISTFT']['overlap']
        _, t_sig = signal.istft(FT_sig,fs=fs, window=window, nperseg=nperseg, noverlap=noverlap) #[C,F,T] -> [T,C]

        return t_sig


class Trainer_Enhance(object):
    def __init__(self,dataset,enhanceModelType, num_spks, tr_loader,dt_loader,model_sep, model, optimizer,scheduler,config,device,log_path):
        self.num_spks = num_spks
        self.tr_loader = tr_loader
        self.dt_loader = dt_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model_sep = model_sep
        self.model = model
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("model parameters :{}".format(params))
        self.dataset = dataset
        self.Spks = config[dataset]['num_spks']
        self.enhanceModelType = enhanceModelType
        self.device = device
        self.log_path = log_path
        #MISO1 separation model load
        self.MISO1_path = config['trainer_en']['MISO1_path']
        self._load()

        #from visdom import Visdom
        self.tr_avg_loss = torch.Tensor(config['trainer_en']['epochs'])
        self.val_avg_loss = torch.Tensor(config['trainer_en']['epochs'])
        self.save_folder = config['trainer_en']['save_folder']
        self.model_path = config['trainer_en']['model_path']
        Path(self.save_folder).mkdir(exist_ok=True, parents=True)
        # self.num_params = sum(
            # [param.nelement() for param in nnet.parameters()]) / 10.0**6
        self.model_load = config['trainer_en']['model_load']
        self._reset()
        self.audiowritter = SummaryWriter(self.log_path+'_audio')
        self.writter = MyWriter(self.config, self.log_path)
        self.clip_norm = config['trainer_en']['clipping']
        self.check_MISO1 = config['check_MISO1']
    def _load(self):
        ''' Load pretrained MISO1 '''
        print('Loading MISO_1 model %s'% self.MISO1_path)
        package = torch.load(self.MISO1_path, map_location="cuda:"+str(self.device))
        self.model_sep.load_state_dict(package['model_state_dict'])

    def _reset(self):
        # model load, tr&val loss, optimizer, 시작 epoch 추가

        if self.config['trainer_en']['model_load'][0]:
            print("Loading checkpoint model %s" % self.config['trainer_en']['model_load'][1])
            package = torch.load(self.config['trainer_en']['model_load'][1],map_location= "cuda:"+str(self.device))
            self.model.load_state_dict(package['model_state_dict'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.start_epoch = int(package.get('epoch',1))
            self.tr_avg_loss[:self.start_epoch] = package['tr_avg_loss'][:self.start_epoch]
            self.val_avg_loss[:self.start_epoch] = package['val_avg_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0

        self.prev_val_loss = float("inf")
        self.best_val_loss_epoch = float("inf")
        self.val_no_impv = 0
        self.halving = False

    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.config['trainer_en']['epochs'])):
            print('Training...')
            start = time.time()
            self.model.train()
            tr_avg_loss_epoch = self._run_one_epoch(epoch,training=True)
            
            print('-'* 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(epoch+1, time.time()-start,tr_avg_loss_epoch))
            print('_'* 85)
            # writter.add_scalar('data/Train_Loss', tr_avg_loss_epoch, epoch)
            self.writter.log_value(tr_avg_loss_epoch,epoch,'data/Train_Loss')

            #save model per 10 epochs
            if self.config['trainer_en']['check_point'][0]:
                if epoch % self.config['trainer_en']['check_point'][1] == 0 :
                    file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch+1))
                    state_dict = {
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch' : epoch+1,
                        'tr_avg_loss' : self.tr_avg_loss,
                        'val_avg_loss' : self.val_avg_loss
                    }
                    torch.save(state_dict, file_path)
                    print('Saving checkpoint model to %s' % file_path)
            print('validation...')
            self.model.eval()
            val_avg_loss_epoch = self._run_one_epoch(epoch,training=False)
            print('-'* 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s |'
                  'Valid Loss {2:.3f}'.format(epoch+1, time.time()-start,val_avg_loss_epoch))
            print('_'* 85)
            # writter.add_scalar('data/Validation_Loss', val_avg_loss_epoch, epoch)
            self.writter.log_value(val_avg_loss_epoch,epoch,'data/Validation_Loss')

            # scheduler
            # if self.config['trainer']['half_lr']:
            #     if val_avg_loss_epoch >= self.prev_val_loss:
            #         self.val_no_impv += 1
            #         if self.val_no_impv >= 3:
            #             self.halving = True
            #         assert(self.val_no_impv <10 and not(self.config['trainer']['early_stop'])), "No improvement for 10 epochs, ealry stopping"
            # if self.halving:
            #     optim_state = self.optimizer.state_dict()
            #     optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2.0
            #     self.optimizer.load_state_dict(optim_state)
            #     print('Learning rate adjusted to :{lr:.6f}'.format(lr=optim_stae['param_groups'][0]['lr']))
            #     self.halving = False
            # self.prev_val_loss = val_avg_loss_epoch

            # save best model
            self.tr_avg_loss[epoch] = tr_avg_loss_epoch
            self.val_avg_loss[epoch] = val_avg_loss_epoch
            if val_avg_loss_epoch < self.best_val_loss_epoch:
                self.best_val_loss_epoch = val_avg_loss_epoch
                file_path = os.path.join(self.save_folder, self.model_path)
                state_dict = {
                        'model_state_dict' : self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch' : epoch+1,
                        'tr_avg_loss' : self.tr_avg_loss,
                        'val_avg_loss' : self.val_avg_loss
                    }
                torch.save(state_dict, file_path)
                print("Find better validated model, saving to %s" % file_path)

            self.scheduler.step(val_avg_loss_epoch)
        # writter.close()
    
    def _run_one_epoch(self, epoch, training= True):
        self.model_sep.eval()

        start = time.time()
        total_loss = 0
        loss = 0
        data_loader = self.tr_loader if training else self.dt_loader

        for idx, (data) in enumerate(data_loader):
            
            """
            Input : [B,Mic,T,F]
            """
            mix_stft, ref_stft, s1_bf, s2_bf = data
            pdb.set_trace()
            mix_stft = mix_stft.cuda(self.device)
            s1_bf = s1_bf.cuda(self.device)
            s2_bf = s2_bf.cuda(self.device)
        
            # reference mic 1 : Train
            ref_stft_1ch = [ [] for _ in range(self.num_spks)]
            for spk_idx in range(self.num_spks):
                #[B,1,T,F]
                ref_stft_1ch[spk_idx] = ref_stft[spk_idx][:,0,:,:].cuda(self.device)
                ref_stft_1ch[spk_idx] = torch.unsqueeze(ref_stft_1ch[spk_idx], dim=1)

            for spk_idx in range(self.num_spks):
                ref_stft[spk_idx] = ref_stft[spk_idx].cuda(self.device)
            B, Mic, T, F = mix_stft.size()

            ''' Source Alignment across Microphones'''
            s1_ch6 = torch.empty(B,Mic,T,F, dtype=torch.complex64)
            s2_ch6 = torch.empty(B,Mic,T,F, dtype=torch.complex64)
            ref1 = torch.empty(B,1,T,F, dtype=torch.complex64)
            ref2 = torch.empty(B,1,T,F, dtype=torch.complex64)

            with torch.set_grad_enabled(False):
                ref_sources = self.model_sep(mix_stft) #[B,Spk,T,F]
    
            #######################################
            ''' Source-wise Alignment ''' 
            ## Source wise alignment 기능 정상 작동
            #######################################
            s1_ch6[:,0,:,:] = ref_sources[:,0,:,:] 
            s2_ch6[:,0,:,:] = ref_sources[:,1,:,:]
            
            s_ref_sources = torch.unsqueeze(ref_sources,dim=2) #[B,Spks,1,T,F]
            ref_magnitude = torch.abs(torch.sqrt(s_ref_sources.real**2 + s_ref_sources.imag**2)) #[B,Spks,1,T,F]
            with torch.set_grad_enabled(False):
                for ref_mic in range(1,Mic):
                    # Select the reference microphone by circular shifting the microphone
                    # [Yq, ... , Yp, ... , Yq-1]
                    
                    s_mix_stft = torch.roll(mix_stft,-ref_mic, dims=1)
                    estimate_sources = self.model_sep(s_mix_stft) #[B,Spks,T,F]
                    # calculate magnitude distance between ref mic and non ref mic
                    s_estimate_sources = torch.unsqueeze(estimate_sources,dim=1) #[B,1,Spks,T,F]
                    magnitude_distance = torch.sum(torch.abs(ref_magnitude - abs(s_estimate_sources)),[3,4]) #[B,Spks,Spks,T,F]
                    
                    perms = estimate_sources.new_tensor(list(permutations(range(self.Spks))), dtype=torch.long) #[[0,1],[1,0]]
                    index = torch.unsqueeze(perms, dim=2)
                    perms_one_hot = estimate_sources.new_zeros((*perms.size(), self.Spks), dtype=torch.float).scatter_(2,index,1)

                    batchwise_distance = torch.einsum('bij,pij->bp',[magnitude_distance, perms_one_hot])
                    min_distance_idx = torch.argmin(batchwise_distance,dim=1)

                    for ii in range(B):
                        if min_distance_idx[ii] == 1:
                            s1_ch6[ii,ref_mic,:,:] = estimate_sources[ii,1,:,:]
                            s2_ch6[ii,ref_mic,:,:] = estimate_sources[ii,0,:,:]
                        else:
                            s1_ch6[ii,ref_mic,:,:] = estimate_sources[ii,0,:,:]
                            s2_ch6[ii,ref_mic,:,:] = estimate_sources[ii,1,:,:]

                # calculate magnitude distance between ref mic and target signal

                for spk_idx in range(self.num_spks):
                    if spk_idx == 0 :
                        ref_ = ref_stft_1ch[spk_idx]
                    else:
                        ref_ = torch.cat((ref_,ref_stft_1ch[spk_idx]), dim=1)

                s_ref = torch.unsqueeze(ref_, dim=1)
                magnitude_distance = torch.sum(torch.abs(ref_magnitude - abs(s_ref)),[3,4])
                perms = ref_.new_tensor(list(permutations(range(self.Spks))), dtype=torch.long) #[[0,1],[1,0]]
                index = torch.unsqueeze(perms, dim=2)
                perms_one_hot = ref_.new_zeros((*perms.size(), self.Spks), dtype=torch.float).scatter_(2,index,1)
                batchwise_distance = torch.einsum('bij,pij->bp',[magnitude_distance, perms_one_hot])
                min_distance_idx = torch.argmin(batchwise_distance, dim=1)

                for ii in range(B):
                    if min_distance_idx[ii] == 1:
                        ref1[ii,...] = ref_[ii,1,:,:]
                        ref2[ii,...] = ref_[ii,0,:,:]
                    else:
                        ref1[ii,...] = ref_[ii,0,:,:]
                        ref2[ii,...] = ref_[ii,1,:,:]
                ref1 = ref1.cuda(self.device)
                ref2 = ref2.cuda(self.device)
            s1_ch6 = s1_ch6.cuda(self.device) # B, Ch, T, F
            s2_ch6 = s2_ch6.cuda(self.device)


            # ###########################################################################################
            # Temp code
            # mix_stft, ref_stft, s1_bf, s2_bf, s1_ch6, s2_ch6 = data
            # mix_stft = mix_stft.cuda(self.device)
            # s1_bf = s1_bf.cuda(self.device)
            # s2_bf = s2_bf.cuda(self.device)
            # s1_ch6 = s1_ch6.cuda(self.device)
            # s2_ch6 = s2_ch6.cuda(self.device)
            # ###########################################################################################

            if self.enhanceModelType == 'MISO3':
                ''' MISO3 Enhancement train '''
                estimate_sources_MISO3_s1 = self.model(mix_stft, torch.unsqueeze(s1_bf, dim=1), torch.unsqueeze(s1_ch6[:,0,...],dim=1))
                loss_s1 = loss_Enhance(estimate_sources_MISO3_s1, ref1)
                if training:
                    self.optimizer.zero_grad()
                    loss_s1.backward()
                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['trainer_sp']['max_norm'])
                    self.optimizer.step()
                total_loss += loss_s1.item()/2

                estimate_sources_MISO3_s2 = self.model(mix_stft, torch.unsqueeze(s2_beamout, dim=1), torch.unsqueeze(s2_ch6[:,0,...], dim=1))
                loss_s2 = loss_Enhance(estimate_sources_MISO3_s2, ref2)
                if training:
                    self.optimizer.zero_grad()
                    loss_s2.backward()
                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['trainer_sp']['max_norm'])
                    self.optimizer.step()
                total_loss += loss_s2.item()/2
            else:
                ''' MISO2 Enhancement train '''
                estimate_sources_MISO2 = self.model(mix_stft, torch.unsqueeze(s1_bf, dim=1), torch.unsqueeze(s2_beamout, dim=1), torch.unsqueeze(s1_ch6[:,0,...],dim=1), torch.unsqueeze(s2_ch6[:,0,...], dim=1))
                loss = loss_uPIT(self.num_spks, estimate_sources_MISO2, ref_stft_1ch)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['trainer_sp']['max_norm'])
                    self.optimizer.step()
                total_loss += loss.item()/2

            if not training and idx == 0:
                if self.enhanceModelType == 'MISO3':
                    Enhance_out = torch.cat((estimate_sources_MISO3_s1, estimate_sources_MISO3_s2), dim=1) # B Ch T F
                else:
                    Enhance_out = estimate_sources_MISO2

                # [B,1,T,F] -> [T,F]
                mix_test = np.transpose(mix_stft[0,0,:,:].cpu().detach().numpy(),[1,0])
                ref_test = [ np.transpose(ref_sig[0,0,:,:].cpu().detach().numpy(),[1,0]) for ref_sig in ref_stft_1ch]
                estimate_test = [ np.transpose(estim_sig.cpu().detach().numpy(),[1,0]) for estim_sig in Enhance_out[0,:,:,:]]

                self.writter.log_spec(mix_test,'mix',epoch+1)
                for spk_idx in range(self.num_spks):
                    self.writter.log_spec(ref_test[spk_idx],'clean'+str(spk_idx+1),epoch+1)
                    # self.writter.log_spec(ref2_test,'clean2',epoch+1)
                    self.writter.log_spec(estimate_test[spk_idx],'enhance'+str(spk_idx+1),epoch+1)
                    # self.writter.log_spec(estim2_test,'estim2',epoch+1) 
                mix, clean, estim = self.writter.log_audio(self.num_spks,mix_test,ref_test,estimate_test,epoch+1)

                self.audiowritter.add_audio('mix', mix/max(abs(mix)), epoch+1, self.config[self.dataset]['fs'])
                for spk_idx in range(self.num_spks):
                    self.audiowritter.add_audio('clean'+str(spk_idx+1), clean[spk_idx]/max(abs(clean[spk_idx])), epoch+1, self.config[self.dataset]['fs'])
                    self.audiowritter.add_audio('estim'+str(spk_idx+1), estim[spk_idx]/max(abs(estim[spk_idx])), epoch+1, self.config[self.dataset]['fs'])
            
            if idx % self.config['trainer_en']['print_freq'] == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} |'
                        'Current Loss {3:6f} | {4:.1f} ms/batch'.format(
                            epoch + 1, idx+1, total_loss / (idx+1), 
                            loss_s1.item()/2 + loss_s2.item()/2, 1000*(time.time()-start)/(idx+1)),
                            flush = True)

        return total_loss /(idx+1)

    def ISTFT(self,FT_sig,index): 

        '''
        input : [F,T]
        output : [T,C]
        '''
        # if FT_sig.shape[1] != self.config['ISTFT']['length']+1:
            # FT_sig = np.transpose(FT_sig,(0,1)) # [C,T,F] -> [C,F,T]

        fs = self.config['ISTFT']['fs']; window = self.config['ISTFT']['window']; nperseg=self.config['ISTFT']['length']; noverlap=self.config['ISTFT']['overlap']
        _, t_sig = signal.istft(FT_sig,fs=fs, window=window, nperseg=nperseg, noverlap=noverlap) #[C,F,T] -> [T,C]
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d

    def ISTFT(self,FT_sig): 

<<<<<<< HEAD
        '''
        input : [F,T]
        output : [T,C]
        '''
        # if FT_sig.shape[1] != self.config['ISTFT']['length']+1:
            # FT_sig = np.transpose(FT_sig,(0,1)) # [C,T,F] -> [C,F,T]

        fs = self.config['ISTFT']['fs']; window = self.config['ISTFT']['window']; nperseg=self.config['ISTFT']['length']; noverlap=self.config['ISTFT']['overlap']
        _, t_sig = signal.istft(FT_sig,fs=fs, window=window, nperseg=nperseg, noverlap=noverlap) #[C,F,T] -> [T,C]
=======
        MAX_INT16 = np.iinfo(np.int16).max
        t_sig=  t_sig * 2**15
        t_sig = t_sig.astype(np.int16)
        sf.write('sample'+index+'.wav',t_sig.T, self.config['ISTFT']['fs'],'PCM_24')

        return t_sig
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d

        return t_sig