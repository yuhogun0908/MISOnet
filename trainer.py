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
import scipy 
from pathlib import Path
import pickle
from tqdm import tqdm
import random

class Trainer_Separate(object):

    def __init__(self,dataset, num_spks, ref_ch, tr_loader,dt_loader,model, optimizer,scheduler,config,device,cuda_flag,log_path):
        self.dataset = dataset
        self.num_spks = num_spks
        self.ref_ch = ref_ch
        self.tr_loader = tr_loader
        self.dt_loader = dt_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model = model
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("model parameters :{}".format(params))
        self.device = device
        self.cuda_flag = cuda_flag
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
            print('Separate Training Start ...')
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
            mix_stft = torch.roll(mix_stft,-self.ref_ch, dims=1)
            if self.cuda_flag:
                mix_stft = mix_stft.cuda(self.device)
            B, Mic, T, F = mix_stft.size()
            estimate_sources = self.model(mix_stft) #[B,Spk,T,F]

            # reference mic 1 : Train
            ref_stft_1ch = [ [] for _ in range(self.num_spks)]
            for spk_idx in range(self.num_spks):
                #[B,1,T,F]
                ref_stft_1ch[spk_idx] = ref_stft[spk_idx][:,self.ref_Ch,:,:]
                if self.cuda_flag:
                    ref_stft_1ch[spk_idx] = ref_stft_1ch[spk_idx].cuda(self.device)

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
                if self.cuda_flag:
                    mix_test = np.transpose(mix_stft[0,0,:,:].cpu().detach().numpy(),[1,0])
                    ref_test = [ np.transpose(ref_sig[0,0,:,:].cpu().detach().numpy(),[1,0]) for ref_sig in ref_stft_1ch]
                    estimate_test = [ np.transpose(estim_sig.cpu().detach().numpy(),[1,0]) for estim_sig in estimate_sources[0,:,:,:]]
                else:
                    mix_test = np.transpose(mix_stft[0,0,:,:].numpy(),[1,0])
                    ref_test = [ np.transpose(ref_sig[0,0,:,:].numpy(),[1,0]) for ref_sig in ref_stft_1ch]
                    estimate_test = [ np.transpose(estim_sig.numpy(),[1,0]) for estim_sig in estimate_sources[0,:,:,:]]
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

class Trainer_Enhance(object):

    def __init__(self,dataset,enhanceModelType, num_spks, ref_ch, tr_loader,dt_loader, model, optimizer,scheduler,config,device,cuda_flag, log_path):
        self.dataset = dataset
        self.num_spks = num_spks
        self.ref_ch = ref_ch
        self.tr_loader = tr_loader
        self.dt_loader = dt_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.model = model
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("model parameters :{}".format(params))
        self.Spks = config[dataset]['num_spks']
        self.enhanceModelType = enhanceModelType
        self.device = device
        self.cuda_flag = cuda_flag
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
            # Train
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
            # Validation
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

            mix_stft, ref_stft_1ch, MISO1_stft, Beamform_stft = data
            
            for spk_idx in range(self.num_spks):
                if len(ref_stft_1ch[spk_idx].shape) == 3:
                    ref_stft_1ch[spk_idx] = torch.unsqueeze(ref_stft_1ch[spk_idx], dim=1) 
            ref1 = ref_stft_1ch[0]
            ref2 = ref_stft_1ch[1]

            if self.cuda_flag:
                mix_stft = mix_stft.cuda(self.device)
                ref1 = ref1.cuda(self.device)
                ref2 = ref2.cuda(self.device)

                        
            s1_bf = Beamform_stft[0].cuda(self.device)
            s2_bf = Beamform_stft[1].cuda(self.device)
            MISO1_spk1= torch.unsqueeze(MISO1_stft[0][:,0,:,:],dim=1).cuda(self.device)
            MISO1_spk2= torch.unsqueeze(MISO1_stft[1][:,0,:,:],dim=1).cuda(self.device)


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
                    if self.cuda_flag:
                        Enhance_out = torch.cat((estimate_sources_MISO3_s1.detach().cpu(), estimate_sources_MISO3_s2.detach().cpu()), dim=1) # B Ch T F
                    else:
                        Enhance_out = torch.cat((estimate_sources_MISO3_s1, estimate_sources_MISO3_s2), dim=1) 
                else:
                    if self.cuda_flag:
                        Enhance_out = estimate_sources_MISO2.detach().cpu()
                    else:
                        Enhance_out = estimate_sources_MISO2

                batch_idx = random.randint(0,mix_stft.shape[0]-1)    
                if self.cuda_flag:
                    Reference_sources = torch.cat((ref1.detach().cpu(), ref2.detach().cpu()), dim=1)
                    mix_test = np.transpose(mix_stft[batch_idx,0,:,:].cpu().detach().numpy(),[1,0])
                    Beamform_sources = torch.cat((s1_bf.detach().cpu(), s2_bf.detach().cpu()), dim=1)
                    MISO1_sources = torch.cat((MISO1_spk1.detach().cpu(), MISO1_spk2.detach().cpu()), dim=1)
                else:
                    Reference_sources = torch.cat((ref1, ref2), dim=1)
                    mix_test = np.transpose(mix_stft[batch_idx,0,:,:].numpy(),[1,0])
                    Beamform_sources = torch.cat((s1_bf, s2_bf), dim=1)
                    MISO1_sources = torch.cat((MISO1_spk1, MISO1_spk2), dim=1)

                # [B,1,T,F] -> [T,F]
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