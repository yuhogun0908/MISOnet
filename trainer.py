import torch
import time
from tqdm import tqdm
from criterion import loss_uPIT
from torch.utils.tensorboard import SummaryWriter
import pdb
from pathlib import Path
import os
#import criterion import cal_loss
from utils.writer import MyWriter
import numpy as np
import math
class Trainer(object):
    def __init__(self,tr_loader,dt_loader,model, optimizer,scheduler,config,device,log_path):
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
        self.log_path = log_path
        #from visdom import Visdom
        self.tr_avg_loss = torch.Tensor(config['trainer']['epochs'])
        self.val_avg_loss = torch.Tensor(config['trainer']['epochs'])
        self.save_folder = config['trainer']['save_folder']
        self.model_path = config['trainer']['model_path']
        Path(self.save_folder).mkdir(exist_ok=True, parents=True)
        # self.num_params = sum(
            # [param.nelement() for param in nnet.parameters()]) / 10.0**6
        self.model_load = config['trainer']['model_load']
        self._reset()
        self.audiowritter = SummaryWriter(self.log_path)
        self.writter = MyWriter(self.config, self.log_path)
        self.clip_norm = config['trainer']['clipping']
    def _reset(self):
        # model load, tr&val loss, optimizer, 시작 epoch 추가

        if self.config['trainer']['model_load'][0]:
            print("Loading checkpoint model %s" % self.config['trainer']['model_load'][1])
            package = torch.load(self.config['trainer']['model_load'][1],map_location= "cuda:"+str(self.device))
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
        for epoch in tqdm(range(self.start_epoch, self.config['trainer']['epochs'])):
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
            if self.config['trainer']['check_point'][0]:
                if epoch % self.config['trainer']['check_point'][1] == 0 :
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
            mix_stft, ref1_stft, ref2_stft = data
            mix_stft = mix_stft.cuda(self.device)
            ref1_stft = ref1_stft.cuda(self.device)
            ref2_stft = ref2_stft.cuda(self.device)
            B, Mic, T, F = mix_stft.size()
            
            # for ref_mic in range(Mic):
            #     """
            #     Select the reference microphone by circular shifting the microphone
            #     [Yq,...,YP,Y1,...,Yq-1]
            #     """
            #     mix_stft = torch.roll(mix_stft,-ref_mic,dims=1)
            #     ref1_stft_1ch = ref1_stft[:,ref_mic,:,:] #[B,1,T,F]
            #     ref2_stft_1ch = ref2_stft[:,ref_mic,:,:] #[B,1,T,F]
                
            #     estimate_sources = self.model(mix_stft) #[B,Spk,T,F]
            #     loss = loss_uPIT(estimate_sources, ref1_stft_1ch, ref2_stft_1ch)
            #     loss += loss 

            # reference mic 1 : Train
            ref1_stft_1ch = ref1_stft[:,0,:,:] #[B,1,T,F]
            ref2_stft_1ch = ref2_stft[:,0,:,:] #[B,1,T,F]
            estimate_sources = self.model(mix_stft) #[B,Spk,T,F]
            loss = loss_uPIT(estimate_sources, ref1_stft_1ch, ref2_stft_1ch)
            if math.isnan(loss):
                pdb.set_trace()
                if len(ref1_stft_1ch.shape) == 3:
                    ref1_stft_1ch = torch.unsqueeze(ref1_stft_1ch, dim=1)
                    ref2_stft_1ch = torch.unsqueeze(ref2_stft_1ch, dim=1)
                ref_1ch  = torch.cat((ref1_stft_1ch,ref2_stft_1ch), dim=1)
                B, Spk, T, F = estimate_sources.size()
                Loss1 = 0
                for Spk_idx in range(Spk):
                    L1_real = torch.sum( torch.abs(estimate_sources[:,Spk_idx,:,:].real - ref_1ch[:,Spk_idx,:,:].real), [0,1,2]) 
                    L1_imag = torch.sum( torch.abs(estimate_sources[:,Spk_idx,:,:].imag - ref_1ch[:,Spk_idx,:,:].imag), [0,1,2])
                    L1_magnitude = torch.sum( torch.abs( torch.sqrt(estimate_sources[:,Spk_idx,:,:].real**2 + estimate_sources[:,Spk_idx,:,:].imag**2) \
                                                            - abs(ref_1ch[:,Spk_idx,:,:])),[0,1,2])
                    Loss1 += L1_real + L1_imag + L1_magnitude
                    # Loss2= 0

            if not training and idx == 0:
                # [B,1,T,F] -> [T,F]

                mix_test = np.transpose(mix_stft[0,0,:,:].cpu().detach().numpy(),[1,0])
                ref1_test = np.transpose(ref1_stft_1ch[0,:,:].cpu().detach().numpy(),[1,0])
                ref2_test = np.transpose(ref2_stft_1ch[0,:,:].cpu().detach().numpy(),[1,0])
                estim1_test = np.transpose(estimate_sources[0,0,:,:].cpu().detach().numpy(),[1,0])
                estim2_test = np.transpose(estimate_sources[0,1,:,:].cpu().detach().numpy(),[1,0])
                self.writter.log_spec(mix_test,'mix',epoch+1)
                self.writter.log_spec(ref1_test,'clean1',epoch+1)
                self.writter.log_spec(ref2_test,'clean2',epoch+1)
                self.writter.log_spec(estim1_test,'estim1',epoch+1)
                self.writter.log_spec(estim2_test,'estim2',epoch+1) 
                mix, clean1, clean2, estim1, estim2 = self.writter.log_audio(mix_test,ref1_test,ref2_test,estim1_test,estim2_test,epoch+1)

                self.audiowritter.add_audio('mix', mix/max(abs(mix)), epoch+1, self.config['fs'])
                self.audiowritter.add_audio('clean1', clean1/max(abs(clean1)), epoch+1, self.config['fs'])
                self.audiowritter.add_audio('clean2', clean2/max(abs(clean2)), epoch+1, self.config['fs'])
                self.audiowritter.add_audio('estim1', estim1/max(abs(estim1)), epoch+1, self.config['fs'])
                self.audiowritter.add_audio('estim2', estim2/max(abs(estim1)), epoch+1, self.config['fs'])

            #Source Alignment across Microphones module should be implemented
            # when beamforming is performed after training.
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                #gradient threshold to clip
                if self.clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.config['trainer']['max_norm'])
                self.optimizer.step()

            total_loss += loss.item()
            
            if idx % self.config['trainer']['print_freq'] == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} |'
                        'Current Loss {3:6f} | {4:.1f} ms/batch'.format(
                            epoch + 1, idx+1, total_loss / (idx+1), 
                            loss.item(), 1000*(time.time()-start)/(idx+1)),
                            flush = True)
        return total_loss /(idx+1)

                









