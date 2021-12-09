import numpy as np
import torch
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

try : 
    from utils.plotting import spec2plot,MFCC2plot
except ImportError:
    from utils.plotting import spec2plot,MFCC2plot
from scipy import signal
import pdb

# https://pytorch.org/docs/stable/tensorboard.html

class MyWriter(SummaryWriter):
    def __init__(self,config, logdir):
        super(MyWriter, self).__init__(logdir)
        self.config = config
        # self.hp = hp
        # self.window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                            #    dtype=None, layout=torch.strided, device=None,
                            #    requires_grad=False)
    def log_value(self, train_loss, step,tag):
        self.add_scalar(tag, train_loss, step)

    def log_train(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_test(self,test_loss,step) : 
        self.add_scalar('test_loss', test_loss, step)

    def log_audio(self,num_spks,mix,ref,estim,step) : 
        mix = self.ISTFT(mix,'mix')
        clean_ISTFT = [[] for _ in range(num_spks)]
        estim_ISTFT = [[] for _ in range(num_spks)]
        for spk_idx in range(num_spks):
            clean_ISTFT[spk_idx]= self.ISTFT(ref[spk_idx],'clean'+str(spk_idx+1))
            estim_ISTFT[spk_idx]= self.ISTFT(estim[spk_idx],'estim'+str(spk_idx+1))


        # clean1=  self.ISTFT(clean1,'clean1')
        # clean2 = self.ISTFT(clean2,'clean2')
        # estim1 = self.ISTFT(estim1,'estim1')
        # estim2 = self.ISTFT(estim2,'estim2')
        # self.add_audio('mix', mix, step, self.config['fs'])
        # for spk_idx in range(num_spks):
            # self.add_audio('clean'+str(spk_idx+1), clean_ISTFT[spk_idx], step, self.config['fs'])
            # self.add_audio('estim'+str(spk_idx+1), estim_ISTFT[spk_idx], step, self.config['fs'])
            # self.add_audio('clean1', clean1, step, self.config['fs'])
            # self.add_audio('clean2', clean2, step, self.config['fs'])
            # self.add_audio('estim1', estim1, step, self.config['fs'])
            # self.add_audio('estim2', estim2, step, self.config['fs'])
        return mix, clean_ISTFT, estim_ISTFT
<<<<<<< HEAD
    def log_audio_v2(self, num_spks, mix, ref, Separate, Beamform, Enhance, step):
        mix = self.ISTFT(mix,'mix')
        clean_ISTFT = [[] for _ in range(num_spks)]
        Separate_ISTFT = [[] for _ in range(num_spks)]
        Beamform_ISTFT = [[] for _ in range(num_spks)]
        Enhance_ISTFT = [[] for _ in range(num_spks)]

        for spk_idx in range(num_spks):
            clean_ISTFT[spk_idx]= self.ISTFT(ref[spk_idx],'clean'+str(spk_idx+1))
            Separate_ISTFT[spk_idx]= self.ISTFT(Separate[spk_idx],'estim'+str(spk_idx+1))
            Beamform_ISTFT[spk_idx]= self.ISTFT(Beamform[spk_idx],'estim'+str(spk_idx+1))
            Enhance_ISTFT[spk_idx]= self.ISTFT(Enhance[spk_idx],'estim'+str(spk_idx+1))

        return mix, clean_ISTFT, Separate_ISTFT, Beamform_ISTFT, Enhance_ISTFT
=======
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d

    def log_MFCC(self,input,output,clean,step):
        input = input.to('cpu')
        output = output.to('cpu')
        clean= clean.to('cpu')

        noisy = input[0]
        estim = input[1]

        noisy = noisy.detach().numpy()
        estim = estim.detach().numpy()
        output = output.detach().numpy()
        clean= clean.detach().numpy()

        output = np.expand_dims(output,0)
        clean = np.expand_dims(clean,0)

        noisy = MFCC2plot(noisy)
        estim = MFCC2plot(estim)
        output = MFCC2plot(output)
        clean = MFCC2plot(clean)

        self.add_image('noisy',noisy,step,dataformats='HWC')
        self.add_image('estim',estim,step,dataformats='HWC')
        self.add_image('clean',clean,step,dataformats='HWC')
        self.add_image('output',output,step,dataformats='HWC')

        #self.add_image('noisy',noisy,step)
        #self.add_image('estim',estim,step)
        #self.add_image('output',output,step)

    # add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    def log_spec(self,data,label,step) :
        self.add_image(label,
            spec2plot(data), step, dataformats='HWC')
 
    # def log_wav2spec(self,noisy,estim,clean,step) :
    #     noisy = torch.from_numpy(noisy)
    #     estim = torch.from_numpy(estim)
    #     clean = torch.from_numpy(clean)

    #     noisy = torch.stft(noisy,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)
    #     estim = torch.stft(estim,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)
    #     clean = torch.stft(clean,n_fft=self.hp.audio.frame, hop_length = self.hp.audio.shift, window = self.window, center = True, normalized=False, onesided=True)

    #     self.log_spec(noisy,estim,clean,step)

    def ISTFT(self,FT_sig,index): 
        '''
        input : [F,T]
        output : [T,C]
        '''
        # if FT_sig.shape[1] != self.config['ISTFT']['length']+1:
            # FT_sig = np.transpose(FT_sig,(0,1)) # [C,T,F] -> [C,F,T]

        fs = self.config['ISTFT']['fs']; window = self.config['ISTFT']['window']; nperseg=self.config['ISTFT']['length']; noverlap=self.config['ISTFT']['overlap']
        _, t_sig = signal.istft(FT_sig,fs=fs, window=window, nperseg=nperseg, noverlap=noverlap) #[C,F,T] -> [T,C]


        # MAX_INT16 = np.iinfo(np.int16).max
        # t_sig=  t_sig * 2**15
        # t_sig = t_sig.astype(np.int16)
        # wf.write('sample'+index+'.wav',self.config['ISTFT']['fs'],t_sig.T)

        return t_sig

# def check_MFCC():
#     from hparams import HParam
#     ## log MFCC test
#     hp = HParam("../../config/TEST.yaml")
#     log_dir = '/home/nas/user/kbh/MCFE/'+'/'+'log'+'/'+'TEST'

#     writer = MyWriter(hp, log_dir)

#     input = torch.load('input.pt').to('cpu')
#     clean = torch.load('clean.pt').to('cpu')
#     output = torch.load('output.pt').to('cpu')

#     print('input : ' + str(input.shape))
#     print('output : ' + str(output.shape))

#     noisy = input[0]
#     estim = input[1]

#     noisy = noisy.detach().numpy()
#     estim = estim.detach().numpy()
#     output = output.detach().numpy()
#     clean= clean.detach().numpy()


#     output = np.expand_dims(output,0)
#     clean = np.expand_dims(clean,0)

#     print(noisy.shape)
#     print(estim.shape)
#     print(output.shape)
#     print(clean.shape)

#     noisy = MFCC2plot(noisy)
#     estim = MFCC2plot(estim)
#     output = MFCC2plot(output)
#     clean = MFCC2plot(clean)

#     print('MFCC')
#     print(noisy.shape)
#     print(estim.shape)
#     print(output.shape)
#     print(clean.shape)

#     writer.log_MFCC(noisy,estim,output,clean,0)

# if __name__=='__main__':
#     check_MFCC()