import scipy.io.wavfile as wav
import logging
import numpy as np
import math
import xml.etree.ElementTree as ET
import pickle
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import glob
import sys
import getopt
import sys
#sys.path.append('..')
import glob
from torch.utils.data.dataset import Dataset
from pathlib import Path
import pickle
import pdb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import scipy.io.wavfile as wav
import torchaudio.transforms as transforms
import torchaudio
import matplotlib.pyplot as plt
import random
sys.path.append('..')


def downsample_to(waveform, sample_rate, downsample_rate):
    waveform = waveform.T
    downsample_resample = torchaudio.transforms.Resample(
        sample_rate, downsample_rate, resampling_method='sinc_interpolation')
    down_sampled = downsample_resample(waveform)

    return down_sampled.T

def feature_target_extraction(file_path, target_mode, tau, v, segLen, wavList, informList, feature_onoff, spec_augmentation):
        ## extracting feature and target
        ## target_mode 1: (S S), 2: (Q Q), 3: (Q S), 4: (S Q)
        ## tau, v, segLen : Please read the readme.txt
        ## feature_onoff 0: no extracting feature 1: extracting feature
        #pdb.set_trace()
    try:
        
        #pdb.set_trace()
        """
        filePath = Path(file_path)
        wavList = []
        
        labelList = []
        for n in [x for x in filePath.iterdir() if x.is_dir()]:
            wavList.append(list(sorted(filePath.glob('**/*.wav'))))
            labelList.append(list(sorted(filePath.glob('**/*.xml'))))
        #for i in range(len(wavList[:][-1])):
        """
        temp_x = torch.zeros([99, 80, 2])
        for wavfile, xmlfile in zip(wavList, informList):
        
            
            #(fs, sig) = wav.read(wavfile)
            sig, fs = torchaudio.load(wavfile)
            sig = sig.T
            ####################################
            

            if fs != 16000:
                #pdb.set_trace()
                down_fs = 16000
                downsampled_sig = downsample_to(sig,fs,down_fs)
                sig = downsampled_sig
                fs = down_fs
                
            if len(sig[0]) == 1:
                
                temp = torch.zeros(len(sig),2)
                temp[:,0] = sig[:,0]
                temp[:,1] = temp[:,0]
                
                sig = temp
            
            
        
            frame_length = 0.025
            frame_stride = 0.0125
            input_stride = int(round(fs*frame_stride))
            input_nfft = int(round(fs*frame_length))
            wavlen = len(sig)/fs
            
            
            inform = ET.parse(xmlfile)
            root = inform.getroot()
            eval_target_tmp = np.zeros([np.shape(sig)[0], 3])
            cont_target_tmp = np.zeros([np.shape(sig)[0], 3])
            #(960604,2) len(wav)
            eval_target = np.zeros([math.floor(wavlen)*2-1, 3])
            #(119,2)
            cont_target = np.zeros([int((math.floor(wavlen)-1) / 0.1), 3])
            #(590,2)
            disc_target = np.zeros([int((math.floor(wavlen)-1) / 0.1), 3])
            #(590,2)
            eval_wavinf = []
            train_wavinf = []
            
            xmlfile = xmlfile.replace("\\", "/")
            #'./data/A/00001.xml'
            fname = xmlfile.split('/')[-1].split('.')[0]
            #'00001'
            if spec_augmentation: 
                alpa = xmlfile.split('/')[2]
                wavfile = wavfile.replace(alpa,alpa+'_1')
                fold = xmlfile.split(fname)[0].replace(alpa, alpa+'_1')
                #'./data/A_1/'
            else:
                fold = xmlfile.split(fname)[0]
            
            print(np.shape(sig), wavfile, xmlfile)   
            
            #pdb.set_trace()
            #'./dat
           
        
            for i in range(0, len(root[0])):
                classInf = root[0][i][1].text
                #root[][][] : ClassID
                LB = float(root[0][i][3].text)
                #START
                LE = float(root[0][i][4].text)
                #END
                if LE - LB <= 2 * tau:
                    t = (LE - LB)/4
                else:
                    t = tau
                VBQ = max(0, (LE + LB - (LE - LB) * (1 / math.sqrt(1 - v))) / 2)
                VEQ = min(np.shape(sig)[0], (LE + LB + (LE - LB) * (1 / math.sqrt(1 - v))) / 2)
                VBS = max(0, LB - t)
                VES = min(np.shape(sig)[0], LE + t)
                tmp = np.linspace(VBQ, VEQ, int(VEQ * fs) - int(VBQ * fs + 1) + 1)
                if classInf == '2':  # CC
                    # evaluation target
                    eval_target_tmp[int(LB * fs + 1):int(LE * fs + 1), 0] = 1
                    # sigmoid target
                    if target_mode == 1 or target_mode == 4:
                        
                        cont_target_tmp[int(VBS * fs + 1):int((LB + LE)/2 * fs + 1), 0] = 1/(1 + np.exp(- 4 / t * (np.linspace(VBS, (LB + LE)/2, int((LB + LE)/2 * fs)-int(VBS * fs + 1) + 1) - LB)))
                        cont_target_tmp[int((LB + LE)/2 * fs + 1):int(VES * fs + 1), 0] = 1/(1 + np.exp(4 / t * (np.linspace((LB + LE)/2, VES, int(VES * fs)-int((LB + LE)/2 * fs + 1) + 1) - LE)))

                    # Quadratic target
                    else:
                        cont_target_tmp[int(VBQ * fs + 1):int(VEQ * fs + 1), 0] = 1 - (1 - v) * ((2 * tmp - LB - LE) / (LE - LB)) * ((2 * tmp - LB - LE) / (LE - LB))

                if classInf == '3':  # TS
                    # evaluation target
                    eval_target_tmp[int(LB * fs + 1):int(LE * fs + 1), 1] = 1
                    # sigmoid target
                    if target_mode == 1 or target_mode == 3:
                        cont_target_tmp[int(VBS * fs + 1):int((LB + LE)/2 * fs + 1), 1] = 1/(1 + np.exp(- 4 / t * (np.linspace(VBS, (LB + LE)/2, int((LB + LE)/2 * fs)-int(VBS * fs + 1) + 1) - LB)))
                        cont_target_tmp[int((LB + LE)/2 * fs + 1):int(VES * fs + 1), 1] = 1/(1 + np.exp(4 / t * (np.linspace((LB + LE)/2, VES, int(VES * fs)-int((LB + LE)/2 * fs + 1) + 1) - LE)))

                    # Quadratic target
                    else:
                        cont_target_tmp[int(VBQ * fs + 1):int(VEQ * fs + 1), 1] = 1 - (1 - v) * ((2 * tmp - LB - LE) / (LE - LB)) * ((2 * tmp - LB - LE) / (LE - LB))

                if classInf == '4': #SC
                    eval_target_tmp[int(LB * fs + 1):int(LE * fs + 1), 1] = 1
                    # sigmoid target
                    if target_mode == 1 or target_mode == 3:
                        cont_target_tmp[int(VBS * fs + 1):int((LB + LE)/2 * fs + 1), 2] = 1/(1 + np.exp(- 4 / t * (np.linspace(VBS, (LB + LE)/2, int((LB + LE)/2 * fs)-int(VBS * fs + 1) + 1) - LB)))
                        cont_target_tmp[int((LB + LE)/2 * fs + 1):int(VES * fs + 1), 2] = 1/(1 + np.exp(4 / t * (np.linspace((LB + LE)/2, VES, int(VES * fs)-int((LB + LE)/2 * fs + 1) + 1) - LE)))

                    # Quadratic target
                    else:
                        cont_target_tmp[int(VBQ * fs + 1):int(VEQ * fs + 1), 2] = 1 - (1 - v) * ((2 * tmp - LB - LE) / (LE - LB)) * ((2 * tmp - LB - LE) / (LE - LB))
            
            #pdb.set_trace()
            for i in range(0, len(eval_target)):
               
                eval_target[i] = eval_target_tmp[int((i/2 + 0.5 * segLen) * fs)]
                eval_wavinf.append('%sfeature/evaluation/%s_%04d.pickle' % (fold, fname, i))
                # feature
                if feature_onoff == 1:
                    t = i
                    seg_start = int((t / 2) * fs)
                    seg_end = int((t / 2 + 1) * fs - 1)
                    #seg_end-seg_start = 15999
                    
                    spectrogram = transforms.MelSpectrogram(sample_rate= fs ,n_fft = 512, hop_length = 162, n_mels = 80)
                    mel = transforms.AmplitudeToDB()  
                
                    temp_x[:,:,0]= (spectrogram(sig[seg_start:seg_end,0])).permute(1,0)
                    temp_x[:,:,1] = (spectrogram(sig[seg_start:seg_end,1])).permute(1,0)

                    if spec_augmentation:
                        temp_x[:,:,0] = time_mask(freq_mask(temp_x[:,:,0], num_masks = 2, replace_with_zero = True), num_masks = 2 , replace_with_zero = True)
                        temp_x[:,:,1] = time_mask(freq_mask(temp_x[:,:,1], num_masks = 2, replace_with_zero = True), num_masks = 2 , replace_with_zero = True)    
                        #tensor_to_img(temp_x[:,:,0])
                    #print("Shape of spectrogram : {}".format(temp_x.size()))
                    
                    #temp_x[:, :, 0] = logfbank(sig[seg_start:seg_end, 0], fs, winlen=0.025, winstep=0.01, nfilt=80, nfft=512,preemph=0.97)
                    #temp_x[:, :, 1] = logfbank(sig[seg_start:seg_end, 1], fs, winlen=0.025, winstep=0.01, nfilt=80, nfft=512,
                                                #preemph=0.97)
                    
                    with open('%sfeature/evaluation/%s_%04d.pickle' % (fold, fname, i), 'wb') as f:
                        pickle.dump(temp_x, f)

            # confidence target, each frame size = 25ms, overlap = 10ms
            #pdb.set_trace()
            for i in range(0, len(cont_target)): #0 ~ 590
                cont_target[i] = cont_target_tmp[int((i + 0.5 * segLen / 0.1) * fs * 0.1)]
                disc_target[i] = eval_target_tmp[int((i + 0.5 * segLen / 0.1) * fs * 0.1)]
               
                train_wavinf.append('%sfeature/train/%s_%d.pickle' % (fold, fname, i))
                # feature
                if feature_onoff == 1:
                    t = i
                    seg_start = int(t * fs * 0.1)
                    seg_end = int((t + segLen / 0.1) * fs * 0.1 - 1)

                    spectrogram = transforms.MelSpectrogram(sample_rate= fs, n_fft = 512, hop_length = 162, n_mels = 80)
                    mel = transforms.AmplitudeToDB()

                    temp_x[:,:,0]= (spectrogram(sig[seg_start:seg_end,0])).permute(1,0)
                    temp_x[:,:,1] = (spectrogram(sig[seg_start:seg_end,1])).permute(1,0)
                    
                    if spec_augmentation:
                        temp_x[:,:,0] = time_mask(freq_mask(temp_x[:,:,0], num_masks = 2, replace_with_zero = True), num_masks = 2 , replace_with_zero = True)
                        temp_x[:,:,1] = time_mask(freq_mask(temp_x[:,:,1], num_masks = 2, replace_with_zero = True), num_masks = 2 , replace_with_zero = True)
                    #temp_x[:, :, 0] = logfbank(sig[seg_start:seg_end, 0], fs, winlen=0.025, winstep=0.01, nfilt=80, nfft=512, preemph=0.97)
                    #temp_x[:, :, 1] = logfbank(sig[seg_start:seg_end, 1], fs, winlen=0.025, winstep=0.01, nfilt=80, nfft=512, preemph=0.97)
                    #(99,80,2)
                    
                    with open('%sfeature/train/%s_%d.pickle' % (fold, fname, i), 'wb') as f:
                        pickle.dump(temp_x, f)
                        #각 wav파일마다 590개 feature 
            
            cont_target[cont_target >= 0.98] = 1
            cont_target[cont_target <= 0.02] = 0
            # save
            ##.set_trace()
            with open('%starget/evaluation/%s.pickle' % (fold, fname), 'wb') as f:
                pickle.dump(eval_target, f) # (119,2)
                pickle.dump(eval_wavinf, f) # 119

            with open('%starget/confidence/%s.pickle' % (fold, fname), 'wb') as f:
                pickle.dump(cont_target, f) # (590, 2)
                pickle.dump(train_wavinf, f) #1개의 wav를 590 feauture로 뽑은 590개 pickle의 제목이 list형태로 저장
            with open('%starget/discrete/%s.pickle' % (fold, fname), 'wb') as f:
                pickle.dump(disc_target, f)  #(590, 2)
                pickle.dump(train_wavinf, f) # 590

        print("Feature and target extraction is over.")
    except:
        print("Feature and target extraction is fail. Please check the wav file.")



def dataload(file_path,data_partition, test_set, C_B):

    path = sorted([x for x in file_path.iterdir() if x.is_dir()])
    
    foldA = str(path[0]); foldA_1 = str(path[1]); foldB = str(path[2]); foldB_1 = str(path[3]); foldC = str(path[4]); foldC_1 = str(path[5]); foldD = str(path[6]); foldD_1=  str(path[7]);foldE = str(path[8]); foldE_1=  str(path[9])
    #pdb.set_trace()

    t_ListA = sorted(glob.glob("%s/target/evaluation/*.pickle" % foldA))
    if C_B == 'C':
        t_ListB = sorted(glob.glob("%s/target/confidence/*.pickle" % foldB))
        t_ListB_1 = sorted(glob.glob("%s/target/confidence/*.pickle" % foldB_1))
        t_ListC = sorted(glob.glob("%s/target/confidence/*.pickle" % foldC))
        t_ListC_1 = sorted(glob.glob("%s/target/confidence/*.pickle" % foldC_1))
        t_ListD = sorted(glob.glob("%s/target/confidence/*.pickle" % foldD))
        t_ListD_1 = sorted(glob.glob("%s/target/confidence/*.pickle" % foldD_1))
        t_ListE = sorted(glob.glob("%s/target/confidence/*.pickle" %foldE))
        t_ListE_1 = sorted(glob.glob("%s/target/confidence/*.pickle" %foldE_1))
    elif C_B == 'B':
        t_ListB = sorted(glob.glob("%s/target/discrete/*.pickle" % foldB))
        t_ListB_1 = sorted(glob.glob("%s/target/discrete/*.pickle" % foldB_1))
        t_ListC = sorted(glob.glob("%s/target/discrete/*.pickle" % foldC))
        t_ListC_1 = sorted(glob.glob("%s/target/discrete/*.pickle" % foldC_1))
        t_ListD = sorted(glob.glob("%s/target/discrete/*.pickle" % foldD))
        t_ListD_1 = sorted(glob.glob("%s/target/discrete/*.pickle" % foldD_1))
        t_ListE = sorted(glob.glob("%s/target/discrete/*.pickle" % foldE))
        t_ListE_1 = sorted(glob.glob("%s/target/discrete/*.pickle" % foldE_1))
    
    if (test_set == 'A'):
        t_testList = t_ListA
        t_trainList = t_ListB + t_ListC + t_ListD + t_ListE
        #t_trainList = t_ListB_1 + t_ListC_1 + t_ListD_1 + t_ListE_1 
    elif (test_set == 'B'):
        t_testList = t_ListB
        t_trainList = t_ListA + t_ListC + t_ListD
    elif (test_set == 'C'):
        t_testList = t_ListC
        t_trainList = t_ListA + t_ListB + t_ListD
    elif (test_set == 'D'):
        t_testList = t_ListD
        t_trainList = t_ListA + t_ListB + t_ListC

    label = [] ; data =[]
    if data_partition == 'train':
        label_CCTS = []
        data_CCTS = []
        data_Noise = []
        length_Noise = 0
        for x in t_trainList:
            with open(x, 'rb') as f:
                label.extend(pickle.load(f))
                data.extend(pickle.load(f))
    else:
        for x in t_testList:
            with open(x, 'rb') as f:
                label.extend(pickle.load(f))
                data.extend(pickle.load(f))
    #pdb.set_trace()
    return label, data


def randomshuffle(length_CCTS, length_Noise):
    perm = np.arange(length_CCTS*2)
    np.random.shuffle(perm)
    perm_Noise = np.arange(length_Noise)
    np.random.shuffle(perm_Noise)

    return perm, perm_Noise

class MIVIA(Dataset):

    def __init__(self, file_path,  data_partition,  test_set , C_B):
        super(MIVIA, self).__init__()
        self.file_path = Path(file_path)
        self.data_partition = data_partition
        if data_partition == 'train':
            (self.label, self.data) = dataload(self.file_path,self.data_partition, test_set, C_B )
            self.label_CCTS = []
            self.data_CCTS = []
            self.data_Noise = []
            perm = 0
            perm_Noise =0
            length_Noise = 0
            for x in range(len(self.label)):
                if self.label[x][0] != 0 or self.label[x][1] != 0:
                    self.label_CCTS.append(self.label[x])
                    self.data_CCTS.append(self.data[x])
                    #pdb.set_trace()
                    #print(self.label_CCTS)
                    #print(self.data_CCTS)
                else:
                    length_Noise += 1
                    self.data_Noise.append(self.data[x])
            self.label_CCTS = np.array(self.label_CCTS)
            self.data_CCTS  = np.array(self.data_CCTS)
            self.data_Noise = np.array(self.data_Noise)
            #(16292,)
            perm, perm_Noise = randomshuffle(len(self.label_CCTS), length_Noise)
            
            self.temp_data_Noise = self.data_Noise[perm_Noise]
            
            self.temp_label = np.append(np.zeros([len(self.label_CCTS),2]), self.label_CCTS, 0)
            self.temp_data  = np.append(self.temp_data_Noise[0:len(self.label_CCTS)], self.data_CCTS, 0)
           
            self.data = self.temp_data[perm]
            #(20236)
            self.label  = self.temp_label[perm]
            #(20236,2)
            #pdb.set_trace()
            
        elif self.data_partition == 'valid':
            (self.label, self.data) = dataload(self.file_path,self.data_partition, test_set, C_B)
        
        else:
            self.label, self.data = dataload(self.file_path,self.data_partition, test_set, C_B)


    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        
        #print(self.data[idx], self.label[idx])
        with open(self.data[idx], 'rb') as f:  #나중에 pickle 뽑을때 ../~.pickle로 저장하기
           
            data = pickle.load(f)

        torch.Tensor(self.label[idx])
        torch.Tensor(data)
        #print(data.shape, self.label[idx].shape)
        return data, self.label[idx]

  




