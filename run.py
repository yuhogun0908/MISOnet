import sys
sys.path.append('./')
import argparse
# from model import MYNET, ResidualBlock, Share_BN_ResidualBlock
import yaml

# from trainer import ModelTrainer, ModelTester
from torch.utils.data import DataLoader
import trainer
# from utils.setup import setup_solver
# from utils.loss import create_criterion
import pdb
import os
import pickle
from model import MISO_1
import torch
from trainer import Trainer
# Blind Source Separation by using NN
# 1. Feature Extractor
# 2. Build dataloader 
# 3. Train
# 4. Test
#
# Reference
# Z.Q, Wang. "Multi-microphone Complex Spectral Mapping for Utterance-wise and Continuous Speech Separation", IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 29, 2021
def run(args,config):
    if args.mode == 'Extraction':
        if args.dataset == 'REVERB_2MIX':
            from dataloader.REVERB_2MIX import main_reverb
            fs = config['fs']; chunk_time = config['chunk_time']; least_time = config['least_time']
            num_spks = config['REVERB_2MIX']['num_spks']; scp_list = config['REVERB_2MIX']['scp_list']; 
            # select the type of data to save                    
            if config['REVERB_2MIX']['select_mode'] == 1:  # save train 
                tr_wave_list = config['REVERB_2MIX']['tr_parent_wave_list']; save_tr_pickle_dir = config['REVERB_2MIX']['saved_tr_pickle_dir']
                main_reverb('Train',fs, chunk_time, least_time,num_spks,scp_list, tr_wave_list,save_tr_pickle_dir)  
            elif config['REVERB_2MIX']['select_mode'] == 2: # save development
                dt_wave_list = config['REVERB_2MIX']['dt_parent_wave_list']; save_dt_pickle_dir = config['REVERB_2MIX']['saved_dt_pickle_dir']
                main_reverb('Development',fs, chunk_time, least_time,num_spks,scp_list, dt_wave_list,save_dt_pickle_dir)  
            elif config['REVERB_2MIX']['select_mode'] == 3: # save test 
                ###matlab 데이터부터 만들고 작성###
                pdb.set_trace()
    if args.mode == 'Train':
        if args.dataset == 'REVERB_2MIX':
            from dataloader.data import AudioDataset
            tr_pickle_dir = config['REVERB_2MIX']['saved_tr_pickle_dir']
            dt_pickle_dir = config['REVERB_2MIX']['saved_dt_pickle_dir']
            tr_dataset = AudioDataset('Train',tr_pickle_dir,**config['STFT'])
            tr_loader = DataLoader(tr_dataset, **config['dataloader']['Train'])
            # dt 데이터 만들고 주석 풀기
            dt_dataset = AudioDataset('Development',dt_pickle_dir,**config['STFT'])
            dt_loader = DataLoader(dt_dataset,**config['dataloader']['Development'])
        # models
        if args.use_cuda:
            # model_sep = torch.nn.DataParallel(model_sep)
            
            model_sep = MISO_1(**config['MISO_1']).cuda(config['gpu_num'])
            print(model_sep)
        else:
            model_sep = MISO_1(**config['MISO_1'])
            print(model_sep)

        # optimizer
        if config['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(model_sep.parameters(),
                                         lr = config['optimizer']['lr'],
                                         weight_decay = config['optimizer']['weight_decay'])

        #trainer
        trainer = Trainer(tr_loader, dt_loader,model_sep,optimizer,config,config['gpu_num'])
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type = str, default = '.', help = 'Root directory')
    parser.add_argument('-c', '--config', type = str, help = 'Path to option TAML file.')
    parser.add_argument('-d', '--dataset', type = str, help = 'Dataset')
    parser.add_argument('-m', '--mode', type = str, help= 'Extract or Train or Test')
    parser.add_argument('-u', '--use_cuda', type = int, default=1, help='Whether use GPU')
    # Train argument
    

    args = parser.parse_args()

    with open(os.path.join(args.config, args.dataset + '.yml'), mode = 'r') as f:
        config = yaml.load(f,Loader = yaml.FullLoader)

    run(args,config)
    
    # elif args.mode == 'Train':
    #     train()
    # else:
    #     test()
        