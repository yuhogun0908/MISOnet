import torch
#import criterion import cal_loss


class Trainer(object):
    def __init__(self,tr_loader,dt_loader,optimizer,config):
        self.tr_loader = tr_loader
        self.dt_loader = dt_loader
        self.optimizer = optimizer
        self.config = config



    def train(self):
        for epoch in range(self.config['trainer']['epochs']):
            print('Training...')
            self.model.train()

