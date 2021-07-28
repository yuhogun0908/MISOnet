import torch
import torch.nn as nn
import torch.nn.functional as f


class MISO_1(nn.Module):
    def __init__(self, )
    super(MISO_1,self).__init__()

    #init#
    # multi-channel separation network (MISO_1)
    self.encoder_1 = Encoder_1()
    self.TCN = TemporalConvNet()
    self.decoder_1 = Decoder_1()


    def forward(self,mixture):





#MISO_1 Encoder
class Encoder_1(nn.Module):
    def __init__(self, ):



    def forward(self,mixture):


class TemporalConvNet(nn.Module):
    def __init__(self, ):

    def forward(self, ):


class Decoder_1(nn.Module):
    def __init__(self, ):

    def forward(self, ):
        