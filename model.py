import torch
import torch.nn as nn
import torch.nn.functional as f
import pdb

EPS = 1e-8

class MISO_1(nn.Module):
    def __init__(self,num_bottleneck, Ch,norm_type):
        super(MISO_1,self).__init__()

    #init#
    # multi-channel separation network (MISO_1)
    
    # encoder
    # ch = 8 -> real + imag = 16

        """
        num_bottleneck : number of bottleneck
        """

        self.num_bottleneck = num_bottleneck
        self.encoders = []
        B_channels = [2*Ch,24,32,32,32,32,64,128,384]
        for n_b in range(num_bottleneck):
            block = self.make_layer(n_b,B_channels[n_b], B_channels[n_b+1])
            self.encoders.append(block)

        self.TCN = TemporalConvNet(2,7,384,384,384,norm_type)


    def make_layer(self,b_idx,in_channels, out_channels):
        layers = []
        if b_idx <= 5:
            if b_idx == 0:
                layers.append(init_Conv2d_(in_channels,out_channels,kernel_size=(3,3), stride=(1,1),padding=(1,0)))
                layers.append(DenseBlock(out_channels,out_channels,out_channels))
            else:
                layers.append(Conv2d_(in_channels,out_channels,kernel_size=(3,3), stride=(1,2),padding=(1,0)))
                layers.append(DenseBlock(out_channels,out_channels,out_channels))
        else:
            layers.append(Conv2d_(in_channels,out_channels,kernel_size=(3,3), stride=(1,2),padding=(1,0)))

        return nn.Sequential(*layers)



        # self.encoder_1 = Encoder_1(Ch) # [B,2C, T, 257] -> [B,384,T,1]
        # self.decoder_1 = Decoder_1()

    def forward(self,mixture):
        real_spec = mixture.real # [B,C,F,T]
        imag_spec = mixture.imag # [B,C,F,T]

        #reference mic -> circular shift 고려해야 됨.

        x = torch.cat((real_spec,imag_spec),dim=1)
        
        '''Encoder 부분 append하는 식으로 해야됨. 이거 수정하기'''
        # en_out = self.encoder_1(Net_input) 
        xs = []
        for i, encoder in enumerate(self.encoders):
            print(i)

            x = encoder(x)
            xs.append(x)
        #Reshape [B,384, T ,1] -> [B,384,T]
        x = torch.squeeze(x)

        pdb.set_trace()
        #[B,384,T] -> [B,384,T]
        tcn_out = self.TCN(x)
        #Reshape
        
        

        pdb.set_trace()
        
        # tcn_out = self.TCN(en_out)
        # de_out = self.decoder_1(tcn_out)


# #MISO_1 Encoder
# class Encoder_1(nn.Module):
#     def __init__(self,B, Ch):
#         """
#         B : number of bottleneck

#         """
#         super(Encoder_1,self).__init__()



#         # self.init_conv2d = nn.Conv2d(2*Ch,24, kernel_size =(3,3),stride=(1,1),padding=(1,0))
#         # self.denseblock_1 = DenseBlock(24,24,24)
#         # self.denseblock_2 = DenseBlock(32,32,32)
#         # self.denseblock_3 = DenseBlock(32,32,32)
#         # self.denseblock_4 = DenseBlock(32,32,32)
#         # self.denseblock_5 = DenseBlock(32,32,32)

#         # self.conv2d_1 = nn.Conv2d(24,32, kernel_size=(3,3), stride=(1,2),padding=(1,0)) 
#         # self.conv2d_2 = nn.Conv2d(32,32, kernel_size=(3,3), stride=(1,2),padding=(1,0))
#         # self.conv2d_3 = nn.Conv2d(32,32, kernel_size=(3,3), stride=(1,2),padding=(1,0))
#         # self.conv2d_4 = nn.Conv2d(32,32, kernel_size=(3,3), stride=(1,2),padding=(1,0))
#         # self.conv2d_5 = nn.Conv2d(32,64, kernel_size=(3,3), stride=(1,2),padding=(1,0))
#         # self.conv2d_6 = nn.Conv2d(64,128, kernel_size=(3,3), stride=(1,2),padding=(1,0))
#         # self.conv2d_7 = nn.Conv2d(128,384, kernel_size=(3,3), stride=(1,2),padding=(1,0))


#         B_channels = [2*Ch,24,32,32,32,32,644,128,384]
#         """
#         b : encoder block index
#         """
#         for b in range(B):
#             block = self.make_layer(b,B_channels[b], B_channels[b+1])
#             self.encoders.append(block)


#     def make_layer(self,b_idx,in_channels, out_channels):
#         layers = []
#         if b_indx <= 5:
#             if b_indx == 0:
#                 layers.append(init_Conv2d_(in_channels,out_channels,kernel_size=(3,3), stride=(1,1),padding=(1,0)))
#                 layers.append(DenseBlock(out_channels,out_channels,out_channels))
#             else:
#                 layers.append(Conv2d_(in_channels,out_channels,kernel_size=(3,3), stride=(1,2),padding=(1,0)))
#                 layers.append(DenseBlock(out_channels,out_channels,out_channels))
#         else:
#             layers.append(Conv2d_(in_channels,out_channels,kernel_size=(3,3), stride=(1,2),padding=(1,0)))

#         return nn.Sequential(*layers)

#     def forward(self,mixture):
#         # y1 = self.init_conv2d(mixture)
#         # y2 = self.denseblock_1(y1)
#         # y3 = self.conv2d_1(y2)

#         # y4 = self.denseblock_2(y3)
#         # y5 = self.conv2d_2(y4)
#         # y6 = self.denseblock_3(y5)
#         # y7 = self.conv2d_3(y6)
#         # y8 = self.denseblock_4(y7)
#         # y9 = self.conv2d_4(y8)
#         # y10 = self.denseblock_5(y9)
#         # y11 = self.conv2d_5(y10)
#         # y12 = self.conv2d_6(y11)
#         # y13 = self.conv2d_7(y12)

#         # return y13
#         xs = []
#         for i, encoder in enumerate(self.encoders):
#             x = encoder(x)
#             xs.append(x)
#         pdb.set_trace()

#         return 

class init_Conv2d_(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=(3,3),stride=(1,1),padding=(1,0)):
        super(init_Conv2d_, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding)
    def forward(self,x):
        return self.conv2d(x)

class Conv2d_(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=(3,3),stride=(1,2),padding=(1,0), norm_type="IN"):
        super(Conv2d_,self).__init__()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elu = nn.ELU()
        norm = nn.InstanceNorm2d(out_channels) # 384

        self.net = nn.Sequential(conv2d,elu,norm)

    def forward(self,x):
        return self.net(x)


class DenseBlock(nn.Module):

    def __init__(self,init_ch, g1, g2):
        super(DenseBlock,self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*init_ch,g1, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3*g1,g1, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4*g1,g1, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1)
        )        
        self.conv4 = nn.Sequential(
            nn.Conv2d(5*g1,g1, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(6*g1,g2, kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ELU(),
            nn.InstanceNorm2d(g2)
        )
    def forward(self,x):
        y0 = torch.cat((x,x),dim=1)
        y1 = self.conv1(y0)
        
        y1_0 = torch.cat((y0,y1),dim=1)
        y2 = self.conv2(y1_0)

        y2_1_0 = torch.cat((y0,y1,y2),dim=1)
        y3 = self.conv3(y2_1_0)

        y3_2_1_0 = torch.cat((y0,y1,y2,y3),dim=1)
        y4 = self.conv4(y3_2_1_0)

        y4_3_2_1_0 = torch.cat((y0,y1,y2,y3,y4),dim=1)
        y5 = self.conv5(y4_3_2_1_0)
        
        return y5



class TemporalConvNet(nn.Module):
    def __init__(self, R, X, C_in, C_hidden, C_out, norm_type = "IN"):
        """
        R : Number of repeats  R = 2
        X : Number of convolutional blocks in each repeat X = 7
        C_in : Number of channels in input
        C_hidden : Number of channels in first conv block output
        C_out : Number of channels in output
        """
        super(TemporalConvNet,self).__init__()
        
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x  # 0,2,4,8,16,32,64
                # kernel(P) 3 stride 1 padding d dilation d featuremap 384
                padding = 2**x
                blocks += [TemporalBlock(C_in,C_hidden,C_out,
                                         kernel_size= 3, stride = 1, padding=padding, dilation=dilation,
                                         norm_type = norm_type)]
            repeats += [nn.Sequential(*blocks)]
        self.temporal_conv_net = nn.Sequential(*repeats)
        
    def forward(self,x):
        """
        Input : [B,C,T] 
        Output : [B,C,T]
        """
        return self.temporal_conv_net(x)

class TemporalBlock(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,kernel_size,
                 stride,padding,dilation,norm_type="IN"):
        """
        in_channels : 384
        out_channels : 384
        kernel_size : 3
        stride : 1
        padding : d
        dilation : d
        featuremap : 384
        """
        super(TemporalBlock,self).__init__()
        norm_1 = chose_norm(norm_type, in_channels) # 384
        elu_1 = nn.ELU()
        # [B,C,T] -> [B,C,T]
        dsconv_1 = DepthwiseSeparableConv(in_channels,hidden_channels,kernel_size,stride,padding,dilation,norm_type="gLN")
        
        norm_2 = chose_norm(norm_type, hidden_channels) # 384
        elu_2 = nn.ELU()
        dsconv_2 = DepthwiseSeparableConv(hidden_channels,out_channels,kernel_size,stride,padding,dilation,norm_type="gLN")
        
        self.net = nn.Sequential(norm_1, elu_1, dsconv_1, norm_2, elu_2, dsconv_2)

    def forward(self,x):
        """
        Input : [B,C,T]
        Output : [B,C,T]
        """
        residual = x
        out = self.net(x)
        return out + residual


class DepthwiseSeparableConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation,norm_type="gLN"):
        super(DepthwiseSeparableConv,self).__init__()
        depthwise_conv = nn.Conv1d(in_channels,in_channels,kernel_size,stride=stride,
                                   padding=padding,dilation=dilation,groups=in_channels,bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type,in_channels)
        pointwise_conv = nn.Conv1d(in_channels,out_channels,1,bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)
    def forward(self,x):
        """
        Input : [B,C_in,T]
        output : [B,C_out,T]
        """
        return self.net(x)


def chose_norm(norm_type, channel_size):
    """
    input : [B, C, T]
    """
    if norm_type=="gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "IN":
        return nn.InstanceNorm1d(channel_size)
    else:
        return nn.BatchNorm1d(channel_size)

class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y



class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


# class Decoder_1(nn.Module):
#     def __init__(self, ):

#     def forward(self, ):


if __name__ == "__main__":
    
    input = torch.randn(10,8,150,257, dtype=torch.cfloat)
    model = MISO_1(8,8,"IN")    
    output = model(input)
    pdb.set_trace()


    # TCN

    input = torch.randn(10,384,150)
    model = TemporalConvNet(2,7,384,384,384,norm_type)
    output = model(input)
