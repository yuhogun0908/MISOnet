#################################################################################################################################################
class DenseNet(nn.Module):
    def __init__(self,depth,num_classes,growh_rate=12,reduction=0.5,bottleneck=True,dropRate=0.0):
        super(DenseNet,self).__init__()
        num_of_blocks = 3
        in_planes = 16 # 2 * growh_rate
        n = (depth - num_of_blocks - 1)/num_of_blocks # 총 depth에서 첫 conv , 2개의 transit , 마지막 linear 빼고 / num_of_blocks
        if reduction != 1 :
            in_planes = 2 * growh_rate
        if bottleneck == True:
            in_planes = 2 * growh_rate #논문에서 Bottleneck + Compression 할 경우 first layer은 2*growh_rate라고 했다.
            n = n/2 # conv 1x1 레이어가 추가되니까 !
            block = BottleneckBlock 
        else :
            block = BasicBlock
        
        n = int(n) #n = DenseBlock에서 block layer 개수를 의미한다.
        self.conv1 = nn.Conv2d(3,in_planes,kernel_size=3,stride=1,padding=1,bias=False) # input:RGB -> output:growhR*2
        
        
        #1st block
        # nb_layers,in_planes,growh_rate,block,dropRate
        self.block1 = DenseBlock(n,in_planes,growh_rate,block,dropRate)
        in_planes = int(in_planes+n*growh_rate) # 입력 + 레이어 만큼의 growh_rate
        
        # in_planes,out_planes,dropRate
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)),dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        
        
        #2nd block
        # nb_layers,in_planes,growh_rate,block,dropRate
        self.block2 = DenseBlock(n,in_planes,growh_rate,block,dropRate)
        in_planes = int(in_planes+n*growh_rate) # 입력 + 레이어 만큼의 growh_rate
        
        # in_planes,out_planes,dropRate
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)),dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        
        
        #3rd block
        # nb_layers,in_planes,growh_rate,block,dropRate
        self.block3 = DenseBlock(n,in_planes,growh_rate,block,dropRate)
        in_planes = int(in_planes+n*growh_rate) # 입력 + 레이어 만큼의 growh_rate
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        
        self.fc = nn.Linear(in_planes,num_classes) # 마지막에 ave_pool 후에 1x1 size의 결과만 남음.
        
        self.in_planes = in_planes
        
        # module 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv layer들은 필터에서 나오는 분산 root(2/n)로 normalize 함
                # mean = 0 , 분산 = sqrt(2/n) // 이게 무슨 초기화 방법이었는지 기억이 안난다.
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d): # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1) # 
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):# linear layer 초기화.
                m.bias.data.zero_()
        
    def forward(self,x):
        #x : 32*32
        out = self.conv1(x) # 32*32
        out = self.block1(out) # 32*32
        out = self.trans1(out) # 16*16
        out = self.block2(out) # 16*16
        out = self.trans2(out) # 8*8
        out = self.block3(out) # 8*8
        out = self.relu(self.bn1(out)) #8*8
        out = F.avg_pool2d(out,8) #1*1
        out = out.view(-1, self.in_planes) #channel수만 남기 때문에 Linear -> in_planes
        return self.fc(out)




class BasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,dropRate = 0.0):
        #input dimsnsion을 정하고, output dimension을 정하고(growh_rate임), dropRate를 정함.
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace = True) # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride = 1, padding = 1, bias = False)
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate>0:
            out = F.dropout (out,p=self.droprate,training = self.training)
        return torch.cat([x,out],1)
        
class BottleneckBlock(nn.Module):
    def __init__(self,in_planes,out_planes,dropRate=0.0):
        #out_planes => growh_rate를 입력으로 받게 된다.
        super(BottleneckBlock,self).__init__()
        inter_planes = out_planes * 4 # bottleneck layer의 conv 1x1 filter chennel 수는 4*growh_rate이다.
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_planes,inter_planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training = self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training = self.training)
        return torch.cat([x,out],1) # 입력으로 받은 x와 새로 만든 output을 합쳐서 내보낸다
        
class DenseBlock(nn.Module):
    def __init__(self,nb_layers,in_planes,growh_rate,block,dropRate=0.0):
        super(DenseBlock,self).__init__()
        self.layer = self._make_layer(block, in_planes, growh_rate, nb_layers, dropRate)
    
    def _make_layer(self,block,in_planes,growh_rate,nb_layers,dropRate):
        layers=[]
        for i in range(nb_layers):
            layers.append(block(in_planes + i*growh_rate ,growh_rate,dropRate))
            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        return self.layer(x)

class TransitionBlock(nn.Module):
    def __init__(self,in_planes,out_planes,dropRate=0.0):
        super(TransitionBlock,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training=self.training)
        return F.avg_pool2d(out,2)