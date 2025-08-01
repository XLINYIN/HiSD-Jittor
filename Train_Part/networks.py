import jittor as jt
from jittor import nn
import numpy as np
import math
def tile_like(x,target):
    #接收channel*1*1的特征图，将其在H*W上扩增，使其可以与target在Channel上相连
    x=x.view(x.size(0),-1,1,1)
    x=x.repeat(1,1,target.size(2),target.size(3))
    return x
#Discriminator
class Dis(nn.Module):
    def __init__(self,hyperparameters):
        super(Dis,self).__init__()
        self.tags=hyperparameters['tags']
        channels=hyperparameters['discriminators']['channels']
        self.conv=nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'],channels[0],1,1,0),#只改变通道数
            *[DownBlock(channels[i],channels[i+1])for i in range(len(channels)-1)],
            nn.AdaptiveAvgPool2d((1,1))#2048*1*1
        )
        self.fcs=nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels[-1]+hyperparameters['style_dim']+self.tags[i]['tag_irrelevant_conditions_dim'],
            len(self.tags[i]['attributes']*2),1,1,0),
            #2048+256+2通道   特征图大小1*1
        ) for i in range(len(self.tags))])
        #每一个tag都有一个自己独立的卷积
        #包含attribute*2个输出分支,一半用来检测Cycle，一半用来检测translation

    def execute(self,x,s,y,i):
        f=self.conv(x)
        fsy=jt.concat([f,tile_like(s,f),tile_like(y,f)],1)
        return self.fcs[i](fsy).view(f.size(0),2,-1)
        #将输出的2*attribute个通道分为两部分,一半用来检测Cycle,一半用来检测translation

    #防止鉴别器将所有传入S的图像都认为是假的，尽管这里的S是人为提取的
    def calc_dis_loss_real(self,x,s,y,i,j):#x原图,s原始风格...
        #铰链损失+R1正则项,同时优化Cycle和Translation两个鉴别通道
        loss = 0
        out = self.execute(x, s, y, i)[:, :, j]#选对应的attribute
        loss += nn.relu(1 - out[:, 0]).mean()#铰链损失均值
        loss += nn.relu(1 - out[:, 1]).mean()
        loss += self.R1reg(out[:, 0], x)#R1正则均值
        loss += self.R1reg(out[:, 1], x)
        return loss

    #对生成的图像进行鉴别优化D
    def calc_dis_loss_fake_trg(self, x, s, y, i, j):#x修改后的图，s目标风格，j目标attr
        out = self.execute(x, s, y, i)[:, :, j]
        loss = nn.relu(1 + out[:, 0]).mean()
        return loss
    def calc_dis_loss_fake_cyc(self, x, s, y, i, j):#xCycle重建图，s原始风格，j原始attr
        out = self.execute(x, s, y, i)[:, :, j]
        loss = nn.relu(1 + out[:, 1]).mean()
        return loss


    #为G提供指导
    #用来对风格代码提取器进行训练，因为只有s是提取的，促使风格代码提取器提取的风格符合鉴别器要求
    #和鉴别器一起协同，训练出一个可以在真实图像上有效提取出风格代码的提取器
    def calc_gen_loss_real(self, x, s, y, i, j):
        loss = 0
        out = self.execute(x, s, y, i)[:, :, j]
        loss += out[:, 0].mean()
        loss += out[:, 1].mean()
        return loss
    #计算生成图像的真实性损失，此时s被阻断梯度传播，不利用鉴别器的梯度自我优化，防止E破坏协同
    def calc_gen_loss_fake_trg(self, x, s, y, i, j):
        out = self.execute(x, s, y, i)[:, :, j]
        loss = - out[:, 0].mean()
        return loss
    #计算循环图像的真实性损失，此时s被阻断梯度传播，不利用鉴别器的梯度自我优化，防止E破坏协同
    def calc_gen_loss_fake_cyc(self, x, s, y, i, j):
        out = self.execute(x, s, y, i)[:, :, j]
        loss = - out[:, 1].mean()
        return loss

    def R1reg(self, d_out, x_in):#
        batch_size = x_in.shape[0]

        x_in.requires_grad=True
        grad = jt.grad(jt.sum(d_out), x_in)
        grad2 = grad.pow(2).reshape(batch_size, -1).sum(1)  # 每样本 L2²
        r1_penalty = grad2.mean()  # batch 平均
        if r1_penalty.item() == 0:
            print("R1 missing!!!")
        else:
            print(f"R1 grad mean: {r1_penalty.item():.4f}")
        return r1_penalty

class Gen(nn.Module):
    def __init__(self,hyperparameters):
        super().__init__()
        self.tags=hyperparameters['tags']

        self.style_dim = hyperparameters['style_dim']
        self.noise_dim = hyperparameters['noise_dim']

        channels = hyperparameters['encoder']['channels']

        self.encoder = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        channels = hyperparameters['decoder']['channels']
        self.decoder = nn.Sequential(
            *[UpBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Conv2d(channels[-1], hyperparameters['input_dim'], 1, 1, 0)
        )

        self.extractors = Extractors(hyperparameters)  # 接受的参数是图和tag id

        self.translators = nn.ModuleList([Translator(hyperparameters)  # 为每个tag都训练一个风格转换器
                                          for i in range(len(self.tags))]
                                         )

        self.mappers = nn.ModuleList([Mapper(hyperparameters, len(self.tags[i]['attributes']))
                                      for i in range(len(self.tags))]
                                     )

    def encode(self, x):  # 只有一个
        e = self.encoder(x)
        return e

    def decode(self, e):  # 只有一个
        x = self.decoder(e)
        return x

    def extract(self, x, i):  # X是需要提取的图片,i是需要提取的tag编号 只训练一个，但这一个最后有style_dim*num_tag个通道
        return self.extractors(x, i)

    def map(self, z, i, j):  # 对每个tag的每个attribute都训练一个mapper
        return self.mappers[i](z, j)

    def translate(self, e, s, i):  # 对每个tag都训练一个  e是图像编码，s是风格代码，i是tag id
        return self.translators[i](e, s)

    def Sync_Gen(self): #用于在EMA更新模型时强制同步，避免计算图保留
        for _, p in self.named_parameters():
            p.sync()

class Extractors(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        self.num_tags = len(hyperparameters['tags'])
        channels = hyperparameters['extractors']['channels']
        self.model = nn.Sequential(
            nn.Conv2d(hyperparameters['input_dim'], channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.AdaptiveAvgPool2d(1),#自适应平均池化，最后输出一个大小为4*4->1*1的图，实际是对这个图每个通道所有值求和取平均值
            nn.Conv2d(channels[-1],  hyperparameters['style_dim'] * self.num_tags, 1, 1, 0),
        )

    def execute(self, x, i):
        s = self.model(x)
        s=s.view(x.size(0), self.num_tags, -1)
        #print(s.shape)
        return s[:, i]


class Translator(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        channels = hyperparameters['translators']['channels']
        self.model = nn.Sequential(
            nn.Conv2d(hyperparameters['encoder']['channels'][-1], channels[0], 1, 1, 0),
            *[MiddleBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        self.style_to_params = nn.Linear(hyperparameters['style_dim'], self.get_num_adain_params(self.model))
        #接受模型输出的style code，然后线性映射，为每个AdaIn提供参数

        self.features = nn.Sequential(
            nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
        )

        self.masks = nn.Sequential(
            nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
            nn.Sigmoid()
        )#注意力掩码

    def execute(self, e, s):
        p = self.style_to_params(s)  # 风格在这里作为参数，通过线性层影响每个AdaIN的参数
        self.assign_adain_params(p, self.model)# 这里为所有的IN分配参数

        mid = self.model(e)
        f = self.features(mid)
        m = self.masks(mid)

        return f * m + e * (1 - m)  # 原编码中权值大的部分受影响更大

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:  # 对于每一个AdaIN分配值，adain_params第一维是batch
                m._bias = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features, 1)
                m._weight = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features,1) + 1 #这里令参数为NCL三维，方便展平计算
                if adain_params.size(1) > 2 * m.num_features:#如果参数还有剩下的，就裁掉前面的继续往下发
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        #计算有多少个AdaIN
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:  # 如果模块的类的名字是AdaIN,那就需要2*通道数个参数
                num_adain_params += 2 * m.num_features
        return num_adain_params


class Mapper(nn.Module):
    def __init__(self, hyperparameters, num_attributes):
        super().__init__()
        # 风格映射器就纯纯大力出奇迹了，噪音采样，多个线性层和激活函数叠加
        # 每个tag独立一次生成，每个attribute独立一个Mapper
        channels = hyperparameters['mappers']['pre_channels']
        self.pre_model = nn.Sequential(
            nn.Linear(hyperparameters['noise_dim'], channels[0]),
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        # pre训练的是一个整体风格生成器，认为是多个attribute的共同前身? 的确对于控制统一属性的风格而言，应当有共同之处

        channels = hyperparameters['mappers']['post_channels']
        self.post_models = nn.ModuleList([nn.Sequential(
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Linear(channels[-1], hyperparameters['style_dim']),
        ) for i in range(num_attributes)
        ])
        # post是对每个attribute都单独再加几个线性层，产生差异

    def execute(self, z, j):
        z = self.pre_model(z)
        return self.post_models[j](z)






class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

        self.AvgPool2d=nn.AvgPool2d(2)

    def execute(self, x):
        residual = self.AvgPool2d(self.sc(x))
        out = self.conv2(self.activ(self.AvgPool2d(self.conv1(self.activ(x.clone())))))
        out = residual + out
        return out / math.sqrt(2)


class DownBlockIN(nn.Module):#下采样块 每执行一次图片的长和宽都减半，依靠池化层实现
    def __init__(self, in_dim, out_dim):#参数是 这一层输入通道数 和 输出通道数
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in1 = InstanceNorm2d(in_dim) #实例归一化层
        self.in2 = InstanceNorm2d(in_dim) #实例归一化层

        self.activ = nn.LeakyReLU(0.2)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

        self.AvgPool2d = nn.AvgPool2d(2)

    def execute(self, x):
        residual = self.AvgPool2d(self.sc(x))#对sc的卷积结果进行平均池化，实现残差连接
        out = self.conv2(self.activ(self.in2(self.AvgPool2d(self.conv1(self.activ(self.in1(x.clone())))))))
        #一系列的处理，最后是用卷积卷成输出通道数
        out = residual + out
        return out / math.sqrt(2)#这里再做一次归一化，因为加上了残差


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def execute(self, x):
        residual = nn.interpolate(self.sc(x), scale_factor=2, mode='nearest')#使用缩放因子，而不显式指定size
        out = self.conv2(self.activ(self.conv1(nn.interpolate(self.activ(x.clone()), scale_factor=2, mode='nearest'))))
        out = residual + out
        return out / math.sqrt(2)

class UpBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def execute(self, x):
        residual = nn.interpolate(self.sc(x), scale_factor=2, mode='nearest')#做插值，临近填充
        out = self.conv2(self.activ(self.in2(self.conv1(nn.interpolate(self.activ(self.in1(x.clone())), scale_factor=2, mode='nearest')))))
        out = residual + out #这里也做了残差
        return out / math.sqrt(2)

class MiddleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.adain1 = AdaptiveInstanceNorm2d(in_dim)#这里提及的Adain，参数是通道数
        self.adain2 = AdaptiveInstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)
        #不需要改变图尺寸所以没有池化或者插值层

    def execute(self, x):
        residual = self.sc(x)
        out = self.conv2(self.activ(self.adain2(self.conv1(self.activ(self.adain1(x.clone()))))))
        out = residual + out
        return out / math.sqrt(2)

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activ = nn.ReLU()

    def execute(self, x):
        return self.linear(self.activ(x))

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = jt.ones((1, num_features, 1))
        self.bias = jt.zeros((1, num_features, 1))
        #首先初始化为全1和全0，但是是可学习的
    def execute(self, x):
        N, C, H, W = x.size()    #批次内图数,通道数,高度,宽度
        x = x.view(N, C, -1)     #先转成一个三维的 就只在乎图和通道，不在乎长宽
        bias_in = x.mean(-1, keepdim=True)#对最后1维做均值，且仍保留原格式  返回值是一个N*C*1的三维张量，其中这个1代表 某张图在某个频道上的均值
        weight_in = x.std(-1, keepdim=True)#标准差这里也一样

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias   #先归一化求出来值后*权重再加上偏置   权重和偏置都可以学习
        #其中的eps是防止标准差为0造成除0错误的安全符
        return out.view(N, C, H, W)#返回的还是4维的张量

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'



class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features#参数是通道数，对于每一个通道都需要一个bias和weight
        self.eps = eps

        #Adain的偏移和权重都来自于风格图
        self._bias = None
        self._weight = None

    def execute(self, x):
        assert self._bias is not None, "Please assign weight and bias before calling AdaIN!"
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        _bias_in = x.mean(-1, keepdim=True)
        _weight_in = x.std(-1, keepdim=True)

        out = (x - _bias_in) / (_weight_in + self.eps) * self._weight + self._bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'