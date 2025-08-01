import Device_set
import jittor as jt
from jittor import nn
from networks import Gen, Dis
from Dataset import get_data_iters
from utils import weights_init
import yaml
import random
import copy
import os


def update_average(model_tgt, model_src, beta=0.99):
    #平均化更新，每次只更新1%
    with jt.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.assign(beta * p_tgt + (1 - beta) * p_src)
            #p_tgt.sync()每次都计算太慢，因此每个Batch结束统一sync

class HiSD(nn.Module):
    def __init__(self, conf):
        super(HiSD, self).__init__()
        self.G = Gen(conf)
        self.D = Dis(conf)
        self.hyperparameters = conf
        self.noise_dim = conf['noise_dim']
        self.hyperparameters = conf

    def execute(self, args, mode):
        if mode == 'gen':
            return self.gen_losses(*args)
        elif mode == 'dis':
            return self.dis_losses(*args)
        else:
            pass

    def gen_losses(self, x, y, i, j, j_trg):
        batch = x.size(0)

        # non-translation path
        e = self.G.encode(x)
        x_rec = self.G.decode(e)

        # self-translation path
        s = self.G.extract(x, i)
        e_slf = self.G.translate(e, s, i)
        x_slf = self.G.decode(e_slf)

        # cycle-translation path
        ## translate
        s_trg = self.G.map(jt.randn(batch, self.hyperparameters['noise_dim']), i, j_trg)
        e_trg = self.G.translate(e, s_trg, i)
        x_trg = self.G.decode(e_trg)
        ## cycle-back
        e_trg_rec = self.G.encode(x_trg)
        s_trg_rec = self.G.extract(x_trg, i)
        e_cyc = self.G.translate(e_trg_rec, s, i)
        x_cyc = self.G.decode(e_cyc)

        loss_gen_adv = self.D.calc_gen_loss_real(x, s, y, i, j) + \
                       self.D.calc_gen_loss_fake_trg(x_trg, s_trg.detach(), y, i, j_trg) + \
                       self.D.calc_gen_loss_fake_cyc(x_cyc, s.detach(), y, i, j)

        loss_gen_sty = nn.l1_loss(s_trg_rec, s_trg)

        loss_gen_rec = nn.l1_loss(x_rec, x) + \
                       nn.l1_loss(x_slf, x) + \
                       nn.l1_loss(x_cyc, x)

        loss_gen_total = self.hyperparameters['adv_w'] * loss_gen_adv + \
                         self.hyperparameters['sty_w'] * loss_gen_sty + \
                         self.hyperparameters['rec_w'] * loss_gen_rec

        return loss_gen_total, loss_gen_adv, loss_gen_sty, loss_gen_rec, \
            x_trg.detach(), x_cyc.detach(), s.detach(), s_trg.detach()

    def dis_losses(self, x, x_trg, x_cyc, s, s_trg, y, i, j, j_trg):
        loss_dis_adv = self.D.calc_dis_loss_real(x, s, y, i, j) + \
                       self.D.calc_dis_loss_fake_trg(x_trg, s_trg, y, i, j_trg) + \
                       self.D.calc_dis_loss_fake_cyc(x_cyc, s, y, i, j)
        return loss_dis_adv


class HiSD_Trainer(nn.Module):
    def __init__(self, conf):
        super(HiSD_Trainer, self).__init__()
        self.loss_gen_rec = None
        self.loss_gen_sty = None
        self.loss_gen_adv = None
        self.loss_gen_total = None
        self.models = HiSD(conf)
        beta1 = conf['beta1']
        beta2 = conf['beta2']
        self.dis_opt = jt.optim.Adam(self.models.D.parameters(),
                                     lr=conf['lr_dis'], betas=(beta1, beta2),
                                     weight_decay=conf['weight_decay'])

        self.gen_opt = jt.optim.Adam([{'params': self.models.G.encoder.parameters()},
                                      {'params': self.models.G.translators.parameters()},
                                      {'params': self.models.G.extractors.parameters()},
                                      {'params': self.models.G.decoder.parameters()},
                                      # Different LR for mappers.
                                      {'params': self.models.G.mappers.parameters(),
                                       'lr': conf['lr_gen_mappers']},
                                      ],
                                     lr=conf['lr_gen_others'], betas=(beta1, beta2),
                                     weight_decay=conf['weight_decay'])
        self.models.G.apply(weights_init(conf['init']))
        self.models.D.apply(weights_init(conf['init']))
        self.G_test = copy.deepcopy(self.models.G)  #用来进行平均化更新
        for p in self.G_test.parameters():
            p.requires_grad = False

    def update(self, x, y, i, j, j_trg,PrintFlag=False,Iters=None):
        current_model = self.models
        #For G
        for p in current_model.D.parameters():
            p.requires_grad = False
        for p in current_model.G.parameters():
            p.requires_grad = True
        self.gen_opt.zero_grad()

        self.loss_gen_total, self.loss_gen_adv, self.loss_gen_sty, self.loss_gen_rec, \
            x_trg, x_cyc, s, s_trg = current_model.gen_losses(x, y, i, j, j_trg)

        self.loss_gen_adv = self.loss_gen_adv.detach().mean()  # 对抗性能
        self.loss_gen_sty = self.loss_gen_sty.detach().mean()  # 风格提取
        self.loss_gen_rec = self.loss_gen_rec.detach().mean()  # 重建损失

        # 获得损失和阻断梯度传播的信息，
        self.gen_opt.step(self.loss_gen_total)#优化模型
        self.gen_opt.clip_grad_norm(100, norm_type=2)



        #For D
        for p in current_model.D.parameters():
            p.requires_grad = True
        for p in current_model.G.parameters():
            p.requires_grad = False

        self.dis_opt.zero_grad()

        self.loss_dis_adv = current_model.dis_losses(x, x_trg, x_cyc, s, s_trg, y, i, j, j_trg)

        self.dis_opt.step(self.loss_dis_adv)#优化模型
        self.dis_opt.clip_grad_norm(100, norm_type=2)
        self.loss_dis_adv = self.loss_dis_adv.detach().mean()


        update_average(self.G_test, current_model.G)
        Iters.preload()#预加载一下
        self.G_test.Sync_Gen()

        if(PrintFlag):
            print("loss_gen_adv", self.loss_gen_adv.numpy())
            print("loss_gen_sty", self.loss_gen_sty.numpy())
            print("loss_gen_rec", self.loss_gen_rec.numpy())
            print("loss_dis_adv", self.loss_dis_adv.numpy())

        #return self.loss_gen_adv.numpy(),self.loss_gen_sty.numpy(),self.loss_gen_sty.numpy(),self.loss_dis_adv.numpy()

    def sample(self,x, x_trg, j, j_trg, i):
        G=self.G_test
        out = [x]
        with jt.no_grad():
            e = G.encode(x)

            # Latent-guided 1
            z = jt.randn(1, G.noise_dim).cuda().repeat(x.size(0), 1)
            s_trg = G.map(z, i, j_trg)
            x_trg_ = G.decode(G.translate(e, s_trg, i))
            out += [x_trg_]

            # Latent-guided 2
            z = jt.randn(1, G.noise_dim).cuda().repeat(x.size(0), 1)
            s_trg = G.map(z, i, j_trg)
            x_trg_ = G.decode(G.translate(e, s_trg, i))
            out += [x_trg_]

            s_trg = G.extract(x_trg, i)
            # Reference-guided 1: use x_trg[0, 1, ..., n] as reference
            x_trg_ = G.decode(G.translate(e, s_trg, i))
            out += [x_trg, x_trg_]

            # Reference-guided 2: use x_trg[n, n-1, ..., 0] as reference
            x_trg_ = G.decode(G.translate(e, s_trg.flip([0]), i))
            out += [x_trg.flip([0]), x_trg_]
        return out

    def save(self, snapshot_dir, iterations):
        this_model = self.models
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pkl' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pkl' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pkl')
        jt.save({'gen': this_model.G.state_dict(), 'gen_test': self.G_test.state_dict()}, str(gen_name))
        jt.save({'dis': this_model.D.state_dict()}, str(dis_name))
        jt.save({'dis': self.dis_opt.state_dict(),
                    'gen': self.gen_opt.state_dict()}, str(opt_name))

