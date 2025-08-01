# import packages
import Device_set
from Trainer import HiSD_Trainer
import argparse
import sys
import yaml
import jittor as jt
from PIL import Image
import numpy as np
import time
from Dataset import get_data_iters_samples
import random

# load checkpoint
with open('Configure/celeba-hq.yaml', 'r') as f:
    config = yaml.safe_load(f)

noise_dim = config['noise_dim']
image_size = config['new_size']
checkpoint = 'gen_128_00200000.pkl'
trainer = HiSD_Trainer(config)
state_dict = jt.load(checkpoint)
trainer.models.G.load_state_dict(state_dict['gen_test'])
trainer.models.G

E = trainer.models.G.encode
T = trainer.models.G.translate
G = trainer.models.G.decode
M = trainer.models.G.map
F = trainer.models.G.extract

train_iters = get_data_iters_samples(config)
tags = [0, 1]  #Bangs/Glasses

cnt = 0

import os


def save_batch_images(batch, save_dir, prefix=0):
    batch_np = batch.numpy()
    for idx, img in enumerate(batch_np):
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        img = np.clip((img * 255), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_dir, f"{prefix + idx}.jpg"))


def getSamples():
    global cnt
    i = 0  #bangs
    j = 1  #without
    j_trg = 0  #with
    x, Irr_labels = train_iters[i][j].next()  #without_sample
    x_ref, _ = train_iters[i][j_trg].next()  #with_sample
    train_iters[i][j].preload()
    train_iters[i][j_trg].preload()

    YoungMan = []
    for num in range(0, x.shape[0]):
        if (Irr_labels[num][0] == 1 and Irr_labels[num][1] == 1):
            YoungMan.append(x[num])
    if (len(YoungMan) == 0):
        return
    batch = jt.stack(YoungMan, dim=0)
    x = batch

    e = E(x)

    # Latent
    z = jt.randn(config['batch_size'], config['noise_dim'])
    z = z[:x.shape[0]]
    s_trg = M(z, i, j_trg)
    e_trg = T(e, s_trg, i)
    x_trg_L = G(e_trg)
    x_trg_L = (x_trg_L + 1) / 2

    # Reference
    x_ref = x_ref[:x.shape[0]]
    s_trg = F(x_ref, i)
    e_trg = T(e, s_trg, i)
    x_trg_R = G(e_trg)
    x_trg_R = (x_trg_R + 1) / 2

    save_batch_images(x_trg_L, "disentanglement_L_bangs", prefix=cnt)
    save_batch_images(x_trg_R, "disentanglement_R_bangs", prefix=cnt)

    cnt += len(x_trg_L)

while (cnt <= 3000):
    getSamples()
