# import packages
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

save_dir = "output_images"
cnt = 0

import os


def save_batch_imagesA(batch, save_dir, prefix=0):
    """
    batch: jt.Var of shape [B, C, H, W]
    """
    batch_np = batch.numpy()
    for idx, img in enumerate(batch_np):
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        img = np.clip((img * 255), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_dir, f"A{prefix + idx}.jpg"))

def save_batch_imagesB(batch, save_dir, prefix=0):
    """
    batch: jt.Var of shape [B, C, H, W]
    """
    batch_np = batch.numpy()
    for idx, img in enumerate(batch_np):
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        img = np.clip((img * 255), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_dir, f"B{prefix + idx}.jpg"))


def getSamples():
    global cnt
    i = 0  #bangs
    j = 1  #without
    j_trg = 0  #with
    x, _= train_iters[i][j].next()  #without_sample
    train_iters[i][j].preload()

    epsilon=1e-3
    e = E(x)
    # Latent
    z1 = jt.randn(config['batch_size'], config['noise_dim'])
    z2 = jt.randn(config['batch_size'], config['noise_dim'])
    s1_trg = M(z1, i, j_trg)
    s2_trg = M(z2, i, j_trg)

    t=jt.randint(0, 2,shape=(config['batch_size'], 1))
    s1=s1_trg*(jt.ones((config['batch_size'],1))-t)+s2_trg*t
    s2=s1_trg*(jt.ones((config['batch_size'],1))-t-epsilon)+s2_trg*(t+epsilon)

    e1=T(e,s1,i)
    e2=T(e,s2,i)
    x_trg_1 = G(e1)
    x_trg_1 = (x_trg_1 + 1) / 2

    x_trg_2 = G(e2)
    x_trg_2= (x_trg_2 + 1) / 2

    save_batch_imagesA(x_trg_1, "PPL", prefix=cnt)
    save_batch_imagesB(x_trg_2, "PPL", prefix=cnt)
    cnt += len(x_trg_1)

while (cnt <= 2400):
    getSamples()
