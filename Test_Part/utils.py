"""
The main codes are form MUNIT
"""
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageAttributeDataset
import torch
import torch.nn as nn
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
import multiprocessing
import sys

def get_data_iters(conf, gpus):
    batch_size = conf['batch_size']
    new_size = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    num_workers = conf['num_workers']
    tags  = conf['tags']

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list 
    transform_list = [transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)] + transform_list 
    transform = transforms.Compose(transform_list)

    #对图片的处理，先添加色彩抖动，然后随机旋转，放缩到指定size，再随机切一个height-weight大小的图，最后映射成张量，并且归一化

    loaders = [[DataLoader(
        dataset=ImageAttributeDataset(tags[i]['attributes'][j]['filename'], transform),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
                            for j in range(len(tags[i]['attributes']))] for i in range(len(tags))]

    iters = [[data_prefetcher(loader, batch_size, gpus) for loader in loaders] for loaders in loaders]

    return iters


def get_data_iters_samples(conf, gpus):
    batch_size = conf['batch_size']
    new_size = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    num_workers = conf['num_workers']
    tags  = conf['tags']

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize(new_size)] + transform_list
    transform = transforms.Compose(transform_list)


    loaders = [[DataLoader(
        dataset=ImageAttributeDataset(tags[i]['attributes'][j]['filename'], transform),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
                            for j in range(len(tags[i]['attributes']))] for i in range(len(tags))]

    iters = [[data_prefetcher(loader, batch_size, gpus) for loader in loaders] for loaders in loaders]

    return iters

def get_data_iters_test(conf, gpus):
    batch_size = conf['batch_size']
    tags  = conf['tags']

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    loaders = [[DataLoader(
        dataset=ImageAttributeDataset(tags[i]['attributes'][j]['filename'], transform),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
                            for j in range(len(tags[i]['attributes']))] for i in range(len(tags))]

    iters = [[data_prefetcher(loader, batch_size, gpus) for loader in loaders] for loaders in loaders]

    return iters


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

class data_prefetcher():
    def __init__(self, loader, batch_size, gpus):

        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.batch_size = batch_size
        self.gpu0 = int(gpus[0])
        self.preload()

    def preload(self):
        try:
            self.x, self.y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            self.x, self.y = next(self.iter)

        if self.x.size(0) != self.batch_size:
            self.iter = iter(self.loader)
            self.x, self.y = next(self.iter)
        
        with torch.cuda.stream(self.stream):
            self.x, self.y = self.x.cuda(self.gpu0, non_blocking=True), self.y.cuda(self.gpu0, non_blocking=True)

    def next(self):
        return self.x, self.y