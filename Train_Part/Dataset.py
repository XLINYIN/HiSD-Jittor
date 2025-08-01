from jittor.dataset import Dataset, DataLoader
from PIL import Image
import numpy as np
import jittor.transform
import time

class CelebA_HQ_Attribute(Dataset):
    def __init__(self, filename, transform, batch_size, shuffle, drop_last, num_workers):
        super().__init__()
        self.lines = [line.rstrip().split() for line in open(filename, 'r')]
        self.transform = transform
        self.length = len(self.lines)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.set_attrs(batch_size=self.batch_size, shuffle=self.shuffle,
                       total_len=self.length,drop_last=self.drop_last, num_workers=self.num_workers)

    def __getitem__(self, index):
        line = self.lines[index]
        image = Image.open(line[0]).convert('RGB')
        conditions = [np.int32(condition) for condition in line[1:]]
        return self.transform(image), jittor.array(conditions)
        #Lazy load

#用于训练采样的iters,进行了数据增强以提升模型鲁棒性
def get_data_iters(conf):
    batch_size = conf['batch_size']
    new_size = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    num_workers = conf['num_workers']
    tags = conf['tags']

    transform_list = [jittor.transform.ImageNormalize((0.5, 0.5, 0.5),
                                                      (0.5, 0.5, 0.5))]
    transform_list = [jittor.transform.RandomCrop((height, width))] + transform_list
    transform_list = [jittor.transform.Resize(new_size)] + transform_list
    transform_list = [jittor.transform.RandomHorizontalFlip()] + transform_list  #随机翻转
    transform_list = [jittor.transform.ColorJitter(0.1, 0.1, 0.1, 0.1)] + transform_list  #色操扰动，亮度对比度饱和度色调
    transform_list = jittor.transform.Compose(transform_list)
    #对图片的处理，先添加色彩抖动，然后随机旋转，放缩到指定size，再随机切一个height-weight大小的图，最后映射成张量，并且归一化

    Sets = [[
        CelebA_HQ_Attribute(tags[i]['attributes'][j]['filename'], transform_list, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=num_workers)
        for j in range(len(tags[i]['attributes']))] for i in range(len(tags))]

    loaders = [[DataLoader(attribute_set) for attribute_set in tag_set] for tag_set in Sets]
    iters = [[data_prefetcher(attribute_loader) for attribute_loader in tag_loader] for tag_loader in
             loaders]

    return iters

#用来计算FID和PPL,不添加扰动
def get_data_iters_samples(conf):
    batch_size = conf['batch_size']
    new_size = conf['new_size']
    num_workers = conf['num_workers']
    tags = conf['tags']

    transform_list = jittor.transform.Compose([jittor.transform.Resize(new_size),
                                   jittor.transform.ToTensor(),
                                   jittor.transform.ImageNormalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))])
    Sets = [[
        CelebA_HQ_Attribute(tags[i]['attributes'][j]['filename'], transform_list, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=num_workers)
        for j in range(len(tags[i]['attributes']))] for i in range(len(tags))]

    loaders = [[DataLoader(attribute_set) for attribute_set in tag_set] for tag_set in Sets]
    iters = [[data_prefetcher(attribute_loader) for attribute_loader in tag_loader] for tag_loader in
             loaders]
    return iters

#直接采样原图，用来计算FID的Iters，要满足Fid的归一化标准
def get_data_iters_test(conf):
    batch_size = conf['batch_size']
    tags  = conf['tags']
    num_workers = conf['num_workers']
    transform = jittor.transform.Compose([
        jittor.transform.Resize((299, 299)),
        jittor.transform.ToTensor(),
        jittor.transform.ImageNormalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    Sets = [[
        CelebA_HQ_Attribute(tags[i]['attributes'][j]['filename'], transform, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=num_workers)
        for j in range(len(tags[i]['attributes']))] for i in range(len(tags))]

    loaders = [[DataLoader(attribute_set) for attribute_set in tag_set] for tag_set in Sets]
    iters = [[data_prefetcher(attribute_loader) for attribute_loader in tag_loader] for tag_loader in
             loaders]
    return iters


class data_prefetcher():
    def __init__(self, loader):

        self.loader = loader
        self.iter = iter(self.loader)
        self.preload()
    def preload(self):
        try:
            self.x, self.y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            self.x, self.y = next(self.iter)
        self.x.cuda()
        self.y.cuda()

    def next(self):
        return self.x, self.y

