import Device_set
from utils import prepare_sub_folder, write_2images
import argparse
from Trainer import HiSD_Trainer
import os
import shutil
import random
from Dataset import get_data_iters
import yaml
import jittor as jt
import numpy as np

def main():
    jt.set_global_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Configure/celeba-hq.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='128Version', help="outputs path")
    opts = parser.parse_args()

    # Load experiment setting
    with open(opts.config,'r') as f:
        config = yaml.safe_load(f)


    total_iterations = config['total_iterations']

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    #这里传入的config里面已经写入了超参数
    #通过Trainer创建生成器判别器之类的
    #Trainer里面创建了一个模型
    trainer = HiSD_Trainer(config)
    iterations = 0


    trainer.cuda()

    # Setup data loader
    train_iters = get_data_iters(config)
    tags = list(range(len(train_iters)))


    import time
    start = time.time()
    while True:
        """
        i: tag
        j: 原attribute, j_trg: 目标attribute
        x: 原图, y: 属性无关标签
        """
        i = random.sample(tags, 1)[0]
        j, j_trg = random.sample(list(range(len(train_iters[i]))), 2)
        x, y = train_iters[i][j].next()

        trainer.update(x, y, i, j, j_trg,PrintFlag=((iterations + 1) % config['log_iter'] == 0),Iters=train_iters[i][j])

        if (iterations + 1) % config['image_save_iter'] == 0:
            for i in range(len(train_iters)):
                j, j_trg = random.sample(list(range(len(train_iters[i]))), 2)

                x, _ = train_iters[i][j].next()
                x_trg, _ = train_iters[i][j_trg].next()
                train_iters[i][j].preload()
                train_iters[i][j_trg].preload()

                test_image_outputs = trainer.sample(x, x_trg, j, j_trg, i)
                write_2images(test_image_outputs,
                              config['batch_size'],
                              image_directory, 'sample_%08d_%s_%s_to_%s' % (iterations + 1, config['tags'][i]['name'], config['tags'][i]['attributes'][j]['name'], config['tags'][i]['attributes'][j_trg]['name']))


        if (iterations + 1) % config['log_iter'] == 0:
            #jt.display_memory_info()
            now = time.time()
            print(f"[#{iterations + 1:06d}|{total_iterations:d}] {now - start:5.2f}s")
            start = now

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        if (iterations + 1) == total_iterations:
            print('Finish training!')
            exit(0)

        iterations += 1
if __name__=='__main__':
    main()