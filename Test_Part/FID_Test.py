import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.linalg import sqrtm
from utils import get_config
from utils import get_data_iters_test
import torch
from torchvision import models, transforms

config = get_config('Configure/celeba-hq.yaml')
train_iters = get_data_iters_test(config, ['0'])

Iter=train_iters[0][0] # bangs.with

def get_activations_sample(folder, model, device, image_size=299,length=8000):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    model.eval()
    activations = []
    CNT=0
    with torch.no_grad():
        for img_name in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
            CNT+=1
            img_path = os.path.join(folder, img_name)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            feat = model(img_tensor).squeeze().cpu().numpy()
            activations.append(feat)
            if(CNT==length):
                break

    return np.array(activations)

def get_activations_Orginal(Iter,model, device,length=5000):
    model.eval()
    activations = []
    CNT=0
    with torch.no_grad():
        while(1):
            Images , Irr_label =Iter.next()
            Iter.preload()
            Images=[Images[i] for i in range(0,len(Irr_label)) if Irr_label[i][0]==1 and Irr_label[i][1]==1]  # for disentanglement
            if(len(Images)==0):
                continue
            Images=torch.stack(Images,dim=0)
            CNT+=Images.shape[0]
            Images=Images.to(device)
            feats = model(Images).cpu().numpy()
            activations.extend(feats)
            if(CNT>=length):
                break
    return np.array(activations)


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_fid_score(gen_folder):
    device = torch.device('cuda')
    inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Remove classification layer

    act1 = get_activations_Orginal(Iter, inception, device)
    act2 = get_activations_sample(gen_folder, inception, device)

    fid_value = calculate_fid(act1, act2)
    print(f"\nFID score: {fid_value:.4f}")
    return fid_value

# 使用示例：
#compute_fid_score('../HiSD-Jittor/disentanglement_L_bangs')
#compute_fid_score('../HiSD-Jittor/disentanglement_R_bangs')

compute_fid_score('disentanglement_L_bangs')
compute_fid_score('disentanglement_R_bangs')
