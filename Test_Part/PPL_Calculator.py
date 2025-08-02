import os
from PIL import Image
import torch
import lpips
from torchvision import transforms
from tqdm import tqdm

def compute_ppl_from_folder(folder_path, epsilon=1e-3, image_size=128):
    # 初始化 LPIPS (默认用 VGG)
    loss_fn = lpips.LPIPS(net='vgg').to('cuda')

    # 图像预处理：Resize + [-1, 1] Normalize
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize((0.5,)*3, (0.5,)*3)  # → [-1,1]
    ])

    # 找出成对图像的 index（以 A/B+数字+.jpg 命名）
    all_files = os.listdir(folder_path)
    indices = sorted(set(
        f[1:-4] for f in all_files if f.startswith('A') and f"A{f[1:]}" in all_files and f"B{f[1:]}" in all_files
    ))

    ppl_values = []

    for idx in tqdm(indices):
        try:
            imgA = Image.open(os.path.join(folder_path, f"A{idx}.jpg")).convert('RGB')
            imgB = Image.open(os.path.join(folder_path, f"B{idx}.jpg")).convert('RGB')
        except Exception as e:
            print(f"跳过图像对 {idx}: {e}")
            continue

        imgA_tensor = transform(imgA).unsqueeze(0).to('cuda')
        imgB_tensor = transform(imgB).unsqueeze(0).to('cuda')

        with torch.no_grad():
            d = loss_fn(imgA_tensor, imgB_tensor).item()
            ppl = (d ** 2) / (epsilon ** 2)
            ppl_values.append(ppl)

    if ppl_values:
        mean_ppl = sum(ppl_values) / len(ppl_values)
        print(f"\n图像文件夹 PPL（ε={epsilon}）：{mean_ppl:.4f}")
        return mean_ppl
    else:
        print("没有有效图像对")
        return None
compute_ppl_from_folder("../HiSD-Jittor/PPL")