import re
Jittor=[]
Pytorch=[]
with open("Jittor/Jittor_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[0]=='['):
            match = re.search(r"([0-9.]+)s", line)
            match=float(match.group(1))
            if(match>=5):
                print(match)
            Jittor.append(match)
with open("Pytorch/Pytorch_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[0]=='['):
            match = re.search(r"([0-9.]+)s", line)
            match=float(match.group(1))
            Pytorch.append(match)

import matplotlib.pyplot as plt

# 横坐标：每10个Epoch，对应编号
Jittor_10000=[]
Pytorch_10000=[]
for i in range(0,20):
    Jittor_10000.append(sum(Jittor[i*1000:(i+1)*1000-1]))
    Pytorch_10000.append(sum(Pytorch[i * 1000:(i + 1) * 1000 - 1]))

batchs = [i for i in range(1, len(Jittor_10000) + 1)]
plt.figure(figsize=(8, 5))
plt.plot(batchs, Jittor_10000, marker='o', label='HiSD128_Jittor', linestyle='-', linewidth=2)
plt.plot(batchs, Pytorch_10000, marker='s', label='HiSD128_Pytorch', linestyle='--', linewidth=2)

plt.title("Training Time per 10000 Batch")
plt.xlabel("10KBatch")
plt.ylabel("Time (s)")
plt.xticks(batchs)  # 保证横坐标只显示整数Epoch
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()