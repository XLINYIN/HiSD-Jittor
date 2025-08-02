import re
Jittor=[]
Pytorch=[]

'''  Gen_loss_adv
with open("Jittor/Jittor_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[5:12]=="gen_adv"):
            match = re.search(r"([0-9.]+)", line)
            match=float(match.group(1))
            print(match)
            Jittor.append(match)
with open("Pytorch/Pytorch_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[5:12]=="gen_adv"):
            match = re.search(r"([0-9.]+)", line)
            match=float(match.group(1))
            print(match)
            Pytorch.append(match)

import matplotlib.pyplot as plt

# 横坐标：每10个Epoch，对应编号
Jittor_10000=[]
Pytorch_10000=[]
print(f"Jittor_10000_Batch10K {sum(Jittor[0:999])/1000}")
print(f"Pytorch_10000_Batch10K {sum(Pytorch[0:999])/1000}")
for i in range(1,20):
    Jittor_10000.append(sum(Jittor[i*1000:(i+1)*1000-1])/1000)
    Pytorch_10000.append(sum(Pytorch[i * 1000:(i + 1) * 1000 - 1])/1000)

batchs = [i for i in range(2, len(Jittor_10000) + 2)]
plt.figure(figsize=(8, 5))
plt.plot(batchs, Jittor_10000, marker='o', label='HiSD128_Jittor', linestyle='-', linewidth=2)
plt.plot(batchs, Pytorch_10000, marker='s', label='HiSD128_Pytorch', linestyle='--', linewidth=2)

plt.title("Gen_Loss_Adv")
plt.xlabel("10KBatch")
plt.ylabel("Time (s)")
plt.xticks(batchs)  # 保证横坐标只显示整数Epoch
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#Gen_Loss_Sty
with open("Jittor/Jittor_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[5:12]=="gen_sty"):
            match = re.search(r"([0-9.]+)", line)
            match=float(match.group(1))
            #print(match)
            Jittor.append(match)
with open("Pytorch/Pytorch_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[5:12]=="gen_sty"):
            match = re.search(r"([0-9.]+)", line)
            match=float(match.group(1))
            #print(match)
            Pytorch.append(match)

import matplotlib.pyplot as plt

# 横坐标：每10个Epoch，对应编号
Jittor_10000=[]
Pytorch_10000=[]
print(f"Jittor_10000_Batch10K {sum(Jittor[0:999])/1000}")
print(f"Pytorch_10000_Batch10K {sum(Pytorch[0:999])/1000}")
for i in range(1,20):
    Jittor_10000.append(sum(Jittor[i*1000:(i+1)*1000-1])/1000)
    Pytorch_10000.append(sum(Pytorch[i * 1000:(i + 1) * 1000 - 1])/1000)

batchs = [i for i in range(2, len(Jittor_10000) + 2)]
plt.figure(figsize=(8, 5))
plt.plot(batchs, Jittor_10000, marker='o', label='HiSD128_Jittor', linestyle='-', linewidth=2)
plt.plot(batchs, Pytorch_10000, marker='s', label='HiSD128_Pytorch', linestyle='--', linewidth=2)

plt.title("Gen_Loss_Sty")
plt.xlabel("10KBatch")
plt.ylabel("Time (s)")
plt.xticks(batchs)  # 保证横坐标只显示整数Epoch
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Gen_Loss_Rec
with open("Jittor/Jittor_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[5:12]=="gen_rec"):
            match = re.search(r"([0-9.]+)", line)
            match=float(match.group(1))
            #print(match)
            Jittor.append(match)
with open("Pytorch/Pytorch_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[5:12]=="gen_rec"):
            match = re.search(r"([0-9.]+)", line)
            match=float(match.group(1))
            #print(match)
            Pytorch.append(match)

import matplotlib.pyplot as plt

# 横坐标：每10个Epoch，对应编号
Jittor_10000=[]
Pytorch_10000=[]
print(f"Jittor_10000_Batch10K {sum(Jittor[0:999])/1000}")
print(f"Pytorch_10000_Batch10K {sum(Pytorch[0:999])/1000}")
for i in range(1,20):
    Jittor_10000.append(sum(Jittor[i*1000:(i+1)*1000-1])/1000)
    Pytorch_10000.append(sum(Pytorch[i * 1000:(i + 1) * 1000 - 1])/1000)

batchs = [i for i in range(2, len(Jittor_10000) + 2)]
plt.figure(figsize=(8, 5))
plt.plot(batchs, Jittor_10000, marker='o', label='HiSD128_Jittor', linestyle='-', linewidth=2)
plt.plot(batchs, Pytorch_10000, marker='s', label='HiSD128_Pytorch', linestyle='--', linewidth=2)

plt.title("Gen_Loss_Rec")
plt.xlabel("10KBatch")
plt.ylabel("Time (s)")
plt.xticks(batchs)  # 保证横坐标只显示整数Epoch
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




#Dis_Loss_Adv
with open("Jittor/Jittor_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[5:12]=="dis_adv"):
            match = re.search(r"([0-9.]+)", line)
            match=float(match.group(1))
            #print(match)
            Jittor.append(match)
with open("Pytorch/Pytorch_128.log",'r') as f:
    lines=f.readlines()
    for line in lines:
        if(line[5:12]=="dis_adv"):
            match = re.search(r"([0-9.]+)", line)
            match=float(match.group(1))
            #print(match)
            Pytorch.append(match)

import matplotlib.pyplot as plt

# 横坐标：每10个Epoch，对应编号
Jittor_10000=[]
Pytorch_10000=[]
print(f"Jittor_10000_Batch10K {sum(Jittor[0:999])/1000}")
print(f"Pytorch_10000_Batch10K {sum(Pytorch[0:999])/1000}")
for i in range(0,20):
    Jittor_10000.append(sum(Jittor[i*1000:(i+1)*1000-1])/1000)
    Pytorch_10000.append(sum(Pytorch[i * 1000:(i + 1) * 1000 - 1])/1000)

batchs = [i for i in range(1, len(Jittor_10000) + 1)]
plt.figure(figsize=(8, 5))
plt.plot(batchs, Jittor_10000, marker='o', label='HiSD128_Jittor', linestyle='-', linewidth=2)
plt.plot(batchs, Pytorch_10000, marker='s', label='HiSD128_Pytorch', linestyle='--', linewidth=2)

plt.title("Dis_Loss_Adv")
plt.xlabel("10KBatch")
plt.ylabel("Time (s)")
plt.xticks(batchs)  # 保证横坐标只显示整数Epoch
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
'''