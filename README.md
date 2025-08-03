# HiSD-Jittor
HiSD implementation based on Jittor  

The examples in Samples are the results of sampling after training in the HiSD official experimental architecture. These images may have undergone sample enhancement processing such as color perturbation and rotation.

[CheckPoint For Jittor Version on 128×128 images](https://drive.google.com/file/d/1UD7pFR8yMLg6bwfLGcWK8EN1wNoSF5NI/view?usp=sharing)  

[CheckPoint For Pytorch Version on 128×128 images](https://drive.google.com/file/d/1AIye0Gs16cepKiyLaalzlCcJNzKEoot5/view?usp=sharing)

## 🚀 Quick Start

This repository has simplified the training code to make it easily executable within an IDE. Please review and follow the steps below carefully before starting:

### 1. Environment Setup

- **Create a new environment** specifically for [Jittor](https://github.com/Jittor/jittor) installation.  
  **Avoid installing it alongside PyTorch** to prevent potential conflicts with NumPy dependencies and unexpected errors.
- **Use Linux** whenever possible. Jittor currently has **known compatibility issues with Windows**.  
  *(If you read Chinese, additional troubleshooting details can be found in `Experiment_log/ErrorReport.pdf`.)*

### 2. Configuration

- Configuration files in YAML format are provided from the official HiSD implementation, tailored for **128×128** and **256×256** image resolutions.
- You’ll need to adjust parameters such as:
  - `batch_size`
  - total batch size
  - `num_workers`
- These settings should be customized according to your **hardware and training requirements**.
- **Set the GPU device** in `Device_set.py` before running.

### 3. Data Preprocessing

- Use `Preprocess.py` in the `Data_Preprocess` directory to generate the `Samples` file.
- This script will create several text files that define image indexes for **each tag and its corresponding attributes**.
- You can modify the script to specify:
  - Which tags/attributes to include
  - Which irrelevant labels to ignore
- ⚠️ Make sure the tag and attribute information stays **synchronized with your YAML config files** to avoid inconsistencies during training.

### 4. Entry Point

- The main training script is `train.py`. Run this file to start the training process.


## Performance Table  
![Performance Table](Experiment_log/Experiment.png)

## Time Lines  
![Time Lines](Experiment_log/Time.png)

## Dis_Loss_Adv  
![Dis_Loss_Adv](Experiment_log/Dis_Loss_Adv.png)

In order to facilitate the observation of the gap, all Gen-Losses are drawn from the 20Kth batch.

This table shows the average of each Gen-Loss on the first 10K batches  
|  | Gen_Loss_Adv | Gen_Loss_Sty |Gen_Loss_Rec|
|--------|--------|--------|--------|
| Pytorch  | 21.046  | 2.236  |1.632|
| Jittor | 582.015  | 13.116  |3.017|

## Gen_Loss_Adv  
![Gen_Loss_Adv](Experiment_log/Gen_Loss_Adv.png)

## Gen_Loss_Sty  
![Dis_Loss_Adv](Experiment_log/Gen_Loss_Sty.png)

## Gen_Loss_Rec  
![Dis_Loss_Adv](Experiment_log/Gen_Loss_Rec.png)
