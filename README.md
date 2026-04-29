# Query-MARFT: Query-Guided Multi-Agent Reinforcement Fine-Tuning for End-to-End Multi-Object Tracking
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-1.11+-yellow.svg)](https://NVIDIA.org/)


The code is still under update, and some bugs may be encountered during runtime.
---

## 📰 News
- **[2026-04-29]** Code released!

---

## 📖 Overview
Query-MARFT is a novel multi-object tracking framework built upon the Flexible Markov Game (Flex-MG) formulation. It introduces four specialized agents and a scene-adaptive DAG topology to achieve state-of-the-art performance on DanceTrack and MOT20 benchmarks.

<!-- 插入整体框架图，建议尺寸：1200x400 -->
![Method Overview](assets/GA.png)

---

## 📊 Results
### Quantitative Results
#### Our Results on DanceTrack and MOT20 Test Set
| Method | HOTA $\uparrow$ | IDF1 $\uparrow$ | MOTA $\uparrow$ | Link |
|--------|-----------------|-----------------|-----------------|-------------------|
| DanceTrack | 74.1 | 77.3 | 92.3 | [joshuawenhit](https://www.codabench.org/competitions/14885/#/results-tab) |
| MOT20 | 65.3 | 80.6 | 77.6 | [joshuawenhit](https://www.codabench.org/competitions/10050/#/results-tab) |

---

### Qualitative Results
#### High-Dynamic Crossover Scenario
<!-- 插入定性对比图，左：MOTRv2，右：Ours -->
![Qualitative DanceTrack](assets/figure-6.png)

---

## 🛠️ Installation
1. Clone this repository:
```bash
git clone https://github.com/JoshuaWenHIT/Query-MARFT.git
cd Query-MARFT
```

2. Create a conda environment and install dependencies:
```bash
conda create -n query-marft python=3.9
conda activate query-marft
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

3. Build the MSDeformAttn CUDA op (required by Deformable DETR)
```bash
cd models/ops
bash make.sh
python test.py   # optional sanity check
cd ../..
```
---

## 📂 Data Preparation
1. Download the datasets:
   - [DanceTrack](https://dancetrack.github.io/)
   - [MOT20](https://motchallenge.net/data/MOT20/)

2. Organize the data structure as follows:
```
data/
├── dancetrack/
│   ├── train/
│   ├── val/
│   └── test/
└── mot20/
    ├── train/
    └── test/
```

---

## 🚀 Getting Started
### Training (with 4 NVIDIA V100 GPUs)
```bash
bash train_marft.sh configs/Query-MARFT-v2.args
```

### Testing
```bash
bash ./tools/inference_marft.sh <your weights file path>
```

### Visualization Results
Download our visualization results on DanceTrack and MOT20 [Baidu Pan](https://pan.baidu.com/s/1k9GP7UvVSzqe0ZeFXE5POw?pwd=gkk8)
---

## 📜 License
This project is released under the Apache 2.0 license.

---

## 🙏 Acknowledgements
We thank the authors of 
[MOTRv2](https://github.com/megvii-research/MOTRv2) 
[MOTR](https://github.com/megvii-research/MOTR)
[Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
for their excellent codebase.

---

## 📧 Contact
For any questions, please contact [wenjiazheng@stu.hit.edu.cn](wenjiazheng@stu.hit.edu.cn).