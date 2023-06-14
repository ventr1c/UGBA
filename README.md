# UGBA: Unnoticeable Backdoor Attack on Graph Neural Networks
An official PyTorch implementation of "Unnoticeable Backdoor Attack on Graph Neural Networks" (WWW 2023). [[paper]](https://arxiv.org/abs/2303.01263) If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{dai2023unnoticeable,
  title={Unnoticeable Backdoor Attacks on Graph Neural Networks},
  author={Dai, Enyan and Lin, Minhua and Zhang, Xiang and Wang, Suhang},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={2263--2273},
  year={2023}
}
```

## Content
- [Unnoticeable Backdoor Attack on Graph Neural Networks](#unnoticeable-backdoor-attack-on-graph-neural-networks)
  - [Content](#content)
  - [1. Overview](#1-overviews)
  - [2. Requirements](#2-requirements)
  - [3. UGBA](#3-ugba)
    - [Abstract](#abstract)
    - [Reproduce the Results](#reproduce-the-results)
  - [4. Compared Methods (to test)](#4-compared-methods-to-test)
    - [Compared with Graph Backdoor Attack Methods](#compared-with-graph-backdoor-attack-methods)
      - [SBA-Gen](#sba-gen)
      - [SBA-Samp](#sba-samp)
      - [GTA](#gta)
    - [Compared with Graph Injection Evasion Attack Methods](#compared-with-graph-injection-evasion-attack-methods)
      - [TDGIA](#tdgia)
      - [AGIA](#agia)
    - [Comparing with Defense Methods](#comparing-with-defense-methods)
      - [Robust GCN](#robust-gcn)
      - [GNNGuard](#gnnguard)
  - [5. Dataset](#5-dataset)

## 1. Overview
* `./models`: This directory contains the model of UGBA.
* `./data`: The directory contains the original datasets used in the experiments
* `./logs`: The directory contains the log of the experimental results 
* `./script`: It contains the scripts to reproduce the major reuslts of our paper.
* `./selected_nodes`: This directory contains the selected poisoned nodes for each dataset.
* `./baseline_atk.py` The framework of baseline node injection attack (i.e., [TDGIA](https://arxiv.org/abs/2106.06663), [AGIA](https://arxiv.org/abs/2202.08057)). 
* `./run_adaptive.py`: The program to run our UGBA attack.
* `./run_clean.py`: The program to run GNNs on clean graph under three defense settings to get clean accuracy.
* `./run_GTA.py`: The program to run baseline GTA attack from Xi, Zhaohan, et al. ["Graph Backdoor"](https://arxiv.org/abs/2006.11890).
* `./run_NodeInjectionAtk.py`: The program to run baseline node injection attack (i.e., [TDGIA](https://arxiv.org/abs/2106.06663), [AGIA](https://arxiv.org/abs/2202.08057)). 
* `./run_SBA.py`: The program to run baseline SBA-Samp and its variant SBA-Gen from Xi, Zhaohan, et al. ["Backdoor Attacks to Graph Neural Networks"](https://arxiv.org/abs/2006.11165).

## 2. Requirements
```
python==3.8.13
torch==1.12.1
torch-geometric==2.1.0
numpy==1.22.4
scipy==1.7.3
scikit-learn==1.1.1
scikit-learn-extra==0.2.0
```
The packages can be installed by directly run the commands in [`install.sh`](https://github.com/ventr1c/UGBA/blob/main/install.sh) by
```
bash install.sh
```

## 3. UGBA

### Abstract
Graph Neural Networks (GNNs) have achieved promising results in various tasks such as node classification and graph classification. Recent studies find that GNNs are vulnerable to adversarial attacks. However, effective backdoor attacks on graphs are still an open problem. In particular, backdoor attack poisons the graph by attaching triggers and the target class label to a set of nodes in the training graph. The backdoored GNNs trained on the poisoned graph will then be misled to predict test nodes to target class once attached with triggers. Though there are some initial efforts in graph backdoor attacks, our empirical analysis shows that they may require a large attack budget for effective backdoor attacks and the injected triggers can be easily detected and pruned. Therefore, in this paper, we study a novel problem of unnoticeable graph backdoor attacks with limited attack budget. To fully utilize the attack budget, we propose to deliberately select the nodes to inject triggers and target class labels in the poisoning phase. An adaptive trigger generator is deployed to obtain effective triggers that are difficult to be noticed. Extensive experiments on real-world datasets against various defense strategies demonstrate the effectiveness of our proposed method in conducting effective unnoticeable backdoor attacks.

### Reproduce the Results
The hyper-parameters settings for the datasets are included in [`train_UGBA.sh`](https://github.com/ventr1c/UGBA/blob/main/script/train_UGBA.sh) To reproduce the performance reported in the paper, you can run the bash file:
```
bash script\train_UGBA.sh
```
To get the results of Baselines, you can run the bash file:
```
bash script\train_baseline.sh
```
To see the reproduce experimental results, please check the logs in [`./logs`](https://github.com/ventr1c/UGBA/tree/main/logs)
## 4. Compared Methods (to test)
### Compared with Graph Backdoor Attack Methods
#### SBA-Samp
From Zhang, Zaixi, et al. "Backdoor Attacks to Graph Neural Networks" [[paper](https://arxiv.org/abs/2006.11165), [code](https://github.com/zaixizhang/graphbackdoor)].
#### SBA-Gen
This is a variant of SBA-Samp, which uses generated features for trigger nodes. Features are from a Gaussian distribution whose mean and variance is computed from real nodes.
#### GTA
From Xi, Zhaohan, et al. "Graph Backdoor" [[paper](https://arxiv.org/abs/2006.11890), [code](https://github.com/HarrialX/GraphBackdoor)].
### Compared with Graph Injection Evasion Attack Methods
#### TDGIA
From Zou, Xu, et al. "TDGIA: Effective Injection Attacks on Graph Neural Networks." [[paper](https://arxiv.org/abs/2106.06663), [code](https://github.com/THUDM/tdgia)].
#### AGIA 
From Chen, Yongqiang, et al. "Understanding and Improving Graph Injection Attack by Promoting Unnoticeability" [[paper](https://arxiv.org/abs/2202.08057), [code](https://github.com/LFhase/GIA-HAO/blob/master/attacks/agia.py)].
### Comparing with Defense Methods
#### Robust GCN
From Zhu, Dingyuan, et al. "Robust graph convolutional networks against adversarial attacks" [[paper](https://dl.acm.org/doi/10.1145/3292500.3330851), [code](https://github.com/ZW-ZHANG/RobustGCN)].
#### GNNGuard
From Zhang, Xiang, et al. "GNNGuard: Defending Graph Neural Networks against Adversarial Attacks" [[paper](https://arxiv.org/abs/2006.08149), [code](https://github.com/mims-harvard/GNNGuard)].
## 5. Dataset
The experiments are conducted on four public real-world datasets, i.e., Cora, Pubmed, Flickr and OGB-Arxiv, which can be automatically downloaded to `./data` through torch-geometric API.

