# SIMSUF
SIMSUF is a framework for multimodal sentiment analysis task.
## Introdction
Multimodal sentiment analysis remains a big challenge due to the lack of effective fusion solutions. An effective fusion is expected to obtain the correct semantic representation for all modalities, and simultaneously thoroughly explore the contribution of each modality. In this paper, weCancel changes propose a SIngle-Modal guided SUpplementary Fusion (SIMSUF) approach to perform effective multimodal fusion for sentiment analysis. The SIMSUF is composed of three major components, the modal-guided supplementary module, the modality enhancement module, and a modal-guided fusion module. The modal-guided supplementary module realizes main-line modality determination by estimating mutual dependence between every two modalities, then the main-line modality is adopted to supplement other modalities for representative feature learning.
To furthest explore the modality contribution, we propose a two-branch modality enhancement module, where a common modality enhancement branch is set to learn a common representation distribution for multiple modalities, and simultaneously a specific modality enhancement branch is presented to perform semantic difference enhancement and distribution difference enhancement for each modality. Finally, we design a modal-guided fusion module to integrate multimodal representations for sentiment analysis. Extensive experiments are evaluated on CMU-MOSEI and CMU-MOSI datasets, and experiment results certify that our approach is superior to the state-of-the-art approaches.
## Architecture
![](https://github.com/HumanCenteredUndestanding/SIMSUF/blob/main/TQ7BRBRQ%5D4EAGNB2S66U9TT.png)
## Requirements
Our code is written by Python, based on Pytorch (Version ≥ 1.4), with NVIDIA GTX 2080Ti (~12G of memory)
## Datasets
[CMU_MOSEI](https://aclanthology.org/P18-1208.pdf)/[CMU_MOSI](https://ieeexplore.ieee.org/document/7742221)
## Usage
1.Clone the repository
``` 
git clone https://github.com/HumanCenteredUndestanding/SIMSUF.git
```
2.Download dataset config and put the split dataset folders into $ROOT_DIR/datasets/. The folders are arranged like this:
```
├datasets         
    
    ├── MOSEI
    │   ├── mosei_data_noalign.pkl    
    │   ├── MOSEI-label
    
    ├── MOSI    
    │   ├── mosi_data_noalign.pkl    
    │   ├── MOSI-label  
 ```
 3.Train the model
  ```
cd src
python main.py
  ```
