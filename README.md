# SIMSUF
SIMSUF is a framework for multimodal sentiment analysis task.
## Introdction
Multimodal sentiment analysis remains a big challenge due to the lack of effective fusion solutions. An effective fusion is expected to obtain the correct semantic representation for all modalities, and simultaneously thoroughly explore the contribution of each modality. In this paper, weCancel changes propose a SIngle-Modal guided SUpplementary Fusion (SIMSUF) approach to perform effective multimodal fusion for sentiment analysis. The SIMSUF is composed of three major components, the modal-guided supplementary module, the modality enhancement module, and a modal-guided fusion module. The modal-guided supplementary module realizes main-line modality determination by estimating mutual dependence between every two modalities, then the main-line modality is adopted to supplement other modalities for representative feature learning.
To furthest explore the modality contribution, we propose a two-branch modality enhancement module, where a common modality enhancement branch is set to learn a common representation distribution for multiple modalities, and simultaneously a specific modality enhancement branch is presented to perform semantic difference enhancement and distribution difference enhancement for each modality. Finally, we design a modal-guided fusion module to integrate multimodal representations for sentiment analysis. Extensive experiments are evaluated on CMU-MOSEI and CMU-MOSI datasets, and experiment results certify that our approach is superior to the state-of-the-art approaches.
## Architecture
We propose a SIngle-Modal guided SUpplementary Fusion (SIMSUF) approach to realize effective multimodal fusion. The SIMSUF comprises three major modules: the modal-guided supplementary module, the modality enhancement module, and the modal-guided fusion module. The modal-guided supplementary module is proposed to automatically select the most important modality as the main-line modality, and use it to supplement other modalities, obtaining semantic supplementary representations. The main-line modality determination relies on calculating mutual dependence coefficients between every two modalities and sorting them to find the maximum one. The modality enhancement module is presented to furthest explore the contribution of each modality. It is composed of two enhancement branches, where a common modality enhancement branch is set to generate common-distribution representations by drawing close multiple-modality distributions, and simultaneously a specific modality enhancement branch is presented to perform semantic difference enhancement and distribution difference enhancement for each modality. Finally, a modal-guided fusion module is designed to fuse multimodal features, where the main-line modality is used as a baseline for interactive fusion with other modalities.
## Requirements
Our code is written by Python, based on Pytorch (Version ≥ 1.4)
## Datasets
[CMU_MOSEI](https://aclanthology.org/P18-1208.pdf)/[CMU_MOSI](https://ieeexplore.ieee.org/document/7742221)

The SIMSUF uses feature files that are organized as follows:
``` 
{
    "train": {
        "raw_text": [],              # raw text
        "audio": [],                 # audio feature
        "vision": [],                # video feature
        "id": [],                    # [video_id$_$clip_id, ..., ...]
        "text": [],                  # bert feature
        "text_bert": [],             # word ids for bert
        "audio_lengths": [],         # audio feature lenth(over time) for every sample
        "vision_lengths": [],        # same as audio_lengths
        "annotations": [],           # strings
        "classification_labels": [], # Negative(0), Neutral(1), Positive(2). Deprecated in v_2.0
        "regression_labels": []      # Negative(<0), Neutral(0), Positive(>0)
    },
    "valid": {***},                  # same as "train"
    "test": {***},                   # same as "train"
}
```
## Results
We perform our methods in two datasets for multimodal sentiment analysis. We use the same metric set that has been consistently presented and compared before. Mean absolute error (MAE) is the average mean difference value between predicted values and truth values. Pearson correlation (Corr) measures the degree of prediction skew. Seven-class classification accuracy (Acc-7) indicates the proportion of predictions that correctly fall into the same interval of seven intervals between -3 and +3 as the corresponding truths. Binary classification accuracy (Acc-2) and F1 score are computed for positive/negative and non-negative/negative classification results. The results are listed in table. 

<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td width="276" colspan="5" valign="top" style="width:207.3pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MOSEI</span></p>
  </td>
  <td width="277" colspan="5" valign="top" style="width:207.5pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MOSI</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td width="55" valign="top" style="width:41.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MSE</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">CORR</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ACC-7</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ACC-2</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">F1</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">MSE</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">CORR</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ACC-7</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">ACC-2</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">F1</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;mso-yfti-lastrow:yes">
  <td width="55" valign="top" style="width:41.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.529</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.772</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">53.68</span></p>
  </td>
  <td width="55" valign="top" style="width:41.45pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.23</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.12</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.709</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.802</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">45.72</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">86.08</span></p>
  </td>
  <td width="55" valign="top" style="width:41.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">85.98</span></p>
  </td>
 </tr>
</tbody></table>

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
