<div align="center">

<p><img src="docs/DFC2025_Logo.jpg"></p>
</div>

<div align="justify">
<p>The 2025 IEEE GRSS Data Fusion Contest, organized by the Image Analysis and Data Fusion Technical Committee, the University of Tokyo, RIKEN, and ETH Zurich, aims to foster the development of innovative solutions for all-weather land cover and building damage mapping using multimodal SAR and optical EO data at submeter resolution. The contest comprises <b>two tracks</b>: <em>land cover mapping</em> and <em>building damage mapping</em>.</p>
	
<p>This repository contains the baseline model for the <b>Track 1 challenge: All-Weather Land Cover Mapping</b>.</p>
</div>
 
<div align="center">
	
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
</div>

## Track 1: All-Weather Land Cover Mapping
<div align="justify">
<p>
The Track 1 challenge focuses on developing methods for land cover mapping in all weather conditions using SAR data. The training data consists of multimodal submeter-resolution optical and SAR images with 8-class land cover labels. These labels are pseudo-labels derived from optical images based on pre-trained models. During the evaluation phase, models will rely exclusively on SAR to ensure they perform well in real-world, all-weather scenarios. It aims to improve the accuracy of land cover mapping under varying environmental conditions, demonstrating the utility of SAR data in monitoring land cover. The mean intersection over union (mIoU) metric is used to evaluate the performance.
</p> 
<p>
	
**Get involved! Check out the following links:** </br>
- Challenge webpage [2025 IEEE GRSS Data Fusion Contest](https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/?tab=data-fusion-contest)
- L3D-IVU Workshop @ CVPR 2024 Conference [https://sites.google.com/view/l3divu2024/overview](https://sites.google.com/view/l3divu2024/overview)
- Dataset Download [https://zenodo.org/records/11396874](https://zenodo.org/records/11396874) 
- ~Submission Portal [https://codalab.lisn.upsaclay.fr/competitions/17568](https://codalab.lisn.upsaclay.fr/competitions/17568).~
***The challenge is over, use the [post-challenge submission portal](https://codalab.lisn.upsaclay.fr/competitions/19210) to evaluate your predictions on the test set.***
</p>
</div>

## Context
<div align="justify">

 The motivation is to enable researchers to develop and benchmark learning methods for generalized few-shot semantic segmentation of high-resolution remote sensing imagery. The challenge is in two phases: the development phase is for developing and testing methods on a *valset*, and the evaluation phase is for tweaking and testing on a *testset* for final submission.

</div>

## Dataset
<div align="justify">

	
This dataset extends the original 8 semantic classes of the [OpenEarthmap](https://open-earth-map.org/) benchmark dataset to 15 classes for **5-shot** generalized few-shot semantic segmentation (GFSS) task with **4 novel classes** and **7 base classes**. It consists of only 408 samples from the original OpenEarthMap dataset. The 408 samples are also split into 258 as *trainset*, 50 as *valset*, and 100 as *testset*. The *valset* is used for the development phase of the challenge, and the *testset* is for the evaluation phase. Both sets consist of *support_set* and *query_set* for GFSS tasks. A detailed description of the dataset can be found [here](https://zenodo.org/records/10828417), where it can also be downloaded. Below are examples of novel classes in the *support_set* (first two columns), and base classes + novel classes in the *query_set* (last two columns).

<p><img src="docs/assets/img/fewshot-examples1.png"></p>
</div>

## Baseline
<div align="justify">

The PSPNet architecture with EfficientNet-B4 encoder from the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file) GitHub repository is adopted as a baseline network.
The network was pretrained using the *trainset* with the [Catalyst](https://catalyst-team.com/) library. Then, the state-of-the-art framework called [distilled information maximization](https://arxiv.org/abs/2211.14126) 
(DIaM) was adopted to perform the GFSS task. The code in this repository contains only the GFSS portion. As mentioned by the baseline authors, any pretrained model can be used with their framework. 
The code was adopted from [here](https://github.com/sinahmr/DIaM?tab=readme-ov-file). To run the code on the *valset*, simply clone this repository and change your directory into the `OEM-Fewshot-Challenge` folder which contains the code files. Then from a terminal, run the `test.sh` script. as:
```bash
bash test.sh 
```
The results of the baseline model on the *valset* are presented below. To reproduce the results, download the pretrained models from [here](https://drive.google.com/file/d/1eLjfUJ2ajAMkJKCsoJr-MGSSzZ-LqDbR/view?usp=sharing). 
Follow the instructions in the **Usage** section, then run the `test.sh` script as explained. 

<table align="center">
    <tr align="center">
        <td>Phase</td>
        <td>base mIoU</td> 
	<td>novel mIoU</td> 
	<td>Average base-novel mIoU</td>
        <td>Weighted base mIoU</td> 
	<td>Weighted novel mIoU</td>
	<td>Weighted-Sum base-novel mIoU</td>
    </tr>
    <tr align="center">
        <td>valset</td>
        <td> 29.48 </td> 
	<td> 03.18 </td> 
	<td> 16.33 </td> 
	<td> 11.79 </td> 
	<td> 1.91 </td> 
	<td> 13.70 </td> 
    </tr>
   <tr align="center">
	<td>testset</td>
        <td> 30.07 </td> 
	<td> 9.39 </td> 
	<td> 19.73 </td> 
	<td> 12.03 </td> 
	<td> 5.36 </td> 
	<td> 17.66 </td> 
    </tr>   
</table>
The weighted mIoUs are calculated using `0.4:0.6 => base:novel`. These weights are derived from the state-of-the-art results presented in the baseline paper.

</div>

## Usage
<div align="justify">

The repository structure consists of a configuration file that can be found in `config/`; data splits for each set in `data/`; and  all the codes for the GFSS task are in `src/`. The testing script `test.sh` is at the root of the repo.
The `docs` folder contains only GitHub page files.

To use the baseline code, you first need to clone the repository and change your directory into the `OEM-Fewshot-Challenge` folder. Then follow the steps below:</br>
1. Install all the requirements. `Python 3.9` was used in our experiments. Install the list of packages in the `requirements.txt` file using `pip install -r requirements.txt`.
2. Download the dataset from [here](https://zenodo.org/records/10591939) into a directory that you set in the config file `oem.yaml`
3. Download the pretrained weights from [here](https://drive.google.com/file/d/1Myd8b2KVFRuYVPyjB6EAv70OsNmjtgB9/view?usp=sharing) into a directory that you set in the config file `oem.yaml`
4. In the `oem.yaml` you need to set only the paths for the dataset and the pretrained weights. The other settings need not be changed to reproduce the results.
5. Test the model by running the `test.sh` script as mentioned in the **Baseline** section. The script will use the *support_set* to adapt and predict the segmentation maps of the *query_set*. After running the script, the results are provided in a `results` folder which contains a `.txt` file of the IoUs and mIoUs, and a `preds` and `targets` folder for the predicted and the targets maps, respectively.

You can pretrained your model using the *trainset* and any simple training scheme of your choice. The baseline paper used the [`train_base.py`](https://github.com/chunbolang/BAM/blob/main/train_base.py) script and base learner models of [BAM](https://github.com/chunbolang/BAM) (see the [baseline paper](https://github.com/sinahmr/DIaM?tab=readme-ov-file) for more info).
 
</div>

## Citation
<div align="justify">
For any scientific publication using this data, the following paper should be cited:
<pre style="white-space: pre-wrap; white-space: -moz-pre-wrap; white-space: -pre-wrap; white-space: -o-pre-wrap; word-wrap: break-word;">
@misc{bronibediako2024GFSS,
      title={Generalized Few-Shot Semantic Segmentation in Remote Sensing: Challenge and Benchmark}, 
      author={Clifford Broni-Bediako and Junshi Xia and Jian Song and Hongruixuan Chen and Mennatullah Siam and Naoto Yokoya},
      year={2024},
      note={arXiv:2409.11227},
      url={https://arxiv.org/abs/2409.11227}, 
}
</pre>
</div>

## Acknowledgements
<div align="justify">

We are most grateful to the authors of [DIaM](https://github.com/sinahmr/DIaM?tab=readme-ov-file), [Semantic Segmentation PyTorch](https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file), 
and [Catalyst](https://catalyst-team.com/) from which the baseline code is built on.
</div>
