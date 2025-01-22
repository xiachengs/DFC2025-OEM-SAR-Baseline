<div align="center">

<p><img src="docs/DFC2025_Logo.jpg"></p>
</div>

<div align="justify">
<p>The 2025 IEEE GRSS Data Fusion Contest, organized by the Image Analysis and Data Fusion Technical Committee, the University of Tokyo, RIKEN, and ETH Zurich, aims to foster the development of innovative solutions for all-weather land cover and building damage mapping using multimodal SAR and optical EO data at submeter resolution. The contest comprises <b>two tracks</b>: <em>land cover mapping</em> and <em>building damage mapping</em>.
This repository contains the baseline model for the <b>Track 1 challenge: All-Weather Land Cover Mapping</b>. Check out <a href="https://github.com/ChenHongruixuan/BRIGHT">here</a> for the<b> Track 2: All-weather building damage mapping</b> info.</p>
</div>
  
  
<div align="center">
	
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
</div>

## 2025 IEEE GRSS Data Fusion Contest Track 1: All-Weather Land Cover Mapping
<div align="justify">
<p>
The Track 1 challenge focuses on developing methods for land cover mapping in all weather conditions using SAR data. The training data consists of multimodal submeter-resolution optical and SAR images with 8-class land cover labels. These labels are pseudo-labels derived from optical images based on pre-trained models. During the evaluation phase, models will rely exclusively on SAR to ensure they perform well in real-world, all-weather scenarios. It aims to improve the accuracy of land cover mapping under varying environmental conditions, demonstrating the utility of SAR data in monitoring land cover. The mean intersection over union (mIoU) metric is used to evaluate the performance. 
</p> 
<p>

**Get involved! Check out the following links:** </br>
- Challenge webpage [2025 IEEE GRSS Data Fusion Contest](https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/?tab=data-fusion-contest)
- Dataset download [https://zenodo.org/records/14622048](https://zenodo.org/records/14622048) 
- Submission portal [https://codalab.lisn.upsaclay.fr/competitions/21121](https://codalab.lisn.upsaclay.fr/competitions/21121)
</p>
</div>

## Dataset
<div align="justify">

The dataset for the Track 1 challenge is OpenEarthMap-SAR. The [OpenEarthMap-SAR](https://zenodo.org/records/14622048) is a synthetic aperture radar dataset benchmark with [OpenEarthMap](https://open-earth-map.org/) optical data for global high-resolution land cover mapping. It consists of 5033 images at a 0.15–0.5m ground sampling distance covering 35 regions from Japan, France and the USA; and with partially manually annotated labels and fully pseudo labels of 8 land cover classes. A detailed description of the dataset can be found [here](https://zenodo.org/records/14622048), where it can also be downloaded. Below are examples of the OpenEarthMap-SAR dataset.

<p><img src="docs/DFC25_T1-min.png"></p>
</div>

## Baseline
<div align="justify">

The UNet architecture with EfficientNet-B4 encoder from the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file) GitHub repository is adopted as a baseline network.
The network was pretrained using the *train set* of the OpenEarthMap-SAR dataset. Download the pretrained weights from [here](https://drive.google.com/file/d/1Myd8b2KVFRuYVPyjB6EAv70OsNmjtgB9/view?usp=sharing).

## Usage
<div align="justify">

The repository structure consists of `dataset/train` and `dataset/test` for training and test data; a `pretrained/` directory for pretrained weights; and all source codes in the `source/`. The training and testing scripts, `train.py` and `test.py`, respectively, are at the root of the repo. The `docs` directory contains only GitHub page files.

To use the baseline code, first, clone the repository and change your directory into the `DFC2025-OEM-SAR-Baseline` folder. Then follow the steps below:</br>
1. Install all the requirements. `Python 3.8` was used in our experiments. Install the list of packages in the `requirements.txt` file using `pip install -r requirements.txt`.
2. Download the dataset from [here](https://zenodo.org/records/14622048) into the respective directories: `dataset/train` and `dataset/test`
3. Download the pretrained weights from [here](https://drive.google.com/file/d/1Myd8b2KVFRuYVPyjB6EAv70OsNmjtgB9/view?usp=sharing) into the `pretrained` directory

Test the model with the pretrained weights by running the script `test.py` as:
```bash
python test.py
```
To train the model, run `train.py` as:
```bash
python train.py
```
</div>

## Citation
<div align="justify">
For any scientific publication using this data, the following paper should be cited:
<pre style="white-space: pre-wrap; white-space: -moz-pre-wrap; white-space: -pre-wrap; white-space: -o-pre-wrap; word-wrap: break-word;">
@misc{xia_2025_oem_sar,
      title={OpenEarthMap-SAR: A Benchmark Synthetic Aperture Radar Dataset for Global High-Resolution Land Cover Mapping}, 
      author={Junshi Xia, Hongruixuan Chen, Clifford Broni-Bediako, Yimin Wei, Jian Song, and Naoto Yokoya},
      year={2025},
      note={arXiv:2501.10891},
      url={https://arxiv.org/abs/2501.10891}, 
}
</pre>
</div>

## Acknowledgements
<div align="justify">

We are most grateful to the authors of [Semantic Segmentation PyTorch](https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file) from which the baseline code is built on.
</div>
