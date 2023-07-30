## Remote Sensing Image Instance Segmentation Network with Transformer and Multi-scale Feature Representation

The goal of remote sensing image (RSI) instance segmentation is to perform instance-level semantic parsing of its contents. Aside from classifying and locating regions of interest (RoI), it also requires assigning finer pixel-wise annotations to objects. However, RSI often suffers from cluttered backgrounds, variable object scales, and complex object edge contours, making the instance segmentation task more challenging. In this work, we analytically customize an instance segmentation model that is more suitable for RSI. Specifically, we propose three novel modules for a region-based instance segmentation framework, namely Channel-Spatial Attention Module (CSA), Multi-Scale Aware Module (MSA), and Semantic Relation Learning Module (SRL). Among them, feature calibration performed by CSA can alleviate the semantic gap between low-level features and high-level semantics in both channel and spatial dimensions. Inheriting the capabilities of both the convolutional neural network (CNN) and the Transformer, SRL can help the network integrate both neighborhood features and long-range dependencies for instance semantic prediction. The MSA module designs a cascaded residual structure with different receptive fields to model the scale variation of objects in RSI. Experimental results on challenging ISAID, NWPU VHR-10, SSDD, BITTC and HRSID datasets demonstrate the superiority of our method, achieving mask APs of $40.2\%$, $68.2\%$, $68.4\%$, $50.4\%$ and $55.8\%$ respectively. . 

The details of this project are presented in the following paper:

[Remote Sensing Image Instance Segmentation Network with Transformer and Multi-scale Feature Representation [ESWA'23]](https://www.sciencedirect.com/science/article/abs/pii/S0957417423015099) 



## Usage 
### Setup 
```
MMdetection mmdet version=2.23.0

After installing mmdetection, put the files in the /cascade-swin/code path into the corresponding location for replacement, put cascade_mask_rcnn_r50_fpn_swin.py in the /configs/_base_/model/ path, and put cascade_mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py Put it under the /configs/swin/ path.

```
### Dataset 
Download the training and test datasets and move them into `./dataset/`, see [isaid](https://captain-whu.github.io/iSAID/dataset.html).



### Well-trained model 
[Baidu Drive](https://pan.baidu.com/s/1MTbk0V63KVhLbM2dSwJs7g) [code:waeo].

##  License
The source code is free for research and education use only. Any commercial use should get formal permission first.

Any advice is welcomed ^.^; please get in touch with **sylgzwc@163.com** or pull the question.

## Acknowledgement
Thanks [MMdetection](https://github.com/open-mmlab/mmdetection) for serving as building blocks of our work.

## Citation

If you find our work/code interesting, welcome to cite our paper >^.^<

```bibtex
@article{YE2023121007,
title = {Remote sensing image instance segmentation network with transformer and multi-scale feature representation},
journal = {Expert Systems with Applications},
volume = {234},
pages = {121007},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.121007},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423015099},
author = {Wenhui Ye and Wei Zhang and Weimin Lei and Wenchao Zhang and Xinyi Chen and Yanwen Wang}
}
```
