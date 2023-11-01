# Recaptured Raw Screen Image and Video Demoiréing via Channel and Spatial Modulations

This repository contains official implementation of our NeurIPS 2023 paper "Recaptured Screen Image Demoiréing in Raw Domain", by Huanjing Yue, Yijia Cheng, Xin Liu, and Jingyu Yang.

<p align="center">
  <img width="1000" src="https://github.com/tju-chengyijia/VD_raw/blob/main/imgs/fw2.png">
</p>

## Paper
[arxiv](http://arxiv.org/abs/2310.20332)

## Demo Video

https://github.com/tju-chengyijia/VD_raw/blob/main/demo/nips_2023.mp4 <br>

## Dataset

<p align="center">
  <img width="1000" src="https://github.com/tju-chengyijia/VD_raw/blob/main/imgs/dataset_show.png">
</p>

You can download our dataset from ......

We provide ground truth frames and moiré frames in raw domain and sRGB domain respectively, which are placed in four folders gt_raw, gt_rgb, moire_raw and moire_rgb. The ground truth raw image is actually pseudo ground truth. The users can regenerate them by utilizing other RGB to raw inversing algorithms. Our raw domain data is stored in npz format, including raw data, black level value, white level value and white balance value.

#### Copyright ####

The dataset is available for the academic purpose only. Any researcher who uses the dataset should obey the licence as below:

All of the Dataset are copyright by [Intelligent Imaging and Reconstruction Laboratory](http://tju.iirlab.org/doku.php), Tianjin University and published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License. This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.

## Code

### Dependencies and Installation

...

### Prepare

...

### Test

...

### Train

...

## Results

### Video Results

 <div align=center><img src="https://github.com/tju-chengyijia/VD_raw/blob/main/imgs/sota_video.png"></div><br>

### Image Results

<div align=center><img src="https://github.com/tju-chengyijia/VD_raw/blob/main/imgs/sota_img.png"></div><br>

## Citation

...

## Acknowledgement

Our work and implementations are inspired by following projects:<br/>
[EDVR](https://github.com/xinntao/EDVR)<br/>
[VideoDemoireing](https://github.com/CVMI-Lab/VideoDemoireing)<br/>

