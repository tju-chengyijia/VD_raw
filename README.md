# Recaptured Raw Screen Image and Video Demoiréing via Channel and Spatial Modulations

This repository contains official implementation of our NeurIPS 2023 paper "Recaptured Screen Image Demoiréing in Raw Domain", by Huanjing Yue, Yijia Cheng, Xin Liu, and Jingyu Yang.

<p align="center">
  <img width="1000" src="https://github.com/tju-chengyijia/VD_raw/blob/main/imgs/fw2.png">
</p>

## Paper
[arxiv](http://arxiv.org/abs/2310.20332)

## Demo Video

[video](https://github.com/tju-chengyijia/VD_raw/blob/main/demo/nips_2023.mp4) <br>

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

- Ubuntu 20.04.5 LTS
- Python 3.8.10
- NVIDIA GPU + CUDA 11.7
- Pytorch-GPU 1.10.0

### Prepare

- Download our dataset and place them in data folder according to the TRAIN_DATASET and TEST_DATASET in [config files](https://github.com/tju-chengyijia/VD_raw/tree/main/config).
- Please note to set the paths for variables MODEL_DIR, fhd_pretrain, VISUALS_DIR, NETS_DIR, VAL_RESULT_DIR, TEST_RESULT_DIR, etc. in the config file appropriately.
- Download pre-trained model. Our baseline model is placed in [this project](https://github.com/tju-chengyijia/VD_raw/blob/main/model_dir_cwb/nets/checkpoint_000040.tar). Please download our complete model in [LINK] and place it [here](https://github.com/tju-chengyijia/VD_raw/tree/main/model_dir_depth2/nets).
- Install Basicsr-GPU. `python setup.py develop` For more information, please refer to [LINK](https://github.com/xinntao/EDVR).

### Test

- Test pretrained model on our testset.
```
python test.py --config=./config/vdm_depth.yaml
```

### Train

- Stage 1: Train the baseline network.
```
python train.py --config=./config/vdm_baseline.yaml
```

- Stage 2: Train the complete network.
```
python train_depth.py --config=./config/vdm_depth.yaml
```

## Results

### Video Results

 <div align=center><img src="https://github.com/tju-chengyijia/VD_raw/blob/main/imgs/sota_video.png"></div><br>

### Image Results

<div align=center><img src="https://github.com/tju-chengyijia/VD_raw/blob/main/imgs/sota_img.png"></div><br>

## Citation

If you find our dataset or code helpful in your research or work, please cite our paper:

```
@article{yue2023recaptured,
  title={Recaptured Raw Screen Image and Video Demoir$\backslash$'eing via Channel and Spatial Modulations},
  author={Yue, Huanjing and Cheng, Yijia and Liu, Xin and Yang, Jingyu},
  journal={arXiv preprint arXiv:2310.20332},
  year={2023}
}
```

## Acknowledgement

Our work and implementations are inspired by following projects:<br/>
[EDVR](https://github.com/xinntao/EDVR)<br/>
[VideoDemoireing](https://github.com/CVMI-Lab/VideoDemoireing)<br/>

