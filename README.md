# Achelous

<div  align="center">    
  <img src="icons/Achelous.png" width = "300" height = "300" alt="Achelous" align=center />
</div>

## Achelous: A Fast Unified Water-surface Panoptic Perception Framework based on Fusion of Monocular Camera and 4D mmWave Radar

<div  align="center">    
  <img src="icons/Achelous_str.jpg" alt="ach" align=center />
</div>

---
## The Power of Achelous
Based on a monocular camera and a 4D mmWave radar,
- [x] Object detection (anchor-free)
- [x] Object semantic segmentation
- [x] Drivable area segmentation
- [x] Waterline segmentation
- [x] Point cloud semantic segmentation
- [ ] Object tracking
- [ ] Instance segmentation
- [ ] Point cloud instance segmentation


<div  align="center">    
  <img src="icons/prediction_results.jpg" alt="pre1" align=center />
</div>

<div  align="center">    
  <img src="icons/compare_yolop.jpg" alt="pre2" align=center />
</div>

***


### Backbone, FPN and Head
| ViT-based Backbones | Lightweight Dual FPNs | Detection heads |
| ---------| --------- | --------- |
| [√] EdgeNeXt (EN) | [√] Ghost-Dual-FPN | [√] YOLOX DetectX Head |
| [√] EdgeViT (EV) | [√] CSP-Dual-FPN | [ ] YOLOv6 Efficient decoupled head |
| [√] EfficientFormer (EF) |   | [ ] YOLOv7 IDetect Head |
| [ ] EfficientFormer V2 (EF2) |   | [ ] YOLOv7 IAuxDetect Head |
| [√] MobileViT (MV) |   |  [ ] FCOS Head |
| [ ] MobileViT V2 (MV2) |   |  |
| [ ] MobileViT V3 (MV3) |   |  |


### Label Assignment Strategy, Loss Function and NMS
| Label Assignment Strategy | Loss Function | NMS |
| ---------| --------- | --------- |
| [√] SimOTA | [√] ComputeLoss(X) | [√] NMS |
| [ ] FreeAnchor | [ ] ComputeLossAuxOTA(v7) | [ ] Soft-NMS |
| [ ] AutoAssign | [ ] ComputeLossOTA(v7)  |  |
|  | [√] CIoU  |  |
|  | [√] GIoU  |  |
|  | [√] Focal  |  |
|  | [√] Dice  |  |


### Point Cloud Models
- [x] PointNet (PN)
- [x] PointNet++  (PN2)
- [ ] Point-NN   (PNN)



---
### Performances
input size: 320 ×320 \
point cloud number per sample: 512 \
Three model size: S0, S1, S2 \
⬇ \
S0 channels: [32, 48, 96, 176] \
S1 channels: [32, 48, 120, 224] \
S2 channels: [32, 64, 144, 288]

| Methods | Sensors | Task Num | Params (M) | FLOPs (G) | FPSe | FPSg |
| :---------:| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| EN-CDF-PN-S0| 2 | 5 | 3.59 | 5.38 | 17.5 | 59.8 |
| EN-GDF-PN-S0 | 2 | 5 | 3.55 | 2.76 | 17.8 | 61.3 |
| EN-CDF-PN2-S0 | 2 | 5 | 3.69 | 5.42 | 15.2 | 56.5 |
| EN-GDF-PN2-S0 | 2 | 5 | 3.64 | 2.84 | 14.8 | 57.7 |
| EF-GDF-PN-S0 | 2 | 5 | 5.48 | 3.41 | 17.3 | 50.6 |
| EV-GDF-PN-S0 | 2 | 5 | 3.79 | 2.89 | 16.4 | 54.9 |
| MV-GDF-PN-S0 | 2 | 5 | 3.49 | 3.04 | 16.0 | 53.7 | 
| EN-GDF-PN-S1 | 2 | 5 | 5.18 | 3.66 | 16.6 | 59.7 |
| EF-GDF-PN-S1 | 2 | 5 | 8.07 | 4.52 | 16.6 | 46.8 |
| EV-GDF-PN-S1 | 2 | 5 | 4.14 | 3.16 | 16.7 | 56.6 |
| MV-GDF-PN-S1 | 2 | 5 | 4.67 | 4.29 | 15.8 | 55.8 | 
| EN-GDF-PN-S2 | 2 | 5 | 6.90 | 4.59 | 16.1 | 58.1 |
| EF-GDF-PN-S2 | 2 | 5 | 14.64 | 7.13 | 13.5 | 39.3 |
| EV-GDF-PN-S2 | 2 | 5 | 8.28 | 5.19 | 14.7 | 47.1 |
| MV-GDF-PN-S2 | 2 | 5 | 7.18 | 6.02 | 15.6 | 52.7 | 


| Methods | mAP50-95 | mAP50 | AR50-95 | mIoU-t | mIoU-d | mIoU-w | mIoU-pc |
| :---------:| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| EN-CDF-PN-S0| 37.2 | 66.3 | 43.1 | 68.1 | 99.0 | 69.4 | 57.1 |
| EN-GDF-PN-S0 | 37.5 | 66.9 | 44.6 | 69.1 | 99.5 | 69.3 | 57.8 | 
| EN-CDF-PN2-S0 | 37.3 | 66.3 | 43.0 | 68.4 | 99.5 | 68.9 | 60.2 |
| EN-GDF-PN2-S0 | 37.7 | 68.1 | 45.0 | 67.2 | 99.4 | 67.3 | 59.6 |
| EF-GDF-PN-S0 | 37.4 | 66.5 | 43.4 | 68.7 | 99.6 | 66.6 | 59.4 | 
| EV-GDF-PN-S0 | 38.8 | 67.3 | 42.3 | 69.8 | 99.6 | 70.6 | 58.0 |
| MV-GDF-PN-S0 | 41.5 | 71.3 | 45.6 | 70.6 | 99.5 | 68.8 | 58.9 |
| EN-GDF-PN-S1 | 41.3 | 70.8 | 45.5 | 67.4 | 99.4 | 69.3 | 58.8 |
| EF-GDF-PN-S1 | 40.0 | 70.2 | 43.8 | 68.2 | 99.3 | 68.7 | 58.2 |
| EV-GDF-PN-S1 | 41.0 | 70.7 | 45.9 | 70.1 | 99.4 | 67.9 | 59.2 |
| MV-GDF-PN-S1 | 43.1 | 75.8 | 47.2 | 71.9 | 99.5 | 69.2 | 59.1 |
| EN-GDF-PN-S2 | 40.8 | 70.9 | 44.4 | 69.6 | 99.3 | 71.1 | 59.0 |
| EF-GDF-PN-S2 | 40.5 | 70.8 | 44.5 | 70.3 | 99.1 | 71.7 | 58.4 |
| EV-GDF-PN-S2 | 40.3 | 69.7 | 43.8 | 74.1 | 99.5 | 67.9 | 58.3 |
| MV-GDF-PN-S2 | 45.0 | 79.4 | 48.8 | 73.8 | 99.6 | 70.8 | 58.5 |

FPSe: FPS on Jetson AGX Xavier \
FPSg: FPS on RTX A4000 GPU \
mIoU-t: mIoU of targets \
mIoU-d: mIoU of drivable area \
mIoU-w: mIoU of waterline segmentation \
mIoU-pc: mIoU of point clouds


## Implementation

### Prepare environments
> git clone  https://github.com/GuanRunwei/Achelous.git \
> conda create -n achelous python=3.7 \
> conda activate achelous \
> pip install -r requirements.txt

### Prepare datasets
<font color=red>Pay attention: images and radar files are both necessary in Achelous.</font>

- Prepare a txt file saving classname for object detection under the library model_data/ (for example, we can name it classes.txt): 
  > pier \
  > buoy \
  > sailor \
  > ship \
  > boat \
  > vessel \
  > kayak
  
- Change annotations of object detection as the following (name it as 2007_train.txt and place it in the root library in Achelous, the same level with train.py):
  > E:/Big_Datasets/water_surface/all-1114/all/VOCdevkit/VOC2007/JPEGImages/1664091257.87023.jpg 1131,430,1152,473,0 920,425,937,451,0 \
  > E:/Big_Datasets/water_surface/all-1114/all/VOCdevkit/VOC2007/JPEGImages/1664091270.40352.jpg 249,374,266,424,0 396,381,412,431,0

  Each line denotes the image url, x_min1, y_min1, x_max1, y_max1, class_index1 x_min2, y_min2, x_max2, y_max2, class_index2 ……

  And the same operations to the validation files or test files (you can name them as 2007_val.txt)
  
  Define them in train.py
  
  > train_annotation_path = '2007_train.txt' \
    val_annotation_path = '2007_val.txt'
    
  To complete the detection task, you also need to generate the npz files, which contains the radar feature maps of each frame, the detail is in radar_feature_map_generate.ipynb. 
  You should keep the consistency of image and radar feature map size. (e.g. image->320\*320, so radar feature map is 320\*320). After that, define the file path of radar feature maps in train.py
  > radar_file_path = " "
  
  
- Define file paths of images, segmentation of targets and lanes in train.py, and change the class number of semantic segmentation (class number + 1, 1 is the background)
  > jpg_path = " " \
    se_seg_path = " "  \
    wl_seg_path = " " \
    num_classes_seg = 9
    
- If you want to segment radar point clouds, define the path of point cloud files with csv format, and change the class number of point segmentation. You could also change the name fields of features and label of point cloud.

  > radar_pc_seg_path = " " \
    radar_pc_seg_features = ['x', 'y', 'z', 'comp_velocity', 'rcs'] \
    radar_pc_seg_label = ['label']
    radar_pc_classes = 8
    
   
   
### Train
1. Open train.py to see the collections of hyper-parameters
2. For example, if you want to train Achelous with MobileViT (backbone), Ghost-Dual-FPN (neck), Nano detection head, 
   , PointNet as the point cloud segmentation head. The model size is S0. The input resolution is 320. Batchsize is 32 and epoch is 100. The initial learning rate is 0.03. Point number of each frame is 512.
   > python train.py --cuda True --fp16 True --is_pc True --backbone mv --neck gdf --nd True --phi S0 --resolution 320 --bs 32 --epoch 100 --lr_init 0.03 --pc_num 512 --pc_model pn



### Test
1. Change configrations in achelous.py
2. See annotations in predict.py and run (including prediction, export, heatmap visualization and export onnx)
---



## Acknowledgement
https://github.com/bubbliiiing

## Citation
```BibTeX
@misc{guan2023achelous,
      title={Achelous: A Fast Unified Water-surface Panoptic Perception Framework based on Fusion of Monocular Camera and 4D mmWave Radar}, 
      author={Runwei Guan and Shanliang Yao and Xiaohui Zhu and Ka Lok Man and Eng Gee Lim and Jeremy Smith and Yong Yue and Yutao Yue},
      year={2023},
      eprint={2307.07102},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
