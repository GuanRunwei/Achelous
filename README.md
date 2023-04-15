# Achelous

<div  align="center">    
  <img src="icons/Achelous.png" width = "300" height = "300" alt="Achelous" align=center />
</div>

Achelous: A Fast Unified Water-surface Panoptic Perception Framework based on Fusion of Monocular Camera and 4D mmWave Radar
---
## Achelous's ability
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
  <img src="icons/prediction_results.jpg" width = "500px" alt="pre1" align=center />
</div>

<div  align="center">    
  <img src="icons/compare_yolop.jpg" width = "500px" alt="pre2" align=center />
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
### Performance
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
