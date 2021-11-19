# ASL Translator

![ASL Schematic Diagram](https://i.imgur.com/mq20CVv.png)

## Dataset
[dataset1](https://github.com/grassknoted/Unvoiced), [Google Drive](https://drive.google.com/file/d/1FpYkbhAb7fX1z_ygNA1i__av2h6uAkUd/view?usp=sharing)

## Types of preprocessing

#### Base-Norm:
- Region-Norm : Hand-ROI
- Illumination-Norm: RGB-to-HSV

#### Data-Aug:
1. rotate
2. flip
3. noise
4. filter
5. dilation 
6. erosion
7. STL

#### Background-Norm:
1. bg_normalization_red_channel()
2. bg_normalization_fg_extraction()

#### Channel-Norm: grayscale

#### Resolution-Norm: resize

## Pipeline Naming Convention

#### Type-A
this type of pipeline doesn't contain any data augmentation structure
```python
pipeline_#() : return np.ndarray
```

#### Type-B
this type of pipeline contains data augmentation structure
```python
pipeline_da_#() : return [np.ndarray, ..., np.ndarray]
```

