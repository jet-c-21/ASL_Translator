# ASL Translator

![ASL Schematic Diagram](https://i.imgur.com/mq20CVv.png)

## Dataset
[dataset1](https://www.kaggle.com/grassknoted/asl-alphabet), [Google Drive](https://drive.google.com/file/d/1FpYkbhAb7fX1z_ygNA1i__av2h6uAkUd/view?usp=sharing)

## Environment
```
conda create --name aslt python=3.8 -y
```
```
// actactivate venv
pip install ipykernel
python -m ipykernel install --user --name aslt --display-name "ASLT"
```

### initialize from package ```rembg```
1. get pyTorch install instructions on [pytorch.org](https://pytorch.org/)
For example:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Types of preprocessing

#### Base-Norm:
- Region-Norm : Hand-ROI
- Illumination-Norm: RGB-to-HSV (adaptive smooth)

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
This type of pipeline doesn't contain any data augmentation structure
```python
pipeline_#() : return np.ndarray
```

#### Type-B
This type of pipeline contains data augmentation structure and it can only used be in training phase
```python
pipeline_with_da_#() : return [np.ndarray, ..., np.ndarray]
```

