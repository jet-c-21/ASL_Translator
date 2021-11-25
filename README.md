# ASL Translator

![ASL Schematic Diagram](https://i.imgur.com/1Kz743O.jpg)

## Dataset
### Original Data on Kaggle
 
In this project, we create our own dataset, the Dataset-A, by arranging the following data from [Kaggle.com](https://www.kaggle.com/):
- [dataset1](https://www.kaggle.com/grassknoted/asl-alphabet), [Google Drive](https://drive.google.com/file/d/1BaOibzn64d_DOrXczXhOJEvOMey6A-c-/view?usp=sharing)
- [dataset2](https://www.kaggle.com/prathumarikeri/american-sign-language-09az), [Google Drive](https://drive.google.com/file/d/1WJS6IuX9dOMcb7wh-9Ve7HG-V5L3n8qm/view?usp=sharing)
- [dataset3](https://www.kaggle.com/debashishsau/aslamerican-sign-language-aplhabet-dataset), [Google Drive](https://drive.google.com/file/d/103V_z3YRq9TuUF023i465i0UGyY2BB8F/view?usp=sharing)
- [dataset4](https://www.kaggle.com/danrasband/asl-alphabet-test), [Google Drive](https://drive.google.com/file/d/103V_z3YRq9TuUF023i465i0UGyY2BB8F/view?usp=sharing)

### Dataset-A
We want to challenge if our model is general and robust or not, so we build this hybrid dataset.

### Structure of Dataset-A
In both train data and test data, they contain alphabet A to Z, 26 classes folder of right hand image instances data.
#### Training Data
- All data are the subset of **dataset1**
- We get rid of some images that cannot pass our image-pipeline in dataset1
- The image count for each alphabet is approximately to the amount 2220. 

Here's the chart of our image count distribution in Dataset-A training set:

![image count distribution in Dataset-A training set](https://i.imgur.com/SApucwT.png)

#### Testing Data
- For each alphabet, we select 555 image (2220 * 0.2 = 555) instances from dataset2, dataset3 and dataset4

Here's the chart of our image count distribution in Dataset-A testing set:

![image count distribution in Dataset-A testing set](https://i.imgur.com/n1hvHff.png)

#### Environment

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

