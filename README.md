# ASL Translator

![ASL Schematic Diagram](https://i.imgur.com/uY6KYo1.jpg)

# Dataset
## Original Data on Kaggle
 
In this project, we create our own dataset, the Dataset-A, by arranging the following data from [Kaggle.com](https://www.kaggle.com/):
- [dataset1](https://www.kaggle.com/grassknoted/asl-alphabet), [Google Drive](https://drive.google.com/file/d/1BaOibzn64d_DOrXczXhOJEvOMey6A-c-/view?usp=sharing)
- [dataset2](https://www.kaggle.com/prathumarikeri/american-sign-language-09az), [Google Drive](https://drive.google.com/file/d/1WJS6IuX9dOMcb7wh-9Ve7HG-V5L3n8qm/view?usp=sharing)
- [dataset3](https://www.kaggle.com/debashishsau/aslamerican-sign-language-aplhabet-dataset), [Google Drive](https://drive.google.com/file/d/103V_z3YRq9TuUF023i465i0UGyY2BB8F/view?usp=sharing)
- [dataset4](https://www.kaggle.com/danrasband/asl-alphabet-test), [Google Drive](https://drive.google.com/file/d/103V_z3YRq9TuUF023i465i0UGyY2BB8F/view?usp=sharing)

## Dataset-A
We want to challenge if our model is general and robust or not, so we build this hybrid dataset.
Links for download:
- [raw images](https://drive.google.com/file/d/1q1IjtWrWjWfJSUG55s1SN0w3bOTuG7cV/view?usp=sharing)
- [images after pipeline](https://drive.google.com/file/d/1TKh-0eYwWLF41tfCfykCuvBVpSY7Fj4k/view?usp=sharing) 
### Structure of Dataset-A
In both train data and test data, they contain alphabet A to Z, 26 classes folder of right hand image instances data.
### Training Data
- All data are the subset of **dataset1**
- We get rid of some images that cannot pass our image-pipeline in dataset1
- The image count for each alphabet is approximately to the amount 2220. 

Here's the chart of our image count distribution in Dataset-A training set:

![image count distribution in Dataset-A training set](https://i.imgur.com/SApucwT.png)

### Testing Data
- For each alphabet, we select 555 image instances (2220 * 0.2 = 555) from dataset2, dataset3 and dataset4

Here's the chart of our image count distribution in Dataset-A testing set:

![image count distribution in Dataset-A testing set](https://i.imgur.com/n1hvHff.png)

# Methodology
## a. Data Preprocessing
![](https://i.imgur.com/NZ1UgBy.png)
The demo images are in the folder ```pipeline-demo```, the image file name prefix indicates the pipeline fucntion type

### image-pipeline, with two different type
#### I. General Pipeline: 
This kind of pipeline can be used in any kinds of preprocessing stage.
```
// work flow
1. roi normalization (by mediapipe)
2. background normalization (by rembg)
3. skin normalization
4. channel normalization
5. resolution normaliztion
```
#### II. Training Pipeline: 
This kind of pipeline can **only be used in training preprocessing stage**.
```
// work flow
1. background normalization (by rembg)
2. roi normalization (by mediapipe)
3. skin normalization
4. channel normalization
5. resolution normaliztion
```
### Data Augmentation
1. Implement by ```keras.ImageDataGenerator``` with ```zoom_range=0.1,```, ```width_shift_range=0.1```, ```height_shift_range=0.1```, ```shear_range=0.1```
2. Implement by ```keras.model.Engine```, we create our own Spatial Transformer Layer ```stn()```.

## b. Model Building
You can download our models at here: [saved_models_v1](https://drive.google.com/file/d/1IxY66z9Oh6gtcW8sr1MjXr8sQx8-mi3E/view?usp=sharing)
### Normal Model - Pure CNN Structure without Spatial Transform Layers:
The implement code is in ```asl_model/models.py```-```get_model_1()```
```python
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())

model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))

# finish feature extraction
model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Dense(26, activation='softmax'))
```

### STL Model - Spatial Transform Layer with CNN Structure
The implement code is in ```asl_model/stl/struct_a/model.py```-```get_stn_a_model_8()```
```python
input_layers = layers.Input((size, size, 1))
x = stn(input_layers)

x = layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                  kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                  kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                  kernel_initializer=initializers.glorot_uniform(seed=STN_SEED))(x)
x = layers.BatchNormalization()(x)

x = layers.Flatten()(x)

x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.25)(x)

output_layers = layers.Dense(26, activation="softmax")(x)

model = tf.keras.Model(input_layers, output_layers)
```
## c. Model Training
#### Basic
- 57717 train images, 20% will become the validation data
- 14430 test image, 555 test images for each alphabet

#### First, data-structure-selection
Select the best structure for normal-model and stl-model. With following hyper-parameters:
```python
lr = 0.001
epoch = 10
batch_size = 128

optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
loss='categorical_crossentropy',
metrics=['accuracy']
```

#### Second, use callback function to train the best model of each type. With following settings:
```python
BATCH = 128
EPOCH = 100 # max epoch

# call back functions
es_callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau()

loss="categorical_crossentropy" 
optimizer="adam" 
metrics=["accuracy"]
```

## d. Model Evaluation

### Normal Model 
#### Validation Data - Epoch Accuracy
- Train : 0.9917 (Orange)
- Valid : 0.9864 (Blue)
<br><br>
![normal-acc-s](https://i.imgur.com/2giV2ZN.jpg)
![normal-acc-l](https://i.imgur.com/aZ7n8WP.jpg)
#### Validation Data - Epoch Loss
- Train : 0.02555 (Orange)
- Valid : 0.05084 (Blue)
<br><br>
![normal-loss-s](https://i.imgur.com/7lWn4xj.jpg)
![normal-loss-l](https://i.imgur.com/eiQQRF9.jpg)

#### Testing Data - Total Accuracy : 89.4%
![](https://i.imgur.com/EeUx1Oe.png)

#### Testing Data - F1-Score Report:
![](https://i.imgur.com/4XIAHfe.png)

### STL Model
#### Validation Data -  Epoch Accuracy
- Train : 0.9871 (Orange)
- Valid : 0.9883 (Blue)
<br><br>
![stl-acc-s](https://i.imgur.com/Cl2eP7L.jpg)
![stl-acc-l](https://i.imgur.com/4r22ZgJ.jpg)

#### Validation Data - Epoch Loss
- Train : 0.05062 (Orange)
- Valid : 0.04649 (Blue)
<br><br>
![stl-loss-s](https://i.imgur.com/WMfywmI.jpg)
![stl-loss-l](https://i.imgur.com/PbrMH6q.jpg)

#### Testing Data - Total Accuracy : 90.6%
![](https://i.imgur.com/O9mPsLY.png)

#### Testing Data - F1-Score Report:
![](https://i.imgur.com/prlZFy2.png)

**There are more evaluation charts in the folder ```charts```** 


## Getting Starting
### Environment
```
conda create --name aslt python=3.8 -y
```
(Options) - If you want to use jupyter run these commands
```
// actactivate venv
pip install ipykernel
python -m ipykernel install --user --name aslt --display-name "ASLT"
```

### install Pytorch for package [rembg](https://github.com/danielgatis/rembg)
1. get pyTorch install instructions on [pytorch.org](https://pytorch.org/)
For example:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

