#!/usr/bin/env python
# coding: utf-8

# # Evaluate Normal Model

# In[1]:


from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# In[2]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[3]:


TEST_DIR = 'DATASET_A_AP/test'
entire_model_s_path = 'saved_model/normal_model'
cm_hm_s_path = 'normal-model-confusion-matrix-heat-map.jpg'


# In[4]:


# Convert folder to dataframe of images' paths & labels
def get_paths_labels(path, allowed_extension="jpg"):
        global Path
        images_dir = Path(path)
        
        filepaths = pd.Series((images_dir.glob(fr'**/*.{allowed_extension}'))).astype(str)
        filepaths.name = "path"
        
        labels = filepaths.str.split("/")[:].str[-2]
        labels.name = "label"

        # Concatenate filepaths and labels
        df = pd.concat([filepaths, labels], axis=1)

        # Shuffle the DataFrame and reset index
        df = df.sample(frac=1).reset_index(drop = True)
        return df


# In[5]:


test_df = get_paths_labels(TEST_DIR)


# In[6]:


test_generator = ImageDataGenerator(rescale=1. / 255.)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical',
    # batch_size=1,
    shuffle=False,
)


# # reload the entire model

# In[7]:


model = tf.keras.models.load_model(entire_model_s_path)


# ### Review Accuracy and Loss

# In[8]:


loss, acc = model.evaluate(test_images)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# ### View each class prediction

# In[9]:


alphabet_ls = list(ascii_uppercase)


# In[10]:


pred_alphabet_idxs = np.argmax(model.predict(test_images), -1)


# In[11]:


pred_res = [alphabet_ls[i] for i in pred_alphabet_idxs]


# In[12]:


ground_truths = [alphabet_ls[i] for i in test_images.labels]


# In[13]:


data_c_per_cls = 555

def get_empty_afp_dict():
    result = dict()
    for a in alphabet_ls:
        result[a] = 0
    
    return result

# Source code credit for this function: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
def print_confusion_matrix_heat_map(confusion_matrix, class_names, figsize = (26,26), fontsize=14, plt_s_path=None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    
    if plt_s_path:
        plt.savefig(plt_s_path)


# In[14]:


afp_dict = get_empty_afp_dict()
for pred_cls, gt in zip(pred_res, ground_truths):
    if pred_cls != gt:
        if gt in afp_dict.keys(): # gt should be the key, not pred_cls
            afp_dict[gt] += 1
        else:
            afp_dict[gt] = 0


# In[15]:


afp_df = pd.DataFrame(
    list(afp_dict.items()),
    columns=['alphabet', 'fc'],
)

afp_df['fcr'] = afp_df['fc'] / data_c_per_cls
afp_df['acc'] = 1 - afp_df['fcr']


# # Failed Prediciton Count View

# ## Plotly

# ### Failed Prediction Count of Each Alphabet

# In[16]:


fig = px.bar(
    afp_df,
    x='alphabet', 
    y='fc',
    
    labels={
        "alphabet": "Alphabet",
        "fc": "Failed Prediction Count",
    },
    title='Failed Prediction Count of Each Alphabet',
)

fig.show()


# ### Accuarcy of Each Alphabet

# In[17]:


fig = px.bar(
    afp_df,
    x='alphabet', 
    y='acc',
    
    labels={
        "alphabet": "Alphabet",
        "acc": "Accuaracy",
    },
    
    title='Accuarcy of Each Alphabet',
)

fig.show()


# ### Confusion Matrix Heat Map and F1 Report

# In[18]:


cm = confusion_matrix(ground_truths, pred_res)


# In[20]:


print_confusion_matrix_heat_map(cm, alphabet_ls, plt_s_path=cm_hm_s_path)


# In[22]:


clf_report = classification_report(ground_truths, pred_res)
print(clf_report)


# In[ ]:





# In[ ]:




