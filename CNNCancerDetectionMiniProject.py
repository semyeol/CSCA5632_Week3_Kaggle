#!/usr/bin/env python
# coding: utf-8

# # Week 3: CNN Cancer Detection Kaggle Mini-Project

# ### Description of Problem 

# Our goal is to classify small image patches as either cancerous (label=1) or noncancerous (label=0). 
# 
# There are about 220,000 training images and 57,000 testing images. 
# 
# 

# In[26]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:





# ### Exploratory Data Analysis

# In[27]:


train_labels_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
train_labels_df.head()


# In[28]:


print(train_labels_df['label'].value_counts())
print(train_labels_df.isnull().sum())


# In[29]:


# Visualization of training label distribution
import seaborn as sns
import matplotlib.pyplot as plt

train_label_counts = train_labels_df['label'].value_counts()
total = len(train_labels_df)

plt.figure(figsize=(6,4))
ax = sns.countplot(data=train_labels_df, x='label')

for p in ax.patches:
    count = p.get_height()
    percent = f'{100 * count / total:.2f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.text(x, y + total * 0.01, percent, ha='center', va='bottom', fontsize=12)

plt.title("Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.ylim(0, train_label_counts.max() * 1.1)
plt.show()


# In[30]:


# Display sample images 
import cv2

def load_image(image_id, base_path='/kaggle/input/histopathologic-cancer-detection/train'):
    path = os.path.join(base_path, f"{image_id}.tif")
    return cv2.imread(path)

def display_samples(df, label, n=3):
    samples = df[df['label'] == label].sample(n)
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for img_id, ax in zip(samples['id'], axes):
        img = load_image(img_id)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.suptitle(f"Label: {label}")
    plt.show()

display_samples(train_labels_df, label=0)
display_samples(train_labels_df, label=1)

img = load_image(train_labels_df['id'][0])
print("Image shape:", img.shape)


# In[31]:


# Visualization of random normal vs tumor image
def plot_color_distribution_histogram(image, title_suffix=""):
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        plt.hist(image[..., i].ravel(), bins=256, color=color, alpha=0.5)
    plt.title(f"Pixel Intensity Distribution - {title_suffix}")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.show()

img_tumor = load_image(train_labels_df[train_labels_df['label'] == 1].sample(1).iloc[0]['id'])
img_normal = load_image(train_labels_df[train_labels_df['label'] == 0].sample(1).iloc[0]['id'])

plot_color_distribution_histogram(img_tumor, "Tumor")
plot_color_distribution_histogram(img_normal, "Normal")


# Comparing the pixel distributions, we can see that noncancerous tissue is uniform whereas the cancerous regions are darker and experience greater oscillations. 

# ### Prepare data

# We'll convert the label to a string and add a column called "filename" so the images are compatible with ImageDataGenerator. 

# In[32]:


train_labels_df['label'] = train_labels_df['label'].astype(str)
train_labels_df['filename'] = train_labels_df['id'] + '.tif'
train_labels_df['filename'] = train_labels_df['filename'].astype(str)


# In[33]:


train_labels_df.head()


# Now, we'll take a small subset of the dataset to create the test and train split. 

# In[34]:


from sklearn.model_selection import train_test_split

df_subset, _ = train_test_split(
    train_labels_df,
    train_size=10000, # start with 10,000 
    stratify=train_labels_df['label'],
    random_state=42
)

train_df, val_df = train_test_split(
    df_subset,
    test_size=0.2, # 20% validation
    stratify=df_subset['label'],
    random_state=42
)

print("Train label distribution:")
print(train_df['label'].value_counts(normalize=True))

print("\nValidation label distribution:")
print(val_df['label'].value_counts(normalize=True))


# In[35]:


import tensorflow as tf

batch_size = 32

def load_image_tf(path):
    img = cv2.imread(path)
    if img is None:
        # Return blank image if file is missing/corrupt
        return np.zeros((96, 96, 3), dtype=np.float32)
    img = cv2.resize(img, (96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0 # normalize

def generate_data(paths, labels):
    for path, label in zip(paths, labels):
        img = load_image_tf(path)
        yield img, label

train_paths = [f"/kaggle/input/histopathologic-cancer-detection/train/{f}" for f in train_df['filename']]
train_labels = train_df['label'].values.astype(np.float32)

val_paths = [f"/kaggle/input/histopathologic-cancer-detection/train/{f}" for f in val_df['filename']]
val_labels = val_df['label'].values.astype(np.float32)

train_data = tf.data.Dataset.from_generator(
    lambda: generate_data(train_paths, train_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).cache().shuffle(1024).repeat().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_data = tf.data.Dataset.from_generator(
    lambda: generate_data(val_paths, val_labels),
    output_signature=(
        tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ### Model Architecture

# We'll train two different models.
# 
# 1. Basic CNN Model with batch normalization. This model has few layers, a simple architecture, and fast training
#    
# 2. VGG Model will likely perform better for our complex image recognition.

# In[36]:


# Basic Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization

def build_basic(input_shape=(96, 96, 3)):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model

basic_model = build_basic()

basic_model.summary()


# In[37]:


# Evaluating the Basic Model
steps_per_epoch = len(train_df) // batch_size
validation_steps = len(val_df) // batch_size

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras import callbacks

callbacks_list = [
    # Stop early if validation loss doesn't improve
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when plateaued
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='best_basic_model.keras',
        monitor='val_auc', 
        save_best_only=True,
        mode='max', 
        verbose=1
    )
]

bm_history = basic_model.fit(
    train_data, 
    validation_data=val_data, 
    epochs=10, 
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks_list
)

results = basic_model.evaluate(val_data, steps=validation_steps)
print(f"\nValidation Results:")
print(f"Loss: {results[0]:.4f}")
print(f"AUC: {results[1]:.4f}")


# The early stop was triggered, meaning we avoid wasting time on epochs that don't improve the result; the model achieved the best version in epoch 3.
# 
# It achieved 81.3% AUC, which is a reasonable baseline. 

# In[38]:


import matplotlib.pyplot as plt

# AUC plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(bm_history.history['auc'], label='Train AUC')
plt.plot(bm_history.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(bm_history.history['loss'], label='Train Loss')
plt.plot(bm_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Judging by the plots, we can that the model is overfitting. In the left plot, the Train AUC increases to 1.0 which means the models learns the training data well, and the Val AUC is stuck around 0.75 and 0.80. This gap signals overfitting. In the right plot, we see that the Val loss increases after epoch 1, but Train loss continusously decreases, which means the model is simply memorizing training data but can not apply it to the validation data. 
# 
# Let's try to improve this with our next model

# Now, we'll create the VGG model. My model has 6 convolution layers, 3 pooling operations, and 3 dense layers. 
# 
# In this model, we double filters after each pooling. It'll start learning edges and textures and end with complex features like tissue structure and cell patterns. 
# We use two convolution layers--the first for detecting initial features, and the second for refining. After two conv layers, it pools and reduces size. 

# In[39]:


# VGG Model
import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg(input_shape=(96, 96, 3)):
    model = models.Sequential(name="VGGNet")
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())  
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))  

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))  
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

vgg_model = build_vgg()
vgg_model.summary()


# In[40]:


vgg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

steps_per_epoch = len(train_df) // batch_size  
validation_steps = len(val_df) // batch_size  

callbacks_list = [
    # Early stopping 
    callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,  
        mode='max',  
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when plateaue
    callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,  # Cut LR in half
        patience=3,
        mode='max',
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='best_vgg_model.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("Training VGG Model...")
vgg_history = vgg_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks_list,
    verbose=1
)

print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)
results = vgg_model.evaluate(val_data, steps=validation_steps, verbose=1)
print(f"\nValidation Loss: {results[0]:.4f}")
print(f"Validation AUC: {results[1]:.4f}")


# In[41]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(14, 5))

# AUC plot
plt.subplot(1, 2, 1)
plt.plot(vgg_history.history['auc'], label='Train AUC', linewidth=2, marker='o')
plt.plot(vgg_history.history['val_auc'], label='Val AUC', linewidth=2, marker='o')
plt.axhline(y=0.9, linestyle='--', color='red', alpha=0.5, linewidth=1, label='Excellent (0.9)')
plt.title('VGG Model AUC', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0.5, 1.05])

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(vgg_history.history['loss'], label='Train Loss', linewidth=2, marker='o')
plt.plot(vgg_history.history['val_loss'], label='Val Loss', linewidth=2, marker='o')
plt.title('VGG Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("VGG MODEL TRAINING SUMMARY")
print("="*60)


# Looking at the AUC plot, we can see the train AUC has smooth progression, but the validation AUC experienced lots of fluctuation. This noise could be attributed to the smaller dataset we used.
# 
# The train loss shows a smooth decrease while the val loss is unstable. 
# 
# While this is a significant increase over our basic model, let's try to increase performance and stability. 

# ### VGG Model: Tuning Learning Rate

# 

# In[42]:


vgg_model_optimized = build_vgg()


vgg_model_optimized.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # half of 0.001 from first VGG
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=3,
        mode='max',
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='best_vgg_model_lr0005.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("Training VGG with LR=0.0005...")
vgg_history_optimized = vgg_model_optimized.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks_list
)

results_optimized = vgg_model_optimized.evaluate(val_data, steps=validation_steps)

print("\n" + "="*60)
print("LEARNING RATE COMPARISON")
print("="*60)
print(f"Original (LR=0.001):  Val AUC = 0.9011")
print(f"Optimized (LR=0.0005): Val AUC = {results_optimized[1]:.4f}")
print(f"Improvement: {(results_optimized[1] - 0.9011):.4f} ({((results_optimized[1] - 0.9011) / 0.9011 * 100):.2f}%)")
print("="*60)


# In[43]:


plt.figure(figsize=(14, 5))

# AUC plot
plt.subplot(1, 2, 1)
plt.plot(vgg_history_optimized.history['auc'], label='Train AUC', linewidth=2, marker='o')
plt.plot(vgg_history_optimized.history['val_auc'], label='Val AUC', linewidth=2, marker='o')
plt.axhline(y=0.9, linestyle='--', color='red', alpha=0.5, linewidth=1, label='Excellent (0.9)')
plt.title('VGG Model AUC (LR=0.0005)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0.5, 1.05])

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(vgg_history_optimized.history['loss'], label='Train Loss', linewidth=2, marker='o')
plt.plot(vgg_history_optimized.history['val_loss'], label='Val Loss', linewidth=2, marker='o')
plt.title('VGG Model Loss (LR=0.0005)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("VGG MODEL TRAINING SUMMARY (LR=0.0005)")
print("="*60)

best_epoch_idx = np.argmax(vgg_history_optimized.history['val_auc'])
best_epoch = best_epoch_idx + 1
best_val_auc = vgg_history_optimized.history['val_auc'][best_epoch_idx]
best_val_loss = vgg_history_optimized.history['val_loss'][best_epoch_idx]

print(f"Total Epochs Trained: {len(vgg_history_optimized.history['auc'])}")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation AUC: {best_val_auc:.4f}")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"\nFinal Validation AUC: {results_optimized[1]:.4f}")
print(f"Final Validation Loss: {results_optimized[0]:.4f}")


# The tuned VGG model showed about a 3% improvement in both AUC and loss. In addition, the epoch history shows much more stable progress. 
# Lowering the learning rate allowed for smoother convergence, better fune-tuning, less sensitivity to batch variance, and higher final performance.  

# ## Results

# From the three models we trained, the tuned VGG model performed the best. Decreasing the learning rate improved the VGG model and tuning other parameters such as dataset size and epoch times could further optimize the model. Finally, we'll test the three models on Kaggle. 

# ## Test on Kaggle

# ### Basic Model
# Validation Results:
# 
# 557.9s	85,	Loss: 0.6035
# 
# 557.9s	86,	AUC: 0.7737
# 
# ### VGG Model
# Validation Results:
# 
# 3883.9s	135, Validation Loss: 0.3385
# 
# 3883.9s	136, Validation AUC: 0.9351
# 
# ### VGG Model Tuned
# Validation Results:
# 4383.9s	176, Validation Loss: 0.3102
# 
# 4383.9s	177, Validation AUC: 0.9498
# 
