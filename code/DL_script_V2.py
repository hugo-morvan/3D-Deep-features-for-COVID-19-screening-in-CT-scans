from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import os
import tensorflow as tf
import keras
from keras import layers
from keras.optimizers import schedules
import datetime
from sklearn.model_selection import train_test_split

""" TO DO:
-Preprocess to 150x150x150 (preprocessing.py)
    -calculate size of resulting numpy array.
-70-20-10 datasplit --V
-Test data to see if data loaders are working
-Add evaluation metrics (AUC, Recall, ...) --V
-Add model load / save options
-Progressive learning rate (20 at high, 10 at lower ? 10 at low lower?)
"""


print("number of GPU detected: ", len(tf.config.list_physical_devices('GPU')))
    
def visualize_results(history):
  figure = plt.figure(figsize=(15, 10))
  plt.subplot(211)
  plt.title('Cross Entropy Loss')
  plt.plot(history.history['loss'], color='blue', label='train')
  plt.plot(history.history['val_loss'], color='orange', label='test')
  # plot accuracy
  plt.subplot(212)
  plt.title('Classification Accuracy')
  plt.plot(history.history['acc'], color='blue', label='train')
  plt.plot(history.history['val_acc'], color='orange', label='test')

## 3D VGG16 implementation

def get_model_3DVGG(shape = (150, 150, 150, 1)):

  model = tf.keras.models.Sequential([
      # BLOCK-1
          tf.keras.layers.Conv3D(32,(3,3,3), activation = 'relu', input_shape=(shape), padding = 'same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.SpatialDropout3D(0.2),
          tf.keras.layers.Conv3D(32,(3,3,3), activation = 'relu', padding = 'same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.SpatialDropout3D(0.2),
          tf.keras.layers.MaxPool3D((2,2,2)),
          #tf.keras.layers.Dropout(0.2),
      # BLOCK-2
          tf.keras.layers.Conv3D(64, (3,3,3), activation = 'relu', padding = 'same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Conv3D(64, (3,3,3), activation = 'relu', padding = 'same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.MaxPool3D((2,2,2)), 
          #tf.keras.layers.Dropout(0.3),
      # BLOCK-3    
          tf.keras.layers.Conv3D(128, (3,3,3), activation = 'relu', padding = 'same'),
          tf.keras.layers.BatchNormalization(), 
          tf.keras.layers.Dropout(0.3),
          tf.keras.layers.Conv3D(128, (3,3,3), activation = 'relu', padding = 'same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPool3D((2,2,2)),
          tf.keras.layers.Dropout(0.2),
      # BLOCK-4
          tf.keras.layers.Conv3D(256, (3,3,3), activation = 'relu', padding = 'same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.3),
          tf.keras.layers.Conv3D(256, (3,3,3), activation = 'relu', padding = 'same'),
          tf.keras.layers.BatchNormalization(),
          #tf.keras.layers.MaxPool3D((2,2,2)),
          tf.keras.layers.Dropout(0.2),
      # End BLOCK-(FC Dense with BN and Activation = RELU)
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation = 'relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(256, activation = 'relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(128, activation = 'relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['acc'])

  return model


# --------------------Loading and splitting the dataset:-------------------------
print('starting loading')
dataset = np.load(r"/home/santosh_lab/shared/HugoM/hpc_fold/PreProcessed_data/data.npy", allow_pickle=True, fix_imports=True)
labels = np.load(r"/home/santosh_lab/shared/HugoM/hpc_fold/PreProcessed_data/labels.npy", allow_pickle=True, fix_imports=True)
print('loading complete')

print("dataset shape is ", dataset.shape )

#----------- 70-20-10 datasplit ----------------
x, x_test, y, y_test = train_test_split (dataset,labels, test_size=0.1, train_size=0.9, random_state=123)

x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size = 0.22222, train_size =0.77777, random_state=123)

#x_test, y_test -> validation data and labels, do not use until model is finished training
#x_train, y_train -> training data and labels
#x_cv, y_cv -> cross-validation set, used to evaluate the model in training

print('Spliting the dataset')


#-------------map functions for data loader--------------------------------------------------
def train_preprocessing(volume, label):
    """Process training data by  adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_cv, y_cv))

batch_size = 100
# Augment the on the fly during training.

train_dataset = (
    train_loader.shuffle(len(x_train), seed=123)
    .map(train_preprocessing)
    .batch(batch_size, drop_remainder=True, num_parallel_calls=16)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_cv), seed=123)
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

#____Number 0f Epochs_________
epochs = 2
#_____________________________


# --------------3D-VGG------------------------
model=get_model_3DVGG(shape = (256, 256, 256, 1))
# Train the model, doing validation at the end of each epoch
model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=["Accuracy","AUC","FalseNegatives","FalsePositives","TrueNegatives","TruePositives"], #Test if it works
)
log_dir = "/home/santosh_lab/shared/HugoM/hpc_fold/" + datetime.datetime.now().strftime("%Y%m%d-%H")


#---------Callbacks------------
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


print("starting vgg training")

history=model.fit(
    train_dataset, 
    validation_data=validation_dataset, 
    epochs=epochs,
    verbose=2, 
    callbacks=[tensorboard_callback],
    use_multiprocessing=True
)

print("training completed")

visualize_results(history)

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])