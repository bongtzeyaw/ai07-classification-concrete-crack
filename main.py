import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import datetime
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, applications, optimizers, losses

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

SEED = 1
IMG_SIZE = (227,227)
BATCH_SIZE = 32

# Load data
data_dir = r"C:\Users\tzeyawbong\Documents\shrdc\git\files"

train_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=SEED,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE)

val_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=SEED,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches//5)
val_ds = val_ds.skip(val_batches//5)

AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_ds.prefetch(buffer_size=AUTOTUNE)
pf_val = val_ds.prefetch(buffer_size=AUTOTUNE)
pf_test = test_ds.prefetch(buffer_size=AUTOTUNE)

# Preprocess data
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

preprocess_input = applications.mobilenet_v2.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)

# Create model
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False, weights='imagenet')

base_model.trainable = False
base_model.summary()

global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(class_names), activation='softmax')

inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs, outputs=outputs)

optimizer = optimizers.Adam(learning_rate=0.0001)

loss = losses.SparseCategoricalCrossentropy()

model.compile(optimizer,loss=loss,metrics=['accuracy'])


EPOCHS = 3
history = model.fit(pf_train, validation_data=pf_val,epochs=EPOCHS)

# Evaluate model
test_loss, test_accuracy = model.evaluate(pf_test)


