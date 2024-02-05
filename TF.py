import matplotlib.pyplot as plt
import numpy as np
import PIL

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

dataset_dir = pathlib.Path("flower_photos")

image_count = len(list(dataset_dir.glob("*/*.jpg")))
batch_size = 32
img_width = 180
img_height = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

train_ds_orig = train_ds

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

class_names = train_ds_orig.class_names
print(f"Class names: {class_names}")

num_classes = len(class_names)
model = Sequential([
	layers.experimental.preprocessing.Rescaling(1, 255, input_shape = (img_height, img_width, 3)),

	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(num_classes)
	])

model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics = ['accuracy'])

model.load_weights("flower_photos_model")

loss, acc = model.evaluate(train_ds, verbose=2)
print("Train Accuracy: {:5.2f}%".format(acc * 100))

img = tf.keras.preprocessing.image.load_img("flower.jpeg", target_size=(img_width, img_height))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("На изображении вероятно всего {} ({:.2f}% вероятность)".format(class_names[np.argmax(score)], 100 * np.max(score)))