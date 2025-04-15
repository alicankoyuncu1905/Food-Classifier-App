
import os
import json
import tensorflow as tf
from tensorflow  import keras 
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras import layers, models

# ✅ Paths
train_dir = "custom_food_dataset/train"
val_dir = "custom_food_dataset/val"
output_model = "food_classifier_tensorflow.h5"
output_labels = "class_names.json"

# ✅ Parameters
img_size = (224, 224)
batch_size = 8
epochs = 7

# ✅ Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=20
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# ✅ Save class names
class_indices = train_generator.class_indices
class_names = [None] * len(class_indices)
for class_name, index in class_indices.items():
    class_names[index] = class_name

with open(output_labels, "w") as f:
    json.dump(class_names, f)

# ✅ Load MobileNetV2 base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base

# ✅ Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Train model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ✅ Save model
model.save(output_model)
print(f"✅ Model saved to {output_model}")
print(f"✅ Class labels saved to {output_labels}")
