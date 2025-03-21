import numpy as np
import tensorflow as tf
from src.config import ORIGINAL_DIR, LABEL_DIR
from src.data_utils import data_generator, get_file_paths, get_valid_data
from keras import layers, models

# Paths to images and labels
image_paths = get_file_paths(ORIGINAL_DIR, '.jpg')
label_paths = get_file_paths(LABEL_DIR, '.tif')

# Valid data pairs
valid_data = get_valid_data(image_paths, label_paths)
train_data = valid_data[0:2]
test_data = valid_data[2:]
print(train_data)
print(test_data)

# Data generator
batch_size = 1
train_data_gen = data_generator(valid_data, batch_size)
test_data_gen = data_generator(test_data, batch_size)

# Step 1: Retrieve a single batch
sample_images, sample_masks = next(data_generator(valid_data, batch_size))

# Step 2: Inspect shapes and data types
print("Image batch shape:", sample_images.shape)
print("Mask batch shape:", sample_masks.shape)
print("Image data type:", sample_images.dtype)
print("Mask data type:", sample_masks.dtype)

# Step 4: Check unique values in the masks
print("Unique values in masks:", [np.unique(mask) for mask in sample_masks])

def build_fcn_model():
    # Define an input layer with dynamic height and width
    inputs = layers.Input(shape=(None, None, 3))  # Variable height and width

    # Example layers
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)

    # Upsampling
    x = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', activation='relu')(x)

    # Output layer for 4 classes (e.g., segmentation classes)
    outputs = layers.Conv2D(4, kernel_size=1, activation='softmax')(x)

    # Create model
    model = models.Model(inputs, outputs)
    return model


# Initialize model
model = build_fcn_model()

# 3. Compile the Model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Train the Model
# Use steps_per_epoch if the dataset is larger than memory can handle
steps_per_epoch = len(train_data) // batch_size
model.fit(train_data_gen, 
          epochs=6, 
          steps_per_epoch=steps_per_epoch)

# 5. Test
steps = len(test_data) // batch_size
loss, accuracy = model.evaluate(test_data_gen, steps=steps)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

