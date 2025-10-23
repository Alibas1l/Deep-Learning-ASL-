"""
Train Sign Language Detection Model
This script trains the CNN model for ASL alphabet detection
"""

import random
import numpy as np
import tensorflow as tf
import os

# Make random processes reproducible
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed()

# Configuration
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

# NOTE: Update this path to your dataset location
DATASET_PATH = r'path/to/your/asl_alphabet_train'

print("="*60)
print("ASL Sign Language Detection - Model Training")
print("="*60)

# Check if dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"\nERROR: Dataset path not found: {DATASET_PATH}")
    print("\nPlease update DATASET_PATH in this script to point to your dataset.")
    print("The dataset should contain folders for each sign (A-Z, del, nothing, space)")
    exit(1)

# Load and split the data
from tensorflow.keras.preprocessing import image_dataset_from_directory

print("\nLoading training data...")
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=42
)

print("Loading validation data...")
temp_ds = image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    validation_split=0.2,
    subset="validation",
    shuffle=True,
    seed=42
)

# Split validation data into validation and test sets
temp_size = tf.data.experimental.cardinality(temp_ds).numpy()
valid_size = temp_size // 2

valid_ds = temp_ds.take(valid_size)
test_ds = temp_ds.skip(valid_size)

# Get class names
class_names = train_ds.class_names
print(f"\nClasses found: {len(class_names)}")
print(f"Class names: {class_names}")

# Optimize data loading
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build the CNN model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("\nBuilding CNN model...")

# Input layer
input_ = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
scale_input = Rescaling(1/255.0)(input_)

# Convolutional Block 1
hidden = Conv2D(64, kernel_size=3, padding='same', kernel_initializer="he_normal")(scale_input)
hidden = BatchNormalization()(hidden)
hidden = ReLU()(hidden)

hidden = Conv2D(64, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
hidden = BatchNormalization()(hidden)
hidden = ReLU()(hidden)

hidden = MaxPool2D()(hidden)

# Convolutional Block 2
hidden = Conv2D(128, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
hidden = BatchNormalization()(hidden)
hidden = ReLU()(hidden)

hidden = Conv2D(128, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
hidden = BatchNormalization()(hidden)
hidden = ReLU()(hidden)

hidden = MaxPool2D()(hidden)

# Convolutional Block 3
hidden = Conv2D(256, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
hidden = BatchNormalization()(hidden)
hidden = ReLU()(hidden)

hidden = Conv2D(256, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
hidden = BatchNormalization()(hidden)
hidden = ReLU()(hidden)

hidden = MaxPool2D()(hidden)

# Flatten before Dense layer
hidden = Flatten()(hidden)

# Dense layer
hidden = Dense(700, kernel_initializer="he_normal")(hidden)
hidden = BatchNormalization()(hidden)
hidden = ReLU()(hidden)

# Output layer
output = Dense(29, activation='softmax', kernel_initializer="glorot_normal")(hidden)

model = Model(inputs=input_, outputs=output)
model.summary()

# Compile the model
print("\nCompiling model...")
optimizer = Adam(learning_rate=0.001)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Train the model
print(f"\nTraining model for {EPOCHS} epochs...")
print("="*60)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

# Evaluate on test set
print("\n" + "="*60)
print("Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
model_path = 'LSignLD.h5'
print(f"\nSaving model to {model_path}...")
model.save(model_path)
print(f"Model saved successfully!")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nYou can now run the real-time detector:")
print("  python real_time_sign_detector.py")
