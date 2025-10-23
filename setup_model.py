#!/usr/bin/env python3
"""
Script to help set up the model for the sign language detection app.
This script will convert the model from the notebook format if needed.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def create_model():
    """Recreate the CNN model from the notebook"""
    image_size = 64
    
    # Input layer
    input_ = Input(shape=(image_size, image_size, 3))
    scale_input = Rescaling(1/255.0)(input_)
    
    # Convolutional layers
    hidden = Conv2D(64, kernel_size=3, padding='same', kernel_initializer="he_normal")(scale_input)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    
    hidden = Conv2D(64, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = MaxPool2D()(hidden)
    
    hidden = Conv2D(128, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    
    hidden = Conv2D(128, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = MaxPool2D()(hidden)
    
    hidden = Conv2D(256, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    
    hidden = Conv2D(256, kernel_size=3, padding='same', kernel_initializer="he_normal")(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = MaxPool2D()(hidden)
    
    # Dense layer
    hidden = Dense(700, kernel_initializer="he_normal")(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = Flatten()(hidden)
    
    # Output layer
    output = Dense(29, activation='softmax', kernel_initializer="glorot_normal")(hidden)
    
    model = Model(inputs=input_, outputs=output)
    return model

def main():
    """Main function to set up the model"""
    print("Setting up Sign Language Detection Model...")
    
    try:
        # Try to load existing model
        model = tf.keras.models.load_model('LSignLD.h5')
        print("✓ Existing model loaded successfully!")
    except:
        print("⚠ Model file not found. Creating a new model structure...")
        print("Note: You'll need to train this model with your data.")
        
        # Create model structure
        model = create_model()
        
        # Compile the model
        from tensorflow.keras.optimizers import Adam
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        # Save the model
        model.save('LSignLD.h5')
        print("✓ New model structure created and saved!")
    
    # Print model summary
    print("\nModel Summary:")
    print("=" * 50)
    model.summary()
    
    print("\n" + "=" * 50)
    print("Setup complete! You can now run the Flask app with:")
    print("python app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()