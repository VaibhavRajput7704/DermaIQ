# train_mobilenetv2.py

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_data_generators

# Settings
base_dir = 'classification_dataset'  # <-- update this
image_size = (224, 224)
batch_size = 32
num_classes = 3
epochs = 10

# Load Data
train_gen, val_gen = get_data_generators(base_dir, image_size, batch_size)

# Build Model
base_model = MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('burn_model_mobilenetv2.h5', save_best_only=True, monitor='val_accuracy', mode='max')
earlystop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint, earlystop]
)

# Save model
model.save("burn_model_final.h5")
