import os
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import cv2
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import random



# Fix a seed to make result always the same
SEED = 123
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



# Define a function to apply gray scaling to images
def random_grayscale(image, p=0.5):
    """Apply grayscaling to the image with probability p."""
    if np.random.rand() < p:
        gray_image = tf.image.rgb_to_grayscale(image)
        # plt.imshow(gray_image/255.0)
        rgb_gray_image = tf.image.grayscale_to_rgb(gray_image)
        # plt.imshow(rgb_gray_image/255.0)
        return rgb_gray_image 

    return image

# Define a function to apply histogram equalization to images
def histogram_equalization(image):
    # Convert to array
    img_array = img_to_array(image)
    # Convert to YUV color space
    yuv_img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    # Apply histogram equalization to the Y channel
    yuv_img_array[:, :, 0] = cv2.equalizeHist(yuv_img_array[:, :, 0].astype('uint8'))
    # Convert back to RGB color space
    equalized_img_array = cv2.cvtColor(yuv_img_array, cv2.COLOR_YUV2RGB)

    return equalized_img_array

# Define a function to apply preprocessing functions
def custom_preprocessing_function(image):
    # Apply histogram equalization or other preprocessing steps first
    image = histogram_equalization(image)
    # Then apply random grayscaling
    image = random_grayscale(image)
    return image



# Define models
# Define a common CNN model
def create_common_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer
    return model

# Define a function to create a model using ResNet-50 for transfer learning
def create_resnet50_transfer_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # Unfreeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Define the ResNet18 model for transfer learning
def create_resnet18_transfer_model(input_shape, num_classes):
    # Load ResNet model from TensorFlow Hub
    resnet_hub_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"  # Example URL for ResNet50
    resnet_hub_model = hub.KerasLayer(resnet_hub_url, input_shape=input_shape, trainable=False)

    model = Sequential([
        resnet_hub_model,
        Dense(1024, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model



# Define a function to run experiments varifying batch size and model type
def run_experiment(batch_size, model_type, augmentation_flag):
    # Define constants
    IMAGE_SIZE = (224, 224)
    NUM_EPOCHS = 200
    COLOR_MODE = 'rgb'
    SEED = 123
    

    # Data preparation
    # Define the path of datasets
    dataset_path = 'dataset'
    val_dataset_path = 'val_dataset'

    # Preprocessing and data augmentation setup
    # Create data generators
    if augmentation_flag == 1:
        datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=custom_preprocessing_function,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    else:
        datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=histogram_equalization,
            validation_split=0.2
        )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        dataset_path,
        shuffle=True,
        seed=SEED,
        color_mode=COLOR_MODE,
        batch_size=batch_size,
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        subset='training'
    )
    test_generator = datagen.flow_from_directory(
        dataset_path,
        shuffle=True,
        seed=SEED,
        color_mode=COLOR_MODE,
        batch_size=batch_size,
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dataset_path,
        shuffle=False,
        seed=SEED,
        color_mode=COLOR_MODE,
        batch_size=1,
        target_size=IMAGE_SIZE,
        class_mode='categorical'
    )


    # Append the color channel to create input_shape
    input_shape = IMAGE_SIZE+(3,)

    # Select model
    if model_type == 'common':
        model = create_common_cnn(input_shape=input_shape, num_classes=4)
    elif model_type == 'resnet50':
        model = create_resnet50_transfer_model(input_shape=input_shape, num_classes=4)
    elif model_type == 'resnet18':
        model = create_resnet18_transfer_model(input_shape=input_shape, num_classes=4)

    # To deal with the problem in training of something with augmentation
    if aug_flags == 0:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)
    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    if model_type == 'common':
        num_patience = 10
    elif model_type == 'resnet18':
        num_patience = 20
    elif model_type == 'resnet50':
        num_patience = 30

    # Define early stopping criteria
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=num_patience,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True
    )
    
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=math.ceil(len(train_generator)),  # This makes error in testing, so I delete it(training with default)
        epochs=NUM_EPOCHS,
        validation_data=test_generator,
        validation_steps=math.ceil(len(test_generator)),
        callbacks=[early_stopping]
    )


    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Test')
    ax1.set_title(f'Model Accuracy ({model_type}, {batch_size}, {augmentation_flag})')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper left')

    # Plot training & validation loss values
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Test')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left')

    val_loss, val_accuracy = model.evaluate(val_generator, steps=len(val_generator))
    print(f'val_loss: {val_loss}, val_accuracy: {val_accuracy}')
    
    # Add text with validation results to the bottom of the figure
    fig.text(0.5, 0.01, f'Validation Accuracy: {val_accuracy:.4f},      Validation Loss: {val_loss:.4f}', 
             ha='center', va='center', fontsize=12)
    
    # Save the plot to a file and show the plot
    plt.tight_layout()
    file_save_path = 'Graphs/05_200epoch_diffPatience/'
    if os.path.exists(file_save_path):
        pass
    else:
        os.mkdir(file_save_path)
    plt.savefig(file_save_path+f'{model_type}_{batch_size}_{augmentation_flag}_{num_patience}.png')

# Running experiments
batch_sizes = [8, 16]
model_types = ['common', 'resnet50', 'resnet18']
aug_flags = [0,1]


# for model_type in model_types:
for batch_size in batch_sizes:
    for augmentation_flag in aug_flags:
        for model_type in model_types:
            run_experiment(batch_size, model_type, augmentation_flag)


# https://stackoverflow.com/questions/63636565/datagen-flow-from-directory-function
# https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data
