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
def create_resnet_hub_model(input_shape, num_classes):
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
    NUM_EPOCHS = 30
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
            preprocessing_function=histogram_equalization,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
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

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # sparsecategoricalcrossentropy

    # Define early stopping criteria
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=math.ceil(len(train_generator)),
        epochs=NUM_EPOCHS,
        validation_data=test_generator,
        validation_steps=math.ceil(len(train_generator)),
        callbacks=[early_stopping]
    )


    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_generator, steps=len(val_generator))

    ## Generate predictions and calculate cm and cr
    # val_predictions = model.predict(val_generator, steps=len(val_generator))
    # predicted_classes = np.argmax(val_predictions, axis=1)
    # true_classes = val_generator.classes
    # cm = confusion_matrix(true_classes, predicted_classes)
    # cr = classification_report(true_classes, predicted_classes)

    # Save results to file
    output_file_name = f'results/multi_results.txt'
    with open(output_file_name, 'a') as file:
        # file.write(f'Validation Loss: {val_loss}\n')
        file.write(f'{model_type:10}, {batch_size:3}, {augmentation_flag}: {val_accuracy}\n\n')
        # file.write("Confusion Matrix:\n")
        # cm_str = '\n'.join(['\t'.join([str(cell) for cell in row]) for row in cm])
        # file.write(cm_str + "\n\n")
        # file.write("Classification Report:\n")
        # file.write(cr)

# Running experiments
batch_sizes = [4]
model_types = ['common', 'resnet50', 'resnet18']
# model_types = ['common']
aug_flags = [0, 1]


for model_type in model_types:
    for batch_size in batch_sizes:
        for augmentation_flag in aug_flags:
            run_experiment(batch_size, model_type, augmentation_flag)


# https://stackoverflow.com/questions/63636565/datagen-flow-from-directory-function
# https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data
