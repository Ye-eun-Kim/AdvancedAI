# Image Classification with CNNs and ResNet Models

## Overview
This project implements and compares various convolutional neural network (CNN) architectures for image classification, including a custom-built untrained CNN model and pretrained ResNet models (ResNet18 and ResNet50). The goal is to classify images into four categories: people, buildings, food, and others, using a small dataset.

## Requirements
To run this project, you need the following:
- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

## Installation
Install the required libraries using the following command:

```pip install tensorflow numpy opencv-python matplotlib```


## Dataset Structure
Organize your dataset in the following structure with separate folders for each category:
```dataset/
│
├── buildings/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
│
├── food/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
│
├── other
│
└── people


val_dataset/
│
├── buildings/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
│
├── food/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
│
├── other
│
└── people
```

## Usage
To start the experiments, run the `run_experiment.py` script. You can modify batch sizes, model types, and augmentation flags according to your requirements.

### Key Functions
- `random_grayscale(image, p)`: Applies grayscale to the image with a specified probability.
- `histogram_equalization(image)`: Enhances image contrast using histogram equalization.
- `create_common_cnn(input_shape, num_classes)`: Builds a custom CNN model.
- `create_resnet50_transfer_model(input_shape, num_classes)`: Creates a ResNet50 model using transfer learning.
- `create_resnet18_transfer_model(input_shape, num_classes)`: Develops a ResNet18 model using transfer learning from TensorFlow Hub.

### Running Experiments
Adjust the batch sizes, model types (`'common'`, `'resnet50'`, `'resnet18'`), and augmentation flags (0 or 1) to run different configurations.

## Results
The results, including accuracy and loss plots, will be saved in the 'Graphs' directory.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any queries, please reach out to [july215215@gmail.com](july215215@gmail.com).
