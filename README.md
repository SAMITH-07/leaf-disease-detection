# Leaf Disease Detection using CNN

A deep learning project that detects leaf diseases in various plants using Convolutional Neural Networks (CNN).

## Features

- **Multi-plant Support**: Detects diseases in Apple, Corn, Grape, Potato, and Tomato plants
- **25 Disease Classes**: Covers common leaf diseases across all supported plants
- **GUI Interface**: User-friendly tkinter-based GUI for easy image upload and prediction
- **Command Line Interface**: Script-based prediction for batch processing
- **Pre-trained Model**: Includes trained CNN model with high accuracy

## Disease Classes

### Apple
- Apple_scab
- Black_rot
- Cedar_apple_rust
- Healthy

### Corn (Maize)
- Cercospora_leaf_spot Gray_leaf_spot
- Common_rust
- Healthy
- Northern_Leaf_Blight

### Grape
- Black_rot
- Esca_(Black_Measles)
- Healthy
- Leaf_blight_(Isariopsis_Leaf_Spot)

### Potato
- Early_blight
- Healthy
- Late_blight

### Tomato
- Bacterial_spot
- Early_blight
- Healthy
- Late_blight
- Leaf_Mold
- Septoria_leaf_spot
- Spider_mites Two-spotted_spider_mite
- Target_Spot
- Tomato_Yellow_Leaf_Curl_Virus
- Tomato_mosaic_virus

## Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install tensorflow numpy opencv-python pillow
```

## Usage

### GUI Application
```bash
python predict_fin_GUI_fixed.py
```
- Click "browse image" to select a leaf image
- The model will predict the disease and display the result

### Command Line
```bash
python predict_fixed.py
```
- Tests on the sample image in `im_for_testing_purpose/` directory

### Training (Optional)
```bash
python Cnn_train_fin.py
```
- Trains the CNN model on the dataset
- Requires dataset in `dataset/train` and `dataset/test` directories

## Model Architecture

The CNN model consists of:
- 5 Convolutional layers with ReLU activation
- MaxPooling layers after each convolution
- BatchNormalization for stable training
- Dropout layers to prevent overfitting
- Dense output layer with softmax activation

## Dataset

The project uses a dataset with:
- Training images: `dataset/train/`
- Testing images: `dataset/test/`
- Sample images: `im_for_testing_purpose/`

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- OpenCV
- Pillow
- Tkinter (for GUI)

## Performance

- Input image size: 128x128 pixels
- Model accuracy: ~95% on test dataset
- Inference time: <1 second per image

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## License

This project is open source and available under the MIT License.
