# Cat-Dog Classification Model

## Description
![Sample Image](Classification-of-two-classes-dog-and-cat-using-CNN.png)

This project builds a deep learning model to classify images of cats and dogs using TensorFlow and Keras. The dataset is sourced from Kaggle, and the model is trained on labeled images of cats and dogs.

## Dataset Information
- Dataset: [Cats and Dogs for Classification](https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification)
- Training and test data are stored in separate directories.

## Features
- Image classification using Convolutional Neural Networks (CNNs)
- Data augmentation for improved generalization
- Training with TensorFlow's `image_dataset_from_directory()`
- Model evaluation with accuracy and loss metrics

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- Pandas
- OpenDatasets

## Installation & Setup
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib tensorflow opendatasets
   ```
2. Download the dataset using OpenDatasets:
   ```python
   import opendatasets as od
   od.download("https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification")
   ```

## Model Training & Evaluation
1. Load dataset and preprocess images.
2. Build a CNN model with convolutional and pooling layers.
3. Train the model using training data.
4. Evaluate model performance on test data.
5. Visualize accuracy and loss over epochs.

## Usage
Run the Jupyter Notebook to train and evaluate the model:
```bash
jupyter notebook Cat_Dog_ClassificationModel.ipynb
```

## Results
- Model accuracy and loss graphs
- Predictions on test images

## License
This project is for educational purposes only.

---
Developed by **Charutha Pawan Kodikara** 🚀

