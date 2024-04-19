
# CNN IN DEFECT DETECTION OF CASTING INDUSTRIAL PRODUCTS

The objective of this project is to develop an accurate and efficient defect detection system for industrial products undergoing casting processes, harnessing the power of deep learning and specifically Convolutional Neural Networks (CNNs). This study aims to explore and compare various optimization algorithms, such as SGD, RMSProp, and Adam, to optimize the accuracy and performance of CNN models in defect detection. By analyzing the performance of each algorithm, the project seeks to determine the best approach to tackle the dynamic and complex challenges of quality inspection in the casting industry context, with a focus on enhancing accuracy and efficiency in defect detection.


## Dataset

 - [casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

| Source      | Kaggle (URL: [casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)) |
|-------------|----------------------------------------------------------------------------------------------------------------------------|
| Format      | Images (JPEG format)                                                                                                      |
| Total Data  | 7348 image data                                                                                                            |
| Categories  | Def and Ok                                                                                                                 |
| Image Size  | 512x512 grayscale  ]

## Data Preparation
A crucial step in the model development and training process is proper data preparation. In this study, images are processed by resizing, converting them to grayscale, normalizing pixel values, and adjusting labe
| Preprocessing   | Deskripsi                 |
|-----------------|---------------------------|
| Resizing        | 300x300 piksel           |
| Color Mode      | grayscale                 |
| Normalisasi     | 1/255                     |
| Label Kelas     | 'ok' = 0 dan 'def' = 1   |
| Batch Size      | 32                        |
| Pengaturan Seed | 0                         |
| Class Mode      | Binary                    |
  
## Model Architecture
The CNN architecture is specifically designed to detect defects in casting industrial products. It consists of Convolutional, MaxPooling, Flatten, and Dense layers.
| Layer          | Output Shape | Parameter  |
|----------------|--------------|------------|
| Conv2D (16)    | (None, 150, 150, 16) | 160   |
| MaxPooling2D   | (None, 75, 75, 16)   | 0     |
| Conv2D (32)    | (None, 75, 75, 32)   | 4640  |
| MaxPooling2D   | (None, 37, 37, 32)   | 0     |
| Conv2D (64)    | (None, 37, 37, 64)   | 18496 |
| MaxPooling2D   | (None, 18, 18, 64)   | 0     |
| Conv2D (128)   | (None, 18, 18, 128)  | 73856 |
| MaxPooling2D   | (None, 9, 9, 128)    | 0     |
| Flatten        | (None, 10368)        | 0     |
| Dense (128)    | (None, 128)          | 1327232|
| Dense (1)      | (None, 1)            | 129   |
|                |                      |        |
| Total Parameter|                      | 1424513  |
| Trainable Parameter|                   | 1424513  |

## Optimizer Comparison: Training Results 
Based on the experimental results, the Adam optimizer at epoch 50 demonstrates superior performance with the highest accuracy and lowest loss on both training and validation data.
![Model Architecture](https://github.com/elangardra/ManufacturingDefectDetection-CNN/blob/main/archive/2.png)
## Confusion Matrix Results
- For "Ok" class: Precision, Recall, and F1-score are 0.9962, indicating accurate classification with good balance.

- Total data for "Ok" class: 262.

- For "Defect" class: Precision, Recall, and F1-score are all 0.9978, indicating highly accurate classification Total data for "Defect" class: 453.
![Model Architecture](https://github.com/elangardra/ManufacturingDefectDetection-CNN/blob/main/archive/3.png)
These metrics illustrate the model's performance in classifying both classes, demonstrating its high capability in distinguishing and classifying data in both categories.

## Miss Classified
![Model Architecture](https://github.com/elangardra/ManufacturingDefectDetection-CNN/blob/main/archive/4.png)
Results show that the model using Adam optimizer for 50 epochs struggled with predictions. It misclassified 3 out of 715 test data instances, all of which were labeled 'Ok' but predicted as 'Defective'

## Summary

- Convolutional Neural Network (CNN) model successfully detects defects in casting industrial products.
- After comparing SGD and RMSProp, the Adam optimizer was selected to maximize defect detection performance and accuracy during training.
- With Adam optimization, the CNN model achieves 99.34% accuracy in detecting "Ok" and 95.8% accuracy in detecting "Defect".
- The model demonstrates high success rates in accurately classifying both categories of defects.
