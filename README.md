## Chest X-Ray Classification Project

A  project that involves classifying chest X-ray images into two categories: pneumonia or normal. 

### Description

This project shows the use of deep learning with transfer learning, using a pretrained ResNet-50 model on PyTorch to achieve this classification.. The dataset consists of chest X-ray images categorized into normal and pneumonia, and split into training, validation, and test sets by default, to keep things organized.

### Process

A pretrained ResNet-50 model is used for transfer learning, with its final fully connected layer (50 layers) corrosponding to output two classes: normal and pneumonia. The model is trained for 100 epochs, but only the model with the best validation score is saved to ensure optimal performance. The training and validation losses, along with accuracy metrics, are closely monitored to track the model's progress. Finally, the trained model is evaluated on the test set to assess its performance.

### Results

![Result](https://github.com/I-Zaifa/PneumoniaDectection/blob/main/Result.jpg)

### Acknowledgements
**Data Source:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia


