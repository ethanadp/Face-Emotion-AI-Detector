# **Facial Key Points and Emotion Detector**
[![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.6.0-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-v2.6.0-red)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v0.24.2-blue)](https://scikit-learn.org/stable/)


## **Description**
This project aims to develop two convolutional neural network (CNN) models to analyze facial features and emotions in images, respectively. The first model analyzes 6400+ images and their associated 15 facial key points, while the second model analyzes 24500+ images with their associated emotions. The goal is to combine the output of the two models to predict an image's emotion and facial key points.

## **Getting Started**
To run this project, follow these steps:
1. Download the data by running the DownloadData.py file. The data is stored on Google Cloud because the file sizes are too large.
2. Run the Emotion_Detector_AI.ipynb notebook to execute the algorithms. Make sure you have the following dependencies installed: Keras, TensorFlow, Seaborn, Pandas, NumPy, cv2, PIL, and MatPlotLib.

## **Datasets**
The main datasets used in this project are data.csv and icml_face_data.csv. data.csv is a table that's 31 columns by 2140 rows, 30 of those columns being x and y components of facial key points where the last column is the image data that's preprocessed to be individual arrays of 96x96 pixels that gets normalized for the convolutional neural networks. The icml_face_data.csv is a table that's 2 columns by 24568 rows. The first column represents the encoded emotion, and the second column represents the image data that gets reshaped into a 96x96 pixel array and normalized for the CNN model.

There are also several .hdf5 and .json files included in this project:

- `FacialKeyPoints_weights.hdf5` : saved weights from the first CNN model taking the 30x2140 dataset.
- `FacialExpression-model.json` : saved weights for the second CNN model taking the 2x24568 dataset.
- `detection.json` : pre-saved weights model that had better parameters for more accurate data on key point detection.
- `emotion.json` : pre-ran weights of the model that detects emotion.

## **CNN Architecture**
The convolutional neural network models used a "ResNet" so that the "vanishing gradient" problem was avoided, and so that we are able to enable 152 layers of training with virtually no gradient issues. The layers of the final model, in order, are as follows:

1. Zeropadding
2. Conv2D
3. BatchNorm/Relu
4. MaxPool2D
5. RES-block (the RES-Blocks obtain a convolution block, and two identity blocks)
6. AveragePooling2D
7. Flatten()
8. Dense Layer/Relu/Dropout
9. Dense Layer/Relu/Dropout
10. Dense Layer/Relu
11. Loading Pre-trained Model Weights
12. To load the pre-trained model weights, run the Notebook with the code, Emotion_Detector_AI.ipynb.

## **Evaluation Metrics**
To evaluate the CNN models, we compiled the results from the pre-trained weights with the "Adam" optimizer from the Keras library that measured the loss with "mean_squared_error" and "accuracy" as its metrics.

We also saved several visuals in the repository, including:
- `Num_of_Emotion_training.jpg` : A bar chart for the number of images with their respective emotions to train our model with.
![IMAGE](https://github.com/ethanadp/Face-Emotion-AI-Detector/blob/main/Num_of_Emotion_training.jpg)
- `Accuracy_Loss.png` : Visual of Accuracy Loss over the Epochs of training for our model.
![IMAGE](https://github.com/ethanadp/Face-Emotion-AI-Detector/blob/main/Accuracy_Loss.png)
- `Confusion_Matrix.png` : Confusion Matrix for the predicted emotions vs actual emotions.
![IMAGE](https://github.com/ethanadp/Face-Emotion-AI-Detector/blob/main/Confusion_Matrix.png)
- `Face_Board.png` : A board of several images of faces with their predicted keypoints and predicted vs actual emotion.
![IMAGE](https://github.com/ethanadp/Face-Emotion-AI-Detector/blob/main/Face_board.png)

## **Contributing**
We welcome contributions from the community! If you have any suggestions or ideas, feel free to open an issue or submit a pull request.

## **Acknowledgements**
We would like to acknowledge the following resources and libraries that were used in this project:
- Keras
- TensorFlow
- Seaborn
- Pandas
- NumPy
- cv2
- PIL
- MatPlotLib

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this project, I developed a Convolutional Neural Network (CNN) using Keras and TensorFlow to detect facial keypoints and expressions. The goal of the project was to accurately predict the expression and corresponding facial keypoints for a given image of a human face. The dataset used for training and testing the model included images of faces with different expressions, with a total of approximately 27000 images split across 5 different emotions plus keypoints. The total amount of data per emotion used to train the model is represented by the bar chart below.

![IMAGE](https://github.com/ethanadp/Face-Emotion-AI-Detector/blob/main/Num_of_Emotion_training.jpg)

The model architecture consisted of several convolutional and pooling layers, followed by a fully connected layer and a regression or classification layer for keypoints and expression detection. The model was trained for a total of 35 epochs, with the training and validation accuracy and loss plotted with respect to time in the graphs below. 

![IMAGE](https://github.com/ethanadp/Face-Emotion-AI-Detector/blob/main/Accuracy_Loss.png)

A confusion matrix representing the model's predicted emotions versus true emotions, labeled 0-4, is shown below. The matrix shows that the model was most accurate in predicting emotions 0, 2, 3, and 4, but struggled with emotion 1, representing "disgust", mostly due to the low amounts of data to train the model as seen above.

![IMAGE](https://github.com/ethanadp/Face-Emotion-AI-Detector/blob/main/Confusion_Matrix.png)

Finally, a grid of emotions with detected keypoints and predicted emotions for their respective image is shown below. The grid provides visualizations of the model's predictions for different emotions, including the facial keypoints detected and the predicted emotion label. The visualizations are useful for gaining insights into the strengths and weaknesses of the model, as well as for identifying areas for improvement in future iterations of the project.

![IMAGE](https://github.com/ethanadp/Face-Emotion-AI-Detector/blob/main/Face_board.png)
