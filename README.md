# Classification of Concrete Bricks With or Without Cracks
A project implementing transfer learning to classify concrete slabs with or without crack, using MobileNetV2 on Keras

# Dataset
The project is based on a dataset from Mendeley Data. The dataset contains concrete images having cracks. The data is collected from various METU Campus Buildings. The dataset is divided into two as negative and positive crack images for image classification. Each class has 20000images with a total of 40000 images with 227 x 227 pixels with RGB channels. The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016). High-resolution images have variance in terms of surface finish and illumination conditions. 

# Model
We use MobileNetV2 as our base model while implementing Data Augmentation, namely Horizontal Flip and Rotation. We then add a Global Averaging Layer and a Dropout Layer to avoid overfitting.

# Accuracy
The training and validation accuracy are not far away from 90%.

# Citation
2018 – Özgenel, Ç.F., Gönenç Sorguç, A. “Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings”, ISARC 2018, Berlin.
