# EPN-SUNRGBD: Joint Classification and Amodal Bounding Box Detection

### Introduction
This work is based on our paper (https://www.researchgate.net/publication/333676944_Pushing_Boundaries_with_3D_Boundaries_for_Object_Recognition). We propose a novel architecture named \textit{Edge-Aware PointNet}, that incorporates complementary edge information with the recently proposed PointNet++ framework, by making use of convolutional neural networks (CNNs) that jointly infers object class and an amodal bounding box. This work is an extension to the original EPN repository that detects amodal boxes in addition to object class.

![prediction example](https://github.com/Merium88/EPN-SUNRGBD/blob/master/doc/method.jpg)

In this repository, we release code and data for training the network Edge-Aware PointNet on point clouds sampled from 3D shapes.

### Usage
The code is written as an extension to the original PointNet++ thus the usage and training procedure is the same as for the original repository. (https://github.com/charlesq34/pointnet2)
To train a model to classify point clouds sampled from ModelNet40:

        python train_modelnet40_edgecnn_sunrgbd.py



