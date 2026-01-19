# YOLOv1 From Scratch – Object Detection Pipeline

## Overview
This repository contains an end-to-end object detection pipeline implemented from scratch using the YOLOv1 (You Only Look Once) algorithm. The project focuses on understanding and building the complete YOLOv1 workflow without relying on high-level object detection frameworks.

## Features
- YOLOv1 model implemented from scratch
- End-to-end training pipeline
- Custom YOLOv1 loss function
- Bounding box prediction and visualization
- Modular and readable code structure

## YOLOv1 Summary
YOLOv1 formulates object detection as a single regression problem. The input image is divided into an S×S grid, where each grid cell predicts bounding boxes, confidence scores, and class probabilities in a single forward pass.
