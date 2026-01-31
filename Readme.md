# YOLOv1 From Scratch – End-to-End Object Detection

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## Overview
- This repository contains an end-to-end object detection pipeline implemented **from scratch** using the **YOLOv1 (You Only Look Once)** algorithm.  
- The project focuses on understanding and building the complete YOLOv1 workflow **without relying on high-level object detection frameworks**.

![YOLO](https://www.mygreatlearning.com/blog/wp-content/uploads/2020/07/2-1.png)

## Features
- YOLOv1 model implemented from scratch
- End-to-end training pipeline
- Custom YOLOv1 loss function
- Bounding box prediction and visualization
- Modular and readable code structure
- Config-driven architecture design

## YOLOv1 Overview

YOLOv1 formulates object detection as a **single regression problem**.

- The input image is divided into an **S × S grid**
- Each grid cell predicts:
  - Bounding boxes
  - Confidence scores
  - Class probabilities
- Detection is performed in a **single forward pass**

This repository follows the original YOLOv1 formulation as described in the paper by Redmon et al.

---

## Repository Structure

```text
.
├── config/             # Architecture & Training hyperparameters (YAML)
├── model/              
│   ├── dataset.py      # Custom DataLoader & grid-assignment logic
│   ├── loss.py         # Multi-part Sum-Squared Error (SSE) Loss
│   ├── yolo.py         # Darknet-inspired CNN Architecture
│   └── training_loop.py# Training & Validation cycles
├── util/               # IOU, NMS, and Visualization utilities
├── train.py            # Main entry point
└── requirements.txt    # Project dependencies
```

## Installation

- Clone the repository
```bash
git clone https://github.com/dhananjaybhole4/YOLO-from-scratch.git
cd YOLO-from-scratch
```
- Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

- Training
```bash
python3 train.py
```
- Model architecture and hyperparameters are defined in:
```bash
config/architecture.yaml
```

## Dataset

- Dataset should follow the YOLOv1 label format
- Each image is divided into grid cells during training
- Labels include bounding box coordinates and class probabilities
- Dataset loading and preprocessing are implemented in:
`model/dataset.py`

## Loss Function

The custom YOLOv1 loss function consists of:
- Localization loss
- Confidence loss for object presence
- Confidence loss for no-object cells
- Classification loss
Implemented in:
`model/loss.py`

## Limitations & Future Work

- YOLOv1 has limited performance on small objects
- No anchor boxes (original YOLOv1 design)
- No real-time inference node
- Future improvements:
  - Upgrade to YOLOv2 / YOLOv3
  - Anchor-based detection
  - ROS integration for real-time perception
