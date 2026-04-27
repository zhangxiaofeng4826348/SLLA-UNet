# SLLA-UNet
Self-Supervised Multi-task Learning Framework for Ultrasound-based Subpleural Pulmonary Lesion Diagnosis

This repository provides the official implementation of a self-supervised multi-task deep learning framework for segmentation and classification of subpleural pulmonary lesions (SPLs) in ultrasound images.

---

## Overview
The overall pipeline consists of two stages: (1) self-supervised pretraining, and (2) supervised fine-tuning for joint segmentation and classification.
We propose a unified framework that integrates:

- Self-supervised contrastive learning for representation pretraining
- U-Net-based segmentation
- Transformer-enhanced feature extraction (Swin Transformer)
- Multi-task learning for joint segmentation and classification
-<img width="1041" height="753" alt="mode structurel" src="https://github.com/user-attachments/assets/056959a6-0b85-424d-91cf-9105179b7c86" />

The framework is designed to improve diagnostic performance and robustness across multi-center datasets.

---

## Project Structure

