---
layout: post
title: "[CV 논문 리뷰] ImageNet Classification with Deep Convolutional Neural Networks"
date: 2024-10-25 00:00:00 +0800
description: CV 논문 리뷰 ImageNet Classification with Deep  # Add post description (optional)
img: architecture.jpg # Add image post (optional)
tags: [CV] # add tag
---

### [1] Abstract
 본 논문에서는 ImageNet LSVRC-2010 contest에 있는 120만개의 고해상도 이미지를 1000개의 클래스로 분류하기 위해 deep convolutional neural network를 학습시켰고 이전 모델들보다 훨씬 좋은 성능을 보였다. 이 neural network는 6천만개의 파라미터 65만개의 뉴런을 가지고 5개의 convolutional layer (max-pooling 레이어로 이어짐), 3개의 fc layer(마지막은 1000-way softmax)를 갖는다. 또한 훈련속도를 더 빠르게 하기 위해 non-saturating neuron과 어떤 GPU 기법을 사용하였고 과적합을 줄이기 위해 fc layer에 dropout 방식을 적용했다.

### [2] Introduction

### [3] Down-Sampling

### [4] Architecture


