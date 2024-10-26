---
layout: post
title: "[CV 논문 리뷰] ImageNet Classification with Deep Convolutional Neural Networks"
date: 2024-10-25 00:00:00 +0800
description: CV 논문 리뷰 ImageNet Classification with Deep  # Add post description (optional)
img: ICWDC/architecture.jpg # Add image post (optional)
tags: [CV] # add tag
---

### [1] Introduction
 본 논문에서는 ImageNet LSVRC-2010 contest에 있는 120만개의 고해상도 이미지를 1000개의 클래스로 분류하기 위해 deep convolutional neural network를 학습시켰고 이전 모델들보다 훨씬 좋은 성능을 보였다. 이 neural network는 6천만개의 파라미터 65만개의 뉴런을 가지고 5개의 convolutional layer (max-pooling 레이어로 이어짐), 3개의 fc layer(마지막은 1000-way softmax)를 갖는다. 또한 훈련속도를 더 빠르게 하기 위해 non-saturating neuron과 어떤 GPU 기법을 사용하였고 과적합을 줄이기 위해 fc layer에 dropout 방식을 적용했다.

### [2] Down-Sampling
 ImageNet 은 다양한 해상도의 이미지를 가지고 있는데 시스템은 고정된 input dimensionality를 필요로했다. 따라서 이미지들을 256 x 256 의 고정된 해상도로 down-sample하였다. 방식은 아래와 같음
 1. 가로와 세로 중 더 짧은 면을 256이 되게 rescale
 2. 가운데 256 x 256을 crop
 
위 방식대로 down sample한 후 각 픽셀에서 훈련 세트의 mean activity를 뺀 것을 제외하고는 어떠한 pre-processing도 하지 않았음

### [3] Architecture
##### 3.1 ReLU Nonlinearity

<p align = "center">
    <img src="/assets/img/ICWDC/relu.png" style = "width:400px; heigth:auto">
</p>

기존의 뉴런의 출력 f를 모델링 하는 방식은 tahn(x)나 sigmoid(x)였음. 그런데 이러한 saturating-linearity들은 non-saturating nonlinearity(ReLU...) 보다 훈련시간(gradient descent)이 훨씬 오래 걸린다. ReLU를 적용한 Deep convolutional nueral networks는 tanh를 적용한 것보다 훨씬 빨랐다. 위 그래프는 CIFAR-10데이터셋에 대한 훈련 에러를 25%까지 도달시키는데 6배 더 빨랐음을 보여준다.

##### 3.2 GPU
GTX 580 GPU하나는 3GB의 메모리밖에 없음 이는 하나의 GPU 위에서 훈련될 수 있는 네트워크의 최대 사이즈를 제한한다는 의미임. 120만개의 training example들이 네트워크들을 훈련시키기에 충분하다고 함 근데 이는 하나의 GPU에 fit 하기에는 너무 큼. 따라서 논문에서는 두개의 GPU에 걸쳐 네트워크를 분산시켰다. 현재의 GPU들은 호스트 머신 메모리를 통하지 않고 서로 다른 메모리에 직접 접근하여 읽고 쓰는 것이 가능하기 때문에 cross-GPU parrelization하기에 좋다.여기서 사용한 병렬화 방식은 기본적으로 각 GPU에 커널(뉴런)의 절반을 배치 **GPU들은 오직 특정 레이어에서만 communicate함** ( 예를들어 layer 3의 커널은 layer 2에 있는 모든 커널 맵에서 인풋을 취한다. 하지만 layer 4에 있는 커널은 오직 동일한 GPU에 있는 layer 3의 커널 맵에서만 입력을 받는다. ) 연결 패턴을 선택하는 건 교차 검증의 문제지만 이를 통해 허용 가능한 계산량의 일부가 될 때까지 communication의 양을 정확하게 조정할 수 있다. 


이를 통해 top-1, top-5 error rates를 각 1.7%, 1.2%까지 감소시킴. two-GPU net은 one-GPU net 보다 약간 더 빨리 train됨.

##### 3.3 Local Response Normalization




