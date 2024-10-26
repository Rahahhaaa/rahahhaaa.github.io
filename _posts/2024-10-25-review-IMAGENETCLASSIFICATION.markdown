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

 객체인식의 성능을 놉히기 위해 더 큰 데이터셋을 수집하고 더 강력한 모델을 학습하고 과적합을 방지하기 위한 더 좋은 기술을 사용한다. 최근(당시 2012년)까지 레이블된 이미지 데이터셋은 상대적으로 작았다. (NORB, Caltech-101/245, CIFAR-10/100). 하지만 현실적인 설정의 객체들은 상당한 변동성을 보이기 때문에 인식하는 방식을 학습하려면 더 큰 훈련 세트를 사용해야 한다. ImageNet은 22000개의 카테고리에 1500만개의 레이블된 고해상도 이미지를 가진다.
 수백만개의 이미지로부터 수천개의 객체를 학습하기 위해 큰 학습 용량을 가진 모델이 필요하다. 하지만 객체 인식의 복잡성은 ImageNet 같은 큰 데이터셋으로도 특정될 수 없기 때문에 모델은 가지고 있지 않은 데이터를 보완하기 위해 많은 사전지식을 가지고 있어야 한다. 

### [2] Down-Sampling
 ImageNet 은 다양한 해상도의 이미지를 가지고 있는데 시스템은 고정된 input dimensionality를 필요로했다. 따라서 이미지들을 256 x 256 의 고정된 해상도로 down-sample하였다. 방식은 아래와 같다.
 1. 가로와 세로 중 더 짧은 면을 256이 되게 rescale
 2. 가운데 256 x 256을 crop
 
위 방식대로 down sample한 후 각 픽셀에서 훈련 세트의 mean activity를 뺀 것을 제외하고는 어떠한 pre-processing도 하지 않았다.

### [3] Architecture
##### 3.1 ReLU Nonlinearity

<p align = "center">
    <img src="/assets/img/ICWDC/relu.png" style = "width:400px; heigth:auto; onclick="window.open(this.src)"">
</p>

기존의 뉴런의 출력을 모델링 하는 방식은 tahn(x)나 sigmoid(x)였다. 그런데 이러한 saturating-linearity들은 non-saturating nonlinearity(ReLU...) 보다 훈련시간(gradient descent)이 훨씬 오래 걸린다. ReLU를 적용한 Deep convolutional nueral networks는 tanh를 적용한 것보다 훨씬 빨랐다. 위 그래프는 CIFAR-10데이터셋에 대한 훈련 에러를 25%까지 도달시키는데 6배 더 빨랐음을 보여준다.

##### 3.2 GPU
GTX 580 GPU하나는 3GB의 메모리밖에 없음 이는 하나의 GPU 위에서 훈련될 수 있는 네트워크의 최대 사이즈를 제한한다는 의미임. 120만개의 training example들이 네트워크들을 훈련시키기에 충분하다고 함 근데 이는 하나의 GPU에 fit 하기에는 너무 큼. 따라서 논문에서는 두개의 GPU에 걸쳐 네트워크를 분산시켰다. 현재의 GPU들은 호스트 머신 메모리를 통하지 않고 서로 다른 메모리에 직접 접근하여 읽고 쓰는 것이 가능하기 때문에 cross-GPU parrelization하기에 좋다.여기서 사용한 병렬화 방식은 기본적으로 각 GPU에 커널(뉴런)의 절반을 배치 **GPU들은 오직 특정 레이어에서만 communicate함** ( 예를들어 layer 3의 커널은 layer 2에 있는 모든 커널 맵에서 인풋을 취한다. 하지만 layer 4에 있는 커널은 오직 동일한 GPU에 있는 layer 3의 커널 맵에서만 입력을 받는다. ) 연결 패턴을 선택하는 건 교차 검증의 문제지만 이를 통해 허용 가능한 계산량의 일부가 될 때까지 communication의 양을 정확하게 조정할 수 있다. 

이를 통해 top-1, top-5 error rates를 각 1.7%, 1.2%까지 감소시킴. two-GPU net은 one-GPU net 보다 약간 더 빨리 train됐다.

##### 3.3 Local Response Normalization 
ReLU는 양수의 입력만을 그대로 사용한다. 그렇게 되면 매우 높은 하나의 픽셀값이 주변의 픽셀에 영향을 끼치게 되는데 이를 방지하기 위해 LRN을 사용하였다. Local Response Normalization은 실제 뉴런에서 발견되는 유형에서 영감을 받은 측면 억제 형태를 구현하여 다양한 커널을 사용하여 계산된 뉴런 출력 간에 큰 활동 경쟁을 불러일으킨다. 측면 억제란 신경생리학 용어로, 한 영역에 있는 신경 세포가 상호 간 연결되어 있을 때 한 그 자신의 축색이나 자신과 이웃 신경세포를 매개하는 중간신경세포(interneuron)를 통해 이웃에 있는 신경 세포를 억제하려는 경향이다. (위키백과, 더 알아보려면 헤르만격자를 찾아보자.) 

##### 3.4 Overlapping Pooling
<p align = "center">
    <img src="/assets/img/ICWDC/pooling.png" style = "width:400px; heigth:auto; onclick="window.open(this.src)"">
</p>
CNN에서 pooling을 사용하는 이유는 특성맵의 크기를 줄이기 위해서이다. ImageNet에서는 stride의 크기를 커널의 크기보다 작게 설정해 overlapping pooling을 적용하였다. 이렇게 해서 top-1과 top-5 에러를 줄였다고 한다.

##### 3.5 Overall Architecture
<p align = "center">
    <img src="/assets/img/ICWDC/arch.png" style = "width:600px; heigth:auto; onclick="window.open(this.src)"">
</p>
위에 나타나는 것처럼 네트워크는 가중치를 가지는 총 8개의 레이어로 구성되어있고 첫 5개는 conv layer 나머지 3개는 fc layer이다.마지막 fc layer의 출력은 1000-way softmax로 공급돼 1000개의 클래스 레이블에 대한 분포를 생성한다. 두번째 네번째 다섯번째 convolutional layer들은 동일한 GPU에 있는 이전 레이어의 커널 맵과만 연결되어있고 세번째 레이어만 두번째 레이어의 모든 커널맵과 연결되어있다. fc 레이어의 모든 뉴런들은 이전 레이어의 모든 뉴런과 연결되어있다. LRN은 첫번째 두번째 convolutional layer에만 따라온다. Max-pooling layer는 두 개의 response-normalization layer과 다섯번째 convolutional layer에 따라온다. ReLU non-linearity는 모든 convolutional layer와 fc layer에 적용된다.

###### Convolutional Layers
* 첫번째 레이어는 




