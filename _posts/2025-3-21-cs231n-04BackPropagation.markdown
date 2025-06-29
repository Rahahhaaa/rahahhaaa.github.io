rno---
layout: post
title: "[cs231n 정리노트] 4. Back-Propagation & Neural Networks"
date: 2025-03-21 00:00:00 +0800
description: # Add post description (optional)
img: # Add image post (optional)
tags: [CV] # add tag
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script><script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

오늘은 computational graphs라고 불리는 프레임워크를 사용해 무작위의 복잡한 함수에 대한 analytic gradient를 어떻게 계산하는지 얘기해봅시다.

## Computational graphs

[##_Image|kage@Xxqgv/btsMQ5AImyz/AAAAAAAAAAAAAAAAAAAAANPBv3gwg_I5xVgrkD9T1yqlBbAlfWMf99uSDXlyXOGN/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=U4dHEaHme%2BtPAiKUC7vO0SmTidY%3D|CDM|1.3|{"originWidth":588,"originHeight":241,"style":"alignCenter"}_##]

모든 함수들을 표현하기 위해 Computational Graphs를 사용할 수 있습니다. 여기서 노드는 각 계산과정들을 의미합니다. 위 그림에 보이는 예시는 Linear Classifier입니다. 입력 x와 W가 있고 곱하기 노드는 파라미터 W와 데이터 X에대한 행렬곱을 의미하며 출력으로 score 벡터를 내놓습니다. hinge loss 노드에서 data loss L\_i를 게산합니다. R노드에서 Regularization을 계산하고 덧셈 노드에서 둘을 합해 최종 Loss 값을 계산합니다.

## Backpropagation

역전파가 어떻게 동작하는지 아래의 쉬운 예제를 통해 알아봅시다.

[##_Image|kage@CIf5D/btsMT6SFrcN/AAAAAAAAAAAAAAAAAAAAADhkC6iKNEgeO_-i6T4FJGkfN-BHG7Upb-gX6ceUK_G4/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=AEVqbh4gWBtLVil%2Fj0QmQQtiksM%3D|CDM|1.3|{"originWidth":636,"originHeight":175,"style":"alignCenter"}_##]

모든 함수의 출력에대한 gradient를 계산해봅시다. 첫번째 단계는 항상 함수를 가지고 computational graph로 나타내는 것입니다.

위 슬라이드에서 오른쪽 사진이 그 computational graph 입니다. 이후 이 네트워크에 대한 순전파(forward pass)를 진행합니다.

[##_Image|kage@cuDSCb/btsMSF3cWJs/AAAAAAAAAAAAAAAAAAAAAEyOEte78ptIHrpSsckVv14fBVI6AUtF7geBUW85l5Gk/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=UEbEZpehj1rF9088N8%2BrayS87Ks%3D|CDM|1.3|{"originWidth":631,"originHeight":296,"style":"alignCenter"}_##]

계산이 끝난 값들에 대해 이름을 지어줍니다. 덧셈 노드의 출력에 대한 변수의 이름을 q(=x+y), 곱셈 노드의 출력에 대한 변수의 이름을 f(=qz)로 지어주겠습니다. 또 q와 f에대한 gradient를 계산해주겠습니다. 우리가 찾고자 하는 것은 x,y,z 에대한 f의 gradient입니다. 역전파는 chain rule를 재귀적으로 적용하는 것입니다. 그래서 맨끝에서의 gradient를 구하고 뒤로 돌아오면서 모든 gradient를 계산합니다.

[##_Image|kage@Il1oa/btsMSnonAai/AAAAAAAAAAAAAAAAAAAAAKsLi_QaXRsNCUUw62-nTdlsRUpIBZkIIRnlvZWRzRhF/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=QykBXfNH9acAvCc34jvHv%2Bs0YPw%3D|CDM|1.3|{"originWidth":623,"originHeight":293,"style":"alignCenter"}_##]

f에대한 f의 gradient를 구하면 당연하게도 1입니다. 

[##_Image|kage@bSf2tj/btsMTfbPRXZ/AAAAAAAAAAAAAAAAAAAAAOG4TWcPRHd9y31pWisEu44AhIrYbrGPLjsM7Tb-OutQ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=8oEfyuzQHq5iqSY9oEGQ4o1sCIY%3D|CDM|1.3|{"originWidth":630,"originHeight":296,"style":"alignCenter"}_##]

z에 대한 gradient를 구하게되면 q와 같고 q는 3이기 때문에 z에 대한 gradient는 3입니다.

[##_Image|kage@lGNGi/btsMTMG0d6S/AAAAAAAAAAAAAAAAAAAAAEvyv9WU9Dj2QZWsVApN_EaQORsTRAYPLUVzEfGFOCXm/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=0bbwbslOs0M2Y00yhIedI0hF4hg%3D|CDM|1.3|{"originWidth":627,"originHeight":306,"style":"alignCenter"}_##]

df/dq는 z와 같고 z는 -4이기 때문에 q에 대한 gradient는 -4입니다.

[##_Image|kage@uxHLp/btsMTIdEPbZ/AAAAAAAAAAAAAAAAAAAAAJ1-un5VzqfxTz3zMQ-si61cPiA6MSZoVQ_bS9z_7H9h/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=gK0fTOxBENBmM14kMmhwnfuEa9E%3D|CDM|1.3|{"originWidth":634,"originHeight":289,"style":"alignCenter"}_##]

df/dy를 구하려고 하는데 y는 직접적으로 f와 연결되어있지 않습니다. y는 중간노드인 q를 통해 f와 연결되어있습니다. 여기서 Chain rule을 사용하여 y에 대한 gradient를 구할 수 있습니다. dq/dy 는 1이고 df/dq 는 z 즉 -4입니다. 따라서 df/dy는 -4입니다.

[##_Image|kage@M5wRm/btsMTB6Ok33/AAAAAAAAAAAAAAAAAAAAAKSV8AK9MCll7CQgnWbkLdfiFiF-d2KFVn42mOUpoylB/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=MEAqPrT3ihBvxOUo9OSjNdhd1Z4%3D|CDM|1.3|{"originWidth":533,"originHeight":260,"style":"alignCenter"}_##]

df/dx도 마찬가지로 계산하여 -2라는 값을 얻을 수 있습니다. 

---

역전파에서 기본적으로 computational graph의 노드들은 모두 각각 주변 환경만을 인식하고있습니다. 그래서 각 노드는 노드에 연결된 local input들과 직접적으로 출력되는 local ouput을 가집니다.

[##_Image|kage@vD5bB/btsMSVLxb4N/AAAAAAAAAAAAAAAAAAAAAHBeox8jReRFVHaOj-C02RHHtvqrX3494BjCYq1_zWLD/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=sphJPGEKoqSndeCRGSarJeoFcZ8%3D|CDM|1.3|{"originWidth":589,"originHeight":288,"style":"alignCenter"}_##]

위 사진에서 입력은 x와 y, 출력은 z입니다.

[##_Image|kage@3npfg/btsMTH6Tqbv/AAAAAAAAAAAAAAAAAAAAABXEkra1wJCMu5wJz2YXAzpA-K7aMWV_dItIvaC_1N8m/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=BdKICHC3CxLlnLSUWAASt6hRaAk%3D|CDM|1.3|{"originWidth":600,"originHeight":274,"style":"alignCenter","filename":"blob"}_##]

또한 각 노드는 local gradient를 가집니다. x에대한 z의 gradient와 y에대한 z의 gradient를 계산해낼 수 있습니다. 위의 예에서 각 노드는 덧셈 혹은 곱셈 노드이기 때문에 복잡한 연산을 요구하지 않습니다. 

[##_Image|kage@55TXu/btsMS3CGwHO/AAAAAAAAAAAAAAAAAAAAALW3qf7pE0sPpaNvf6WevISTVabjbjFlwdaCQ0xuLTpl/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=3ATw%2FY5C3GfNi3PLPOuS4uSk%2Bx8%3D|CDM|1.3|{"originWidth":586,"originHeight":290,"style":"alignCenter"}_##]

역전파 과정에서는 맨 마지막 노드에서 시작해서 맨 첫번째 노드까지 돌아오게됩니다. 돌아오는 과정에서 각 노드에 도달할 때마다 현재 노드의 출력에대한 이전 노드에서의  gradient를 얻게됩니다. 그래서 역전파 과정에서 각 노드에 도달할 때에 이미 z 에대한 최종 Loss L의 gradient가 계산되어있습니다. 이후 우리가 원하는 것은 이전 노드에 대한 gradient입니다.

[##_Image|kage@ZZGEd/btsMSbapwVC/AAAAAAAAAAAAAAAAAAAAANFb9npmF1DfZy3SNHVdjZ7OzRyvkJIFLLAa4QSxKbbO/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=r5JypANiYwATMdaiu0e6iHdcN%2Bo%3D|CDM|1.3|{"originWidth":584,"originHeight":305,"style":"alignCenter"}_##]

이전에서 본 것처럼 chain rule을 사용하게 됩니다. 

[##_Image|kage@Y7VPc/btsMUl3eyTt/AAAAAAAAAAAAAAAAAAAAALBnpHtTPrXcQBNMFykH9l_YChgVdFdapq_eZD8d7QdO/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=%2F27bfyQHkNX%2FYI61ceWdvyqtLjs%3D|CDM|1.3|{"originWidth":624,"originHeight":310,"style":"alignCenter"}_##]

이후 gradient들을 현재 노드에 연결된 바로 이전 노드로 보내게됩니다.

여기서 가장 중요한 점은 각 노드에서 계산하는 Local Gradient들을 추적하고 역전파 과정에서 뒤노드 (upstream 노드) 에서 오는 gradient 값을 받아 local gradient와 곱해 연결된 노드로 보내고, 다음 노드에서는 이러한 주변환경만을 고려하여 또 다시 뒤로 이동한다는 것입니다. 

---

[##_Image|kage@t6l1L/btsMUxP3IHF/AAAAAAAAAAAAAAAAAAAAABNFsxxfMK9ey569b289P7Q8LyxmInYRLkIEhb99F8rl/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=lnvfgUs%2ByatNtOlwp0OZtKJkV0M%3D|CDM|1.3|{"originWidth":621,"originHeight":304,"style":"alignCenter"}_##]

위의 예시에 대해선 직접 계산해보시기 바랍니다.

[##_Image|kage@dYisV2/btsMSBUgK1D/AAAAAAAAAAAAAAAAAAAAADXHkcajjrez94xXGkLz-zl3koRk02Sx0CQcmXoyLBhK/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=hOkxC9jvU%2Bb66Dr3vmett7tnTG8%3D|CDM|1.3|{"originWidth":587,"originHeight":286,"style":"alignCenter"}_##]

한가지 짚고 넘어갈 부분은 computational graph를 만들 때 우리가 원하는 세분화된 계산 노드를 정의할 수 있다는 것입니다. 위의 예에서는 덧셈과 곱셈을 사용해 가장 간단한 방법으로 표현했습니다. 하지만 실제로는 여전히 local gradient 적어낼 수 있을 정도라면 이러한 노드들을 좀더 복잡한 노드로 묶을 수 있습니다. 하나의 예로 위의 sigmoid function이 있습니다. 이 함수는 이후 강의에서도 볼 수 있는 매우 흔한 함수입니다. 이 함수에 대한 gradient를 계산하면 꽤 괜찮은 expression을 얻을 수 있습니다.

computational graph에서 sigmoid 함수에 대한 local gradient 값을 알기 때문에 sigmoid 함수를 구성하는 노드들을 묶어 하나의 큰 노드로 대치할 수 있습니다. 여기서 중요한 점은 local gradient 값만 알 수 있다면 원하는 대로 노드를 합쳐도 된다는 것입니다. 그래서 이 모든 것은 기본적으로 더 간결하고 간단한 그래프를 얻기 위해 얼마나 많은 수학적인 계산을 하고 싶은지와 각 gradient들을 얼마나 단순하게 하고 싶은지 간의 균형을 맞추는 것입니다. 

---

[##_Image|kage@qfnfc/btsMUzmSO4q/AAAAAAAAAAAAAAAAAAAAAIaDC5UD8oGp8xE1ktjNqHgkF1UMLQYSmfCQDfZ2JX7C/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=aLbzFfOBLns7ZtUu9ACYJ%2FnQnUs%3D|CDM|1.3|{"originWidth":597,"originHeight":271,"style":"alignCenter"}_##]

역전파에서 add gate는 gradient 를 분배해줍니다.

max gate는 하나의 브랜치로만 gradient를 보내줍니다. 

mul gate는 다른 브랜치의 값으로 gradient를 scaling 해줍니다.

---

[##_Image|kage@bJKdN4/btsMSy4lLa9/AAAAAAAAAAAAAAAAAAAAACMo4wXHgtLoUlU0v5r2_14CwJc6K5cm1XYhHM6kNCqW/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=Pcb%2Flnyb0De%2F5rH%2F9DrFs4rBKRQ%3D|CDM|1.3|{"originWidth":516,"originHeight":299,"style":"alignCenter"}_##]

$$ \\frac{\\partial f}{\\partial x} = \\sum \\frac{\\partial f}{\\partial q\_i} \\frac{\\partial q\_i}{\\partial x}  $$

노드가 여러개의 브랜치와 연결되어있을 경우에는 모든 upstream gradient를 더해 계산합니다.

---

#### 벡터의 경우엔 어떻게 해야할까??

전체적인 흐름은 동일하지만 가장 큰 하나의 차이점은 gradient가 야코비 행렬이 된다는 것입니다.

[##_Image|kage@lRo8F/btsMTCSdVSR/AAAAAAAAAAAAAAAAAAAAAJxlaImxexcahgF2FLMgBtV46IRDk7CjkhPXxFbjr2wC/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=2yuQATZ2PGfGyyVoVkipX0IMkHA%3D|CDM|1.3|{"originWidth":567,"originHeight":239,"style":"alignCenter"}_##]

위와 같은 예시가 있다고 하였을 때 야코비 행렬의 사이즈는 4096 x 4096이 됩니다.

이 값은 매우 커보이지만 실제로는 미니배치를 사용하며 더 효율적이게 하기 위해 노드에 통째로 집어넣기 때문에 야코히 행렬의 사이즈는 (예를 들어 배치사이즈를 100으로 했을 경우에) 409600 x 409600 이됩니다.

사실 실제로는 대부분의 경우에 이렇게 큰 야코비 행렬을 계산할 필요는 없습니다. 

야코비 행렬은 element-wise하게 이루어지기 때문에 input의 각 요소는 대응하는 해당 ouput 요소에만 영향을 미칩니다. 따라서 야코비 행렬은 대각행렬이 될 것입니다. 실제로 야코비 행렬의 전체를 작성하고 공식화할 필요는 없으며, 출력에 대한 x의 영향을 알아내 gradient를 계산할 때 이 값을 입력하기만 하면 됩니다.

---

## **실습**

**아래 정리한 내용을 더 잘 이해하기 위해 직접 손으로 계산해보는 것을 추천드립니다.** 

[##_Image|kage@cubFdU/btsMZ42OrrS/AAAAAAAAAAAAAAAAAAAAAEvbSjhdKpPVdsJRQ6nFvEH935ZKpz30KwSgUGKyA03T/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=9f4apTCC7N9ikO3QxpcwnBV15oY%3D|CDM|1.3|{"originWidth":619,"originHeight":107,"style":"alignCenter","filename":"blob"}_##]

x는 n차원 벡터이고, W는 n x n 행렬이라고 해보겠습니다.

[##_Image|kage@cpViaZ/btsM0b8PUGt/AAAAAAAAAAAAAAAAAAAAACSXaUOjXnuTdjm-tAm2uBTLaxDu-d7an_14_dXSHu0t/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=%2Be5nt41l2D463FAM%2FUZBBjBXIB8%3D|CDM|1.3|{"originWidth":426,"originHeight":151,"style":"alignCenter"}_##]

우선 첫번째로 위의 예시를 computational graph로 나타내보겠습니다.

[##_Image|kage@6FrpO/btsM2yOBRii/AAAAAAAAAAAAAAAAAAAAAJ0UYgoIs-uLhee9spM0EIImaqvdXy9-3JOvdYV-ymSv/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=xMJwmN7CYcfQvlf7NuSNa8aIVEE%3D|CDM|1.3|{"originWidth":740,"originHeight":373,"style":"alignCenter"}_##]

W와 x에 임의의 값을 넣어주고 주어진 수식에 맞게 순전파를 진행하여 값들을 얻을 수 있습니다. 

이제 역전파를 진행해보겠습니다.

[##_Image|kage@z7rwR/btsM0Ln7UXN/AAAAAAAAAAAAAAAAAAAAAPKVFJ2sCe0_iX2zlAb650RMMXIjAc3FKxtPZXLAyOhP/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=ZZId55ulhNoIlmN7y7KZJY3457g%3D|CDM|1.3|{"originWidth":182,"originHeight":75,"style":"alignCenter","width":294,"height":121}_##]

첫번째로 당연하게도 f에 대한 f의 gradient는 1입니다. 

$ f(q) = q\_1^2 + \\dots q\_n^i$ 이므로 q\_i에 대한 f의 gradient는 다음과 같습니다.

$$ \\frac{\\partial f}{\\partial q\_i} = 2q\_i $$

$$\\nabla\_q f = 2q$$

[##_Image|kage@r520B/btsM049fU1f/AAAAAAAAAAAAAAAAAAAAALCkqgF22hIG0RCiBBsarj1nw_0LLbN-qX1bgLWTtIQW/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=qGzMbTIIbfAxBKz0%2Bkx69t7Jkes%3D|CDM|1.3|{"originWidth":336,"originHeight":134,"style":"alignCenter"}_##]

이제 q에 대한 f의 gradient를 얻었습니다. 바로 다음으로 넘어가 W에 대한 f의 gradient를 구해보겠습니다.

구하는 과정에 아래와같이 앞서 배운 chain rule을 사용할 것입니다. 

$$ \\frac{\\partial f}{\\partial W} = \\frac{\\partial f}{\\partial q} \\frac{\\partial q}{\\partial W}  $$

좌변의 $\\frac{\\partial f}{\\partial q}$는 이미 위에서 구했기 때문에 $\\frac{\\partial q}{\\partial W}$만 구하면 됩니다. 

우선 $q\_k = W\_{k,1}x\_1 + \\dots + W\_{k,n}x\_n$ 이므로 

$\\frac{\\partial q\_k}{\\partial W\_{i,j}} = 1\_{k=i}x\_j$ 입니다.

따라서 

$$ \\frac{\\partial f}{\\partial W\_{i,j}} = \\sum\_{k} \\frac{\\partial f}{\\partial q\_k} \\frac{\\partial q\_k}{\\partial W\_{i,j}} = \\sum\_{k} (2q\_k)(1\_{k=i}x\_j) = 2q\_ix\_j $$

$$ \\nabla\_w f =2q \\cdot x^t $$

위 수식을 바탕으로 값을 계산해보겠습니다. 

$$ 2q \\cdot x^T = 2\\begin{pmatrix}0.22 \\\\ 0.26\\end{pmatrix} \\cdot \\begin{pmatrix}0.2 &  0.4\\\\\\end{pmatrix} = \\begin{pmatrix}0.088 & 0.176 \\\\0.104 & 0.208 \\\\\\end{pmatrix} $$

[##_Image|kage@eoRkQ6/btsM1i7hIgM/AAAAAAAAAAAAAAAAAAAAAFaxLNM3rz5JLZ0tGBgm55x0t6dfcgcuE4ozy2lfmdub/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=DqPPNbH4bNcUq7OxFvXjdZR3c4E%3D|CDM|1.3|{"originWidth":245,"originHeight":95,"style":"alignCenter","width":392,"height":152}_##]

다음으로 넘어가 x에 대한 f의 gradient를 계산해보겠습니다. 

위에서 한 방식과 동일하게 chain rule을 이용합니다. 따라서 먼저 $\\frac{\\partial q\_k}{\\partial x\_i}$를 구해보겠습니다. 

$q\_k = W\_{k,1}x\_1 + \\dots + W\_{k,n}x\_n$ 이므로

$\\frac{\\partial q\_k}{\\partial x\_i} = W\_{i,x}$입니다.

따라서

$$ \\frac{\\partial f}{\\partial x\_i} = \\sum\_{k} \\frac{\\partial f}{\\partial q\_k} \\frac{\\partial q\_k}{\\partial x\_i} = \\sum\_{k} 2q\_kW\_{k,i}$$

$$ \\nabla\_xf = 2W^T \\cdot q $$

이어서 위 수식을 바탕으로 값을 계산해보겠습니다.

$$2W^T \\cdot q = 2 \\begin{pmatrix}0.1 & -0.3 \\\\0.5 &  0.8\\\\\\end{pmatrix} \\cdot \\begin{pmatrix}0.22 \\\\0.26\\end{pmatrix} = \\begin{pmatrix}-0.112 \\\\ 0.636\\end{pmatrix}$$

[##_Image|kage@bqYrbO/btsM1qEgIVU/AAAAAAAAAAAAAAAAAAAAAEdGmpRVrHuowodhIQig-TAwqSEfPN9r-99XVEWGqtFf/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=99P7R%2BlX%2B3odEtKnhIYZkD%2BSOs4%3D|CDM|1.3|{"originWidth":190,"originHeight":91,"style":"alignCenter","width":388,"height":186}_##]

#### 짜잔

[##_Image|kage@bKcZSg/btsM1oGrspC/AAAAAAAAAAAAAAAAAAAAAI8ZucQkb3khGS2Djv8PDm99ZQwK8MvbFLzMeBj6wotS/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=sHjyNvj6S%2BBXFkxP6rOFDetKjKY%3D|CDM|1.3|{"originWidth":713,"originHeight":233,"style":"alignCenter"}_##]

---

## **Modularized implementation : forward / backward API**

우리가 위에서 한 방식들은 모듈화된 구현입니다. computational graph에서 각 노드들 보고 local gradient를 계산하고 upstream gradient와 chain합니다. 그래서 이걸 forward API와 backward API로 생각할 수 있습니다. forward pass에서는 그 노드의 출력값을 계산하고 backward pass에서는 gradient를 계산하도록 구현하면됩니다. 그래서 실제로 우리가 코드로 구현할 때 동일한 방식으로 구현하게됩니다.  

```
class ComputationalGraph(object):
    #...
    def forward(inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph:
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss
    def backward():
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward() # little piece of backprop (chain rule applied)
        return inputs_gradients
```

만약 전체 그래프를 가지고 있을 경우 forward pass를 위상정렬된 그래프 내의 노드를 iterate함으로써 구현할 수 있습니다.

backward pass를 구현할때는 역위상정렬된 그래프의 노드를 iterate하며 backward 함수를 call하면됩니다. 

####  예시

[##_Image|kage@bQRtVB/btsM11KJ2AP/AAAAAAAAAAAAAAAAAAAAAHKHJWpC9D_gdKK3N2L7WyvnlUubYNwFt3aCaywRIE4z/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=wU3fRXJ0QEWx5PGUoKEak7UJvbU%3D|CDM|1.3|{"originWidth":1210,"originHeight":453,"style":"alignCenter"}_##]

[https://github.com/BVLC/caffe/tree/master/src/caffe/layers](https://github.com/BVLC/caffe/tree/master/src/caffe/layers)

 [caffe/src/caffe/layers at master · BVLC/caffe

Caffe: a fast open framework for deep learning. Contribute to BVLC/caffe development by creating an account on GitHub.

github.com](https://github.com/BVLC/caffe/tree/master/src/caffe/layers)

caffe라고 유명한 딥러닝 프레임워크인데 layer들을 보면 위와 비슷한 모듈화 방법을 따르는 것을 볼 수 있습니다. 

#### sigmoid gate

[##_Image|kage@clnrxA/btsM13Pjqoy/AAAAAAAAAAAAAAAAAAAAAFaVD86yfUF8tG1TUdpuE1dgfnQRKAfLG5Ufz6ZoB6fl/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=8x9vKbbTePmyLKdZiifUS48HAWw%3D|CDM|1.3|{"originWidth":1126,"originHeight":577,"style":"alignCenter"}_##]

sigmoid layer를 한번 살펴보겠습니다. forward pass는 동일하게 sigmoid를 계산하고 backward 에서는 top\_diff(upstream gradient)를 input을 local gradient와 계산합니다.

---

## **Neural Networks**

이제 Nerual Netowrk(인공신경망)에 대해 알아봅시다. 보통 사람들이 인공신경망에 대해 말할 때 뇌나 생물학적인 여러 내용들의 유사점을 끌고오는 경우가 많은데 이번에는 그런 것들 제외하고 단순히 함수로서 봐봅시다.

[##_Image|kage@cZFxtB/btsM0KDni6p/AAAAAAAAAAAAAAAAAAAAAIddtf2VLIM9FbIkNKbvH7WM32Jb7nF1PJ3FCIPdv_RE/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=FhlmqCzs1EbmevREo2eychYVoFg%3D|CDM|1.3|{"originWidth":646,"originHeight":63,"style":"alignCenter"}_##]

지금까지는 위에 보이는 것처럼 linear score function들을 다뤘습니다.  
인공신경망의 가장 단순한 형태는 두 개의 함수를 이어서 구성하는 것입니다.

[##_Image|kage@J9vjn/btsM2GfqRss/AAAAAAAAAAAAAAAAAAAAAOQrdYJG3slSonf1uNYnqSJyeGhaeCt8mfkRtg4uG9R_/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=e2czNhV5ngoC4skHPGcr3RtwhJ8%3D|CDM|1.3|{"originWidth":817,"originHeight":228,"style":"alignCenter"}_##]

먼저 W1과 x의 행렬곱을 계산한 후, 그 결과를 비선형 함수인 maxx(0, W1x)에 통과시켜 최종 출력값을 얻습니다.

비선형성은 매우 중요합니다. 선형 레이어만 계속 쌓다보면 결국 전체 모델이 하나의 선형함수로 수렴하게 됩니다.

신경망은 여러개의 간단한 함수들이 서로 겹쳐 쌓여져 있는 함수의 한 종류로, 이러한 함수들이 계층적으로 배열되어 더 복잡한 비선형 함수를 형성합니다. 즉, 여러 단계의 계층적 계산을 통해 복잡한 문제를 해결할 수 있는 아이디어를 담고있습니다.

**인공신경망을 만드는 주요한 방법은 행렬곱 같은 선형 레이어들을 반복해서 쌓고, 그 사이에 비선형 함수를 포함시키는 것입니다.**

---

[##_Image|kage@cYKBJM/btsM1JDJgaT/AAAAAAAAAAAAAAAAAAAAAC_1IUzw15zgRQqyC3KrjtikzsJYAz7DGzkw6RQ9GEnu/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=XUtZXv9NdVpxPxdKIfdwCos5K44%3D|CDM|1.3|{"originWidth":787,"originHeight":109,"style":"alignCenter"}_##]

이전에 다뤘던 linear score function을 생각해보면, weight matrix W의 각 행이 input에 대한 특정 클래스에 대해 기대하는 부분을 표현하는 template 역할을 한다고 말했습니다. 그때 얘기했던 한가지 문제점은 하나의 클래스가 오직 하나의 템플릿을 가진다는 것이었습니다.

다층 네트워크에서는 각 중간 변수인 h나 W1이 여전히 이러한 템플릿 역할을 할 수 있습니다. 그러나 이제 우리는 h에 있는 이 템플릿들에 대한 모든 점수를 가지고 있고, 그 위에 다른 레이어를 두어 이들을 결합할 수 있습니다. 따라서 이제 자동차 클래스는 빨간색 차와 노란색 차와도 연관지을 수 있습니다. 그 이유는 W2 행렬이 h에서 얻은 벡터들을 가중합하여 이를 조합하는 역할을 하기 때문입니다.

> Q. W1처럼 W2도 이미지로 표현가능합니까?  
> A. W1은 직접적으로 input image와 연결되어있기 때문에 해석이 가능하지만 h는 각 템플릿에 대한 이미지의 점수이기 때문에 표현되지 않습니다.  
>   
> Q. 그럼 이제 W1은 10개가 아니라 더 많은 템플릿을 가지는 겁니까?  
> A. 그렇습니다. 예를 들어 왼쪽을 바라보는 말, 오른쪽을 바라보는 말 모두 W1에 포함될 수 있습니다. W2는 이 모든 템플릿들에 대한 가중합을 해 특정 클래스에 대한 최종 점수를 얻습니다.   
>   
> Q. 만약 입력 이미지가 왼쪽을 바라보는 말이고 W1에는 왼쪽을 바라보는 말, 오른쪽을 바라보는 말 둘다 포함되어있으면 어떤 일이 생깁니까?  
> A. 우선 h에서 왼쪽을 바라보는 말에 대한 점수가 굉장히 높을 것입니다. 반면에 오른쪽을 바라보는 말에 대한 점수는 낮을 것입니다. W2는 가중합이기 때문에 특정 템플릿에서 매우 높은 점수를 받거나, 두 개의 템플릿에서 각각 낮은 점수와 중간 점수를 받는 경우라도, 최종적으로 높은 점수가 나올 수 있습니다. 결국, 특정 유형의 말이 존재하면 전반적으로 높은 점수를 받는 경향이 생기게 됩니다.  
>   
> Q. 어디가 비선형 함수입니까??  
> A. 보통 비선형함수는 h 바로 직전에 있습니다. 따라서 h는 비선형 함수로 생긴 값입니다. (위 그림 예시에서는 max 함수)

---

방금 전까지 2-layer짜리 신경망을 다뤘는데 우리는 임의의 깊이의 더 깊은 신경망을 얻기 위해 레이어들을 더 쌓을 수 있습니다. 

[##_Image|kage@kyGKc/btsM9GtRLmy/AAAAAAAAAAAAAAAAAAAAAOVVA_1agRMZOM0hmn2KyhRyt2GajJQo-2Vc0y5hAMqR/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=kAKP9hj%2B49sYKsnd%2Ba5qo%2B53%2FbI%3D|CDM|1.3|{"originWidth":504,"originHeight":71,"style":"alignCenter"}_##]

---

2-layer 신경망은 20줄 정도만으로 구현 가능합니다.

```
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = rand(N, D_in), randn(N,D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))
    
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
```

---

## **Biological Inspiration**

신경망에 대하 얘기할 때 주로 생물학적 연관성이 언급되곤 합니다. 생물학적인 유사도가 그렇게 높진 않지만, 이러한 연관성과 영감 중 일부가 어디에서 오는지 이해하는 것도 꽤 흥미로울 수 있습니다.

[##_Image|kage@cDnB8M/btsM9T7NBSs/AAAAAAAAAAAAAAAAAAAAAIFQ4831BTgEB9-btHRF1dVuPB9vtOi8MyTeKL2ilCUw/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=PKn0hvQl3gG0%2F2MhF5DhQScvMaM%3D|CDM|1.3|{"originWidth":488,"originHeight":234,"style":"alignCenter"}_##]

단순하게 뉴런에 대해 생각해보겠습니다. 뉴런에는 자극이란 게 존재하고 이 자극들은 각 뉴런들을 향해 전달됩니다. 연결된 수많은 뉴런들이 존재하고 각 뉴런들은 뉴런에 들어오는 자극을 받는 dendrite(가지돌기)를 가지고 있습니다. 또한 cell body(신경세포체)는 가지돌기로 들어온 자극들을 통합하고 이후에 axon(축삭돌기)를 통해 연결된 다음 뉴런으로 자극을 보냅니다. 

[##_Image|kage@pHar2/btsNacTvJQa/AAAAAAAAAAAAAAAAAAAAACAs2LWtRYE4oMPlARBKQcg3J-f1-D-f266R3lstRbHC/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=hqdFBDAR2fNCEzQ1S%2FmJFYP7SJE%3D|CDM|1.3|{"originWidth":490,"originHeight":267,"style":"alignCenter"}_##]

지금까지 공부했던 내용들을 살펴보면, 각각의 computational node에서 뉴런과의 유사성을 볼 수 있습니다. 노드들은 서로 연결되어있고 input은 뉴런으로 들어오는 자극과 동일합니다. 각 x0, x1, x2는 예를 들어 가중치 W를 통해 통합됩니다. 이후 activation function을 통해 얻은 값을 출력으로 내보내게 됩니다. 

[##_Image|kage@bnRGOv/btsM9DjQwyr/AAAAAAAAAAAAAAAAAAAAAFJNXb7lA_OGLX4jQZh2Xxoc_EoDKPq5_2VI1dpY6PEJ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=IeywESF9NQKWulEFjerjXkbFfBI%3D|CDM|1.3|{"originWidth":576,"originHeight":192,"style":"alignCenter"}_##]

위의 활성화 함수를 보면 기본적으로 모든 input 값들을 취해서 하나의 숫자를 출력합니다. 이전에 활성화 함수의 예 중 하나로 sigmoid activation function과 여러 비선형성을 다뤘습니다. 

[##_Image|kage@JlnHJ/btsNaNk5jZV/AAAAAAAAAAAAAAAAAAAAAPcz_AhHDpAUefQh6h86aQBgiVZrRL2M9S15ii1wlNNS/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=AqhpkN23m8dMBNk7vgh1Eb7GWdc%3D|CDM|1.3|{"originWidth":586,"originHeight":145,"style":"alignCenter"}_##]

생각해낼 수 있는 조금 느슨한 비유 중 하나는 이런 비선형성이 뉴런의 firing이나 spiking rate를 표현할 수 있다는 것입니다. 뉴런의 연결된 뉴런으로의 신경전달은 이런 discrete spikies를 사용합니다. 만약 spiking이 매우 빠르다면 이후에 전달되는 강한 신호를 갖습니다. 

실제로 이런 내용들을 연구하는 신경학자들은 뉴런들의 동작과 가장 비슷한 비선형성의 한 종류가 ReLU 함수라고 말합니다. 이후에 다룰 함수고 이 함수는 모든 음의 값 입력에 대해 0이고 양의 값에 대해서는 선형 함수입니다.

지금까지 다룬 생물학적인 내용을 만들어낼 때 굉장히 조심해야합니다. 실제 생물학적 뉴런은 훨씬 더 복잡합니다. 다양한 종류의 뉴런들이 존재하고 가지돌기들은 정말 복잡한 비선형성 계산을 수행할 수 있습니다. 또한 시냅스들은 단순히 하나의 가중치가 아니라 복잡한 비선형 동적 시스템입니다. 또 활성화 함수를 rate code 나 firing rate로 해석하는 것은 적합하지 않습니다. 

---

## **Activation Functions**

[##_Image|kage@chd3N0/btsM9CSLMco/AAAAAAAAAAAAAAAAAAAAAIm1cpr6IQQNrjKheY4Iyz-Z1eOcLqppQhZjLY-Rn0kL/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=rzR96SUC5rY5846vK3C1Qt0e4aY%3D|CDM|1.3|{"originWidth":609,"originHeight":252,"style":"alignCenter"}_##]

많은 활성화 함수들이 존재하고 이 함수들에 대해선 이후에 자세하게 다뤄보겠습니다.

---

[##_Image|kage@FNAHI/btsNakwWXM8/AAAAAAAAAAAAAAAAAAAAAAu_6dKOfdKRhkqHpGPdlhWpZwM-ko7n_X9RJbGa8Rpv/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=4jtLt8eM7Nwkf%2B6ZTxa77Lhpu2U%3D|CDM|1.3|{"originWidth":635,"originHeight":262,"style":"alignCenter"}_##]

또 여러 종류의 신경망 아키텍처에 대해 다룰것입니다.

왼쪽 위의 예시를 우리는 2-layer Neural Net라고 불렀는데 1-hidden-layer Nueral Net이라고도 부를 수 있습니다. 행렬곱 횟수를 세는 것 대신에 hidden layer의 개수를 셉니다. 두 용어 다 사용해도 되고 보통은 2-layer Neural Net을 흔히 사용합니다. 오른쪽 예도 마찬가지로 3-layer Neural Net 이나 2-hidden0layer Neural Net이라고 부를 수 있습니다. 

---

## **Example feed-forward computation of a neural network**

[##_Image|kage@bxj1oW/btsNaNMddqS/AAAAAAAAAAAAAAAAAAAAAI1LttcPDaRA_rylCiyU7w_KtIXqF_333hONcVKt_50p/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=wORwxiVL11l%2FEwppYxq4sT4sezs%3D|CDM|1.3|{"originWidth":567,"originHeight":193,"style":"alignCenter"}_##]

신경망에서 순전파를 할 때, 네트워크 안의 각 노드는 이전에 보여줬던 뉴런과 같은 연산을 수행합니다.

은닉층은 여러 개의 뉴런들로 이루어진 하나의 벡터(또는 배열) 처럼 생각할 수 있고, 이 뉴런들의 출력을 계산할 때는 행렬곱셉을 사용해 효율적으로 한번에 전체 층의 출력을 계산할 수 있습니다. 예를 들어, 은닉층에 10개나 50개 혹은 100개의 뉴런이 있더라도 한번의 행렬곱 연산으로 이 모든 뉴런의 출력을 구할 수 있다는 뜻입니다.

[##_Image|kage@piSTV/btsNafvUUCh/AAAAAAAAAAAAAAAAAAAAAIKJaLhvw1lc1yIUDHfdDk9SL4CDNODcY-O3eNtxIR2-/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1751295599&amp;allow_ip=&amp;allow_referer=&amp;signature=dbhlZFA2uc%2BM53%2F1XKl9yvz%2F8VI%3D|CDM|1.3|{"originWidth":603,"originHeight":269,"style":"alignCenter","filename":"blob"}_##]

다시 정리해 보면, 신경망을 행렬-벡터 형태로 표현하면 위 그림처럼 구성됩니다. 먼저 입력 데이터 x (입력 벡터)에 첫번째 가중치 행렬 W1을 곱합니다. 그리고 그 결과에 비선형 함수 f (sigmoid 함수)를 적용합니다. 그다음 두 번째 가중치 행렬을 곱해서 두번째 은닉층 h2를 계싼하고 마지막에는 출력층에 도달하게 됩니다. 이런식으로 신경망의 순전파를 구성할 수 있고 역전파를 사용해서 가중치에 대한 기울기들으 계산하면 신경망을 학습시킬 수 있습니다. 이것이 기본적으로 신경망의 핵심 개념입니다. 

## **끝!**