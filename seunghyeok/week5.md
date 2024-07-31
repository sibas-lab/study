## 학습한 내용
ref: [MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining](https://arxiv.org/abs/2403.13430v1)

> 이 논문은 원격 감지 분야에서 Multi-Task Pretraining(MTP)를 사용하여 다양한 작업에 대한 기반 모델의 성능을 향상시키는 방법을 탐구합니다. MTP는 의미 분할, 인스턴스 분할, 회전 객체 탐지 등의 다중 작업을 통해 사전 학습을 수행함으로써 모델의 다양한 RS 작업에 대한 적응력과 성능을 개선합니다. 광범위한 실험을 통해, MTP가 모델의 일반화 능력을 획기적으로 증가시키고, 여러 RS 다운스트림 작업에서 우수한 성능을 달성함을 보여줍니다.
### 다운스트림이란?
다운스트림 작업(Downstream Task)은 기계 학습 및 딥러닝 분야에서, 특정 모델이나 알고리즘이 사전 학습(Pretraining) 단계를 거친 후, 실제로 적용되는 구체적인 작업을 의미합니다. 사전 학습 단계에서는 대규모 데이터셋을 사용하여 모델이 일반적인 특징이나 패턴을 학습하도록 합니다. 이렇게 사전에 학습된 모델은 다양한 다운스트림 작업에서 특정 목표를 달성하기 위해 추가적으로 미세 조정(Fine-tuning) 됩니다.

다운스트림 작업의 예시
- 이미지 분류(Image Classification): 이미지에 포함된 객체의 종류를 분류하는 작업입니다. 예를 들어, 고양이, 개, 자동차 등의 이미지를 정확하게 분류하는 것이 목표입니다.
- 객체 탐지(Object Detection): 이미지 내에서 객체를 식별하고, 그 위치를 사각형 박스로 표시하는 작업입니다. 이 작업은 객체의 종류뿐만 아니라 위치 정보도 제공합니다.
- 시맨틱 분할(Semantic Segmentation): 이미지의 각 픽셀을 특정 클래스에 할당하는 과정으로, 이미지 내의 객체 경계를 더욱 세밀하게 식별할 수 있습니다.
- 자연어 처리(Natural Language Processing, NLP) 작업: 텍스트 분류, 감정 분석, 기계 번역 등과 같이 텍스트 데이터를 처리하는 작업도 다운스트림 작업의 예가 될 수 있습니다.

사전 학습된 모델을 다운스트림 작업으로 옮길 때 작업 불일치(Task Discrepancy)가 발생할 수 있는데, 이는 사전 학습 단계에서 모델이 학습한 패턴이나 특징이 다운스트림 작업의 요구 사항과 정확히 일치하지 않을 때 발생합니다. 예를 들어, 일반적인 이미지 분류 작업으로 사전 학습된 모델을 시맨틱 분할이나 객체 탐지와 같은 더 복잡한 작업에 적용하려 할 때, 모델의 성능이 기대만큼 나오지 않을 수 있습니다. 이러한 문제를 해결하기 위해, 모델을 특정 다운스트림 작업에 맞게 미세 조정하는 과정이 필요합니다.

### 자연 이미지와 RS 이미지란?

자연 이미지와 원격 탐사(Remote Sensing, RS) 이미지는 이미지의 출처와 특성에서 큰 차이를 가집니다. 자연 이미지는 일반적으로 인터넷, 사진, 비디오 등에서 얻은, 인간의 시각 시스템이 이해하기 쉬운 이미지들을 말합니다. 이러한 이미지들은 주로 사람, 동물, 풍경, 사물 등 자연에서 발견되는 객체들을 포함하고 있습니다. 대표적으로 ImageNet 같은 데이터셋이 자연 이미지를 포함하고 있으며, 다양한 컴퓨터 비전 태스크에서 널리 사용됩니다.
반면, 원격 탐사 이미지는 위성이나 항공기와 같은 원격 탐사 기술을 이용하여 지구의 표면을 관측하여 얻은 이미지입니다. 이러한 이미지는 주로 지리학, 기상학, 환경 모니터링, 도시 계획 등 다양한 응용 분야에서 활용됩니다. 원격 탐사 이미지는 자연 이미지와 달리 색상이 덜 다양하고, 공간 해상도가 낮으며, 대부분 새의 눈 뷰(bird’s-eye view)에서 촬영됩니다. 이러한 차이로 인해 원격 탐사 이미지는 도메인 갭(domain gap)이라고 불리는 자연 이미지와의 차이점을 가지며, 이로 인해 모델의 파인튜닝 성능에 영향을 줄 수 있습니다 .
따라서, 자연 이미지에 기반한 사전 학습 모델을 원격 탐사 이미지 작업에 적용할 때는 이러한 도메인 갭을 고려해야 하며, 모델이 원격 탐사 이미지의 특성을 잘 이해할 수 있도록 적절한 사전 학습 데이터셋 선택 및 파인튜닝 전략이 필요합니다.

### MTP 하는 이유는?

원격 탐사(Remote Sensing, RS) 기술을 통해 얻은 이미지(우리가 RS 이미지라고 부르는 것)를 컴퓨터가 잘 이해하고 분석할 수 있도록 만드는 것은 중요한 과제입니다. 예를 들어, 위성에서 찍은 지구의 사진을 통해 숲, 도시, 강 등을 자동으로 식별하려고 합니다. 이런 작업을 잘 수행하기 위해서는 컴퓨터 모델이 먼저 다양한 이미지를 보면서 학습을 해야 하는데, 이 과정을 '사전 학습'이라고 합니다.
그런데, 이전의 많은 연구들은 주로 한 가지 종류의 이미지만 가지고 모델을 학습시켰습니다. 예를 들어, 어떤 연구는 'MillionAID'라는 큰 데이터셋의 RGB 색상을 가진 항공 이미지만 사용했고, 다른 연구는 'Sentinel-2'라는 위성의 멀티스펙트럼 이미지만 사용했습니다. 최근에는 SAR(합성 개구 레이더) 이미지와 같이 다른 종류의 이미지도 함께 사용하기 시작했지만, 여전히 주로 RS 이미지만을 사용해 모델을 학습시켰습니다.

하지만, RS 이미지만으로는 모델이 세상을 이해하는 데 필요한 모든 정보를 배우기 어렵다는 문제점이 있습니다. 왜냐하면, RS 이미지는 특별한 관점에서 찍힌 사진이기 때문에, 일반적인 사진(자연 이미지)에서 볼 수 있는 다양한 상황이나 객체들의 모습을 학습하는 데 한계가 있기 때문입니다.

이 문제를 해결하기 위해, 일부 연구에서는 RS 이미지뿐만 아니라 일반적인 사진들(예: 인터넷에 널리 퍼져 있는 사진들)을 함께 사용해 모델을 학습시키는 방법을 시도했습니다. 이런 방식으로 학습된 모델은 일반적인 상황과 RS 상황 모두를 이해하는 데 더 좋은 성능을 보일 수 있습니다. 이를 위해 '교사-학생 프레임워크'라고 불리는 방법이 개발되었는데, 이는 기본적으로 일반 이미지로 학습된 지식을 RS 이미지 분석에도 적용할 수 있도록 돕는 방식입니다.
결론적으로, 이러한 접근 방식은 컴퓨터 모델이 RS 이미지 분석을 더 잘 수행할 수 있도록 하며, 우리가 위성 이미지를 통해 지구에 대해 더 많은 것을 이해하고 예측하는 데 도움을 줍니다.

### MTP의 목적
MTP의 목적은 다중 작업 학습을 통해 모델이 다양한 RS 작업에 대한 강력한 표현력을 학습하게 함으로써, 다운스트림 작업에 적용될 때 모델의 성능을 향상시키는 것입니다.


### MTP 과정
1. 사전 학습 데이터셋 준비: 다양한 RS 작업에 해당하는 여러 데이터셋을 준비합니다. 이 데이터셋은 회전 객체 탐지, 인스턴스 분할, 의미 분할 등과 같은 다양한 작업에 대한 레이블을 포함합니다.


2. 공유 백본 네트워크 선택: ViT-B + RVSA, ViT-L + RVSA, InternImage-XL과 같은 사전 학습된 모델들 중 하나를 백본 네트워크로 선택합니다. 이 네트워크는 이미지로부터 특징을 추출하는 기능을 담당합니다.


3. 다중 작업 학습: 선택된 백본 네트워크를 기반으로, 다중 작업을 위한 여러 디코더를 준비합니다. 각 디코더는 특정 RS 작업(예: 회전 객체 탐지, 인스턴스 분할, 의미 분할)에 최적화되어 있습니다.


4. 학습 실행: 준비된 데이터셋과 다중 작업 디코더를 사용하여 모델을 학습시킵니다. 이 과정에서 모델은 동시에 여러 RS 작업을 학습하며, 각 작업에 대한 예측을 생성합니다.
미세 조정을 위한 모델 전달: MTP를 거쳐 학습된 모델은 다양한 RS 다운스트림 작업에 적용될 준비가 됩니다. 이 모델은 장면 분류, 회전 객체 탐지, 변화 탐지, 수평 객체 탐지 등과 같은 특정 작업에 미세 조정될 수 있습니다.


5. 다운스트림 작업 미세 조정: 사전 학습된 모델을 특정 다운스트림 작업에 대한 데이터셋으로 미세 조정합니다. 이 단계에서 모델은 특정 작업에 대해 추가적으로 학습되어 성능을 향상시킵니다.


6. 성능 평가 및 적용: 미세 조정된 모델의 성능을 평가하고, 실제 RS 분석 작업에 적용합니다. 이를 통해 모델이 다양한 RS 작업에서 얼마나 잘 수행하는지를 확인할 수 있습니다.

### MTP 논문을 읽고 추가적인 메모

- 공유 인코더와 디코더를 통해 의미 분할뿐만 아니라 이미지 분류, 객체 탐지등 여러개의 테스크를 사전학습하여 범용적인 모델을 만드는 방법입니다.
- 하지만 대규모 데이터 세트에서는 미세조정을 수행하면 성능이 떨어지며 MAE를 하지 않고 MTP를 하면 기본 모델의 성능보다 떨어지므로 단계별 사전학습의 중요성을 강조합니다.
- 다양한 RS 이미지 해석을 위한 범용적인 모델에 적합한 기술 방법입니다.
