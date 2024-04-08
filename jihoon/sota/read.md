## 소타모델 분석

모델 구성

| Pretrain | Dataset |	Backbone |	Method |
|----------|---------|------------|-------|
| MAE + MTP |	SpaceNetv1 |	ViT-B＋RVSA |	UperNet |

MAE= masked autoencoders 약자이다
MTP= Multi-Task Pretraining 
 이는 모델이 다양한 작업에 대해 사전 훈련되었음을 의미할 수 있습니다. 이러한 멀티 태스크 사전 훈련 방식
 모델이 다양한 시나리오에서 유용한 특징을 학습하도록 도와, 최종 태스크의 성능을 향상시킬 수 있습니다.
모델 설정에서 보면 RVSA_MTP라는 타입이라고 정의 하는데 그부분 , "RVSA_MTP"는 Robust Visual Self-Attention Mechanism을 적용
하고 Multi-Task Pretraining으로 사전 훈련된 Vision Transformer 모델을 지칭하는 것인것 같습니다
UPerNet=Unified Perceptual Parsing for Scene Understanding
<img width="117" alt="image" src="https://github.com/jihoon2819/study/assets/85489123/fd465250-ee01-4c8f-9a59-3622d5787582">
