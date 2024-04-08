이 섹션은 모델의 학습 일정, 옵티마이저, 스케줄러 설정, 그리고 학습 도중 사용할 다양한 훅들에 대한 설정을 포함하고 있습니다. 각 구성 요소를 자세히 살펴보겠습니다.
### 옵티마이저 설정 (optim_wrapper)

```yaml
# optimizer

optim_wrapper = dict(
    optimizer=dict(
    type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor_ViT', 
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.9,
        )
        )
```

- **옵티마이저**: 
  - **type**: `'AdamW'`는 Adam 옵티마이저의 변형으로, 가중치 감소(weight decay) 전략을 개선한 버전입니다.
  - **lr**: `6e-5`로 설정된 학습률입니다. 이는 가중치 업데이트 시 적용되는 스텝 크기를 결정합니다.
  - **betas**: `(0.9, 0.999)`는 첫 번째와 두 번째 모멘텀 계산에 사용되는 감쇠율입니다.
  - **weight_decay**: `0.05`는 가중치 감소율로, 오버피팅을 방지하기 위해 사용됩니다.
- **constructor**: `'LayerDecayOptimizerConstructor_ViT'`는 Vision Transformer 모델을 위한 레이어별 학습률 감소를 구현하는 옵티마이저 생성자입니다.
- **paramwise_cfg**: 
  - **num_layers**: `12`는 모델의 총 레이어 수입니다.
  - **layer_decay_rate**: `0.9`는 더 깊은 레이어로 갈수록 학습률을 감소시키는 비율입니다.

### 학습률 스케줄러 설정 (param_scheduler)

```yaml
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        T_max=78500,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]
```

- **LinearLR**:
  - **type**: `'LinearLR'`는 학습 초기에 학습률을 선형적으로 증가시키는 스케줄러입니다.
  - **start_factor**: `1e-6`은 초기 학습률에 곱해지는 시작 인자입니다.
  - **begin**과 **end**: 학습률 증가가 시작되고 끝나는 반복(iteration) 횟수입니다. 여기서는 0에서 1500까지입니다.
- **CosineAnnealingLR**:
  - **type**: `'CosineAnnealingLR'`는 주어진 최대 반복 횟수(`T_max`) 동안 학습률을 코사인 함수의 형태로 감소시키는 스케줄러입니다.
  - **eta_min**: 최소 학습률로, 여기서는 `0.0`입니다.
  - **T_max**와 **begin**, **end**: 이 스케줄러의 주기와 시작 및 종료 시점을 나타냅니다.



### 학습 일정 설정 (train_cfg, val_cfg, test_훅
### default훅

```yaml
# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=20000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

```
- **train_cfg**:
- **type**: 'IterBasedTrainLoop'은 반복 기반 학습 루프를 의미합니다.
- **max_iters**: 80000은 총 학습 반복 횟수입니다.
- **val_interval**: 20000은 검증을 수행할 반복 횟수 간격입니다.
- **val_cfg**와 **test_cfg**: 각각 검증 및 테스트 루프 설정을 나타냅니다. 여기서는 특별한 추가 설정 없이 기본 구조를 따릅니다.
- **timer**: `'IterTimerHook'`은 각 학습 반복의 시간을 측정합니다.
- **logger**: `'LoggerHook'`은 지정된 간격(`interval=50`)마다 학습 로그를 기록합니다.
- **param_scheduler**: `'ParamSchedulerHook'`은 위에서 정의된 학습률 스케줄러를 관
리합니다.
- **checkpoint**: `'CheckpointHook'`은 지정된 반복 간격(`interval=5000`)마다 모델의 체크포인트를 저장합니다.
- **sampler_seed**: `'DistSamplerSeedHook'`은 분산 학습 시 샘플러의 시드를 관리합니다.
- **visualization**: `'SegVisualizationHook'`은 학습 중 시각화를 위한 설정입니다. `draw=True`는 시각화를 활성화하며, `interval=1`은 매 반복마다 시각화를 수행합니다.

이러한 설정은 모델의 학습과 검증, 테스트 과정을 체계적으로 관리하며 성능 최적화를 도모하기 위해 필요합니다. 각 설정은 특정 목적에 맞게 조정될 수 있으며, 모델의 효율적인 학습을 지원합니다.
