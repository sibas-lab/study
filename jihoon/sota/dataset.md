이 설정은 SpaceNet V1 데이터셋에 기반한 위성 이미지 세그멘테이션 프로젝트를 위한 데이터 처리 및 모델 학습 구성을 제공합니다. 각 섹션별로 자세한 설명을 추가하겠습니다.

### 데이터셋 설정

```yaml
dataset_type: 'SpaceNetV1Dataset'
```
- **`dataset_type`**: 프로젝트에서 사용할 데이터셋의 유형을 정의합니다. 'SpaceNetV1Dataset'은 위성 이미지 분석을 위해 설계된 특정 데이터셋 유형을 나타내며, 이 데이터셋은 도시 지역의 건물, 도로 등을 세그멘테이션하기 위한 고해상도 이미지를 포함합니다.

```yaml
data_root: '/work/share/achk2o1zg1/diwang22/dataset/spacenet/data'
```
- **`data_root`**: 데이터셋 파일(이미지와 레이블)이 위치한 기본 경로입니다. 이 경로는 모든 데이터 접근 작업의 기준점으로 사용됩니다.

```yaml
crop_size: (384, 384)
```
- **`crop_size`**: 모델 학습에 사용될 이미지의 잘라낸 크기를 정의합니다. 이 크기는 모델이 처리할 수 있는 입력 크기에 맞춰 설정되며, 일관된 입력 크기를 제공하기 위해 사용됩니다.

### 데이터 파이프라인 설정

각 데이터 파이프라인(`train_pipeline`, `val_pipeline`, `test_pipeline`)은 데이터 로딩과 전처리 단계를 순차적으로 정의합니다.
이 단계들은 학습, 검증, 테스트 세트에 적용되며, 각각의 목적에 맞게 조정됩니다.

#### 학습 파이프라인 (`train_pipeline`)

```yaml
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(384, 384),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
```

- **`LoadImageFromFile`**: 이미지 파일을 디스크에서 로드합니다.
- **`LoadAnnotations`**: 해당 이미지의 레이블(세그멘테이션 마스크)을 로드합니다. `reduce_zero_label=False`는 레이블 값의 변환 없이 원본을 유지합니다.
- **`RandomResize`와 `RandomCrop`**: 이미지 크기를 무작위로 조절하고, 지정된 `crop_size`에 맞게 이미지를 잘라냅니다. 이는 데이터의 다양성을 증가시키고, 오버피팅을 방지하는 데 도움이 됩니다.
- **`RandomFlip`**: 이미지를 수평으로 무작위로 뒤집어, 데이터의 다양성을 더합니다.
- **`PhotoMetricDistortion`**: 이미지의 색상 관련 속성을 변형시키는 것으로, 조명 변화 등에 모델이 더 강건하게 대응할 수 있도록 합니다.
- **`PackSegInputs`**: 전처리된 이미지와 레이블을 모델이 처리할 수 있는 형태로 패키징합니다.

#### 검증 및 테스트 파이프라인 (`val_pipeline`, `test_pipeline`)

```yaml
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
```

검증 및 테스트 파이프라인은 학습 파이프라인보다 간소화되어 있으며, 주로 이미지 크기 조절과 필요한 레이블 로딩으로 구성됩니다. 이는 평가 시 일관성을 유지하고, 실제 성능을 정확히 측정하기 위함입니다.

### 데이터 로더 설정

`train_dataloader`, `val_dataloader`, `test_dataloader`는 각각 학습, 검증, 테스트 데이터셋을 모델에 공급하기 위한 구성을 정의합니다. 여기서 주
요 속성으로는 `batch_size`, `num_workers`, `persistent_workers`, `sampler`가 있습니다.

```yaml
train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/labels'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/labels'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/labels'),
        pipeline=test_pipeline))
```

- **`batch_size`**: 한 번에 처리될 이미지의 수입니다.
- **`num_workers`**: 데이터 로딩을 위해 병렬로 실행될 프로세스의 수입니다. 이는 로딩 속도를 향상시키는 데 도움이 됩니다.
- **`persistent_workers`**: `True`로 설정 시, 데이터 로더가 epoch 간에 작업자를 유지하여 로딩 시간을 단축합니다.
- **`sampler`**: 데이터 샘플링 전략을 정의합니다. 예를 들어, `InfiniteSampler`는 무한히 데이터를 샘플링하는 방식을, `DefaultSampler`는 기본적인 순차 샘플링을 의미합니다.

### 평가기 설정

```yaml
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=255)
test_evaluator = val_evaluator
```

- **`IoUMetric`**: 모델 성능 평가를 위해 IoU 지표를 사용합니다. `mIoU`는 클래스 간 평균 IoU를 계산하며, 세그멘테이션 작업에서 널리 사용되는 중요한 성능 지표입니다. `ignore_index=255`는 특정 레이블 값을 평가에서 제외합니다.

이 구성은 SpaceNet V1 데이터셋을 활용한 위성 이미지 세그멘테이션 프로젝트의 효율적인 데이터 관리와 모델 학습을 지원합니다. 데이터의 전처리부터 로딩, 모델 학습 및 성능 평가에 이르기까지 프로젝트의 모든 단계에 걸쳐 최적화된 설정을 제공합니다.
