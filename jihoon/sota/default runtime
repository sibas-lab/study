# 프로젝트 설정 가이드

이 문서는 멀티모달 세그멘테이션(Multi-modal Segmentation) 관련 프로젝트의 설정 파일 예시를 설명합니다.

## 기본 설정

```
default_scope: 'mmseg'
```
- `default_scope`는 모델의 기본 범위를 정의합니다. 이 경우 `'mmseg'`로 설정되어 있어, 멀티모달 세그멘테이션에 관련된 설정임을 나타냅니다.

## 환경 설정 (`env_cfg`)

```
env_cfg:
  cudnn_benchmark: True
  mp_cfg:
    mp_start_method: 'fork'
    opencv_num_threads: 0
  dist_cfg:
    backend: 'nccl'
```

- `cudnn_benchmark`: 모델의 학습 속도를 향상시키기 위해 설정됩니다.
- `mp_cfg`는 멀티프로세싱 설정을 정의합니다.
  - `mp_start_method: 'fork'`는 자식 프로세스를 시작하는 방법으로 'fork'를 사용하겠다는 것을 의미합니다.
  - `opencv_num_threads: 0`은 OpenCV의 멀티스레딩을 비활성화합니다.
- `dist_cfg`는 분산 학습 설정을 담고 있으며, `backend: 'nccl'`은 NVIDIA의 NCCL(NVIDIA Collective Communications Library)을 분산 학습의 통신 백엔드로 사용하겠다는 것을 나타냅니다.

## 시각화 설정

```
vis_backends:
  - type: 'LocalVisBackend'
visualizer:
  type: 'SegLocalVisualizer'
  vis_backends: vis_backends
  name: 'visualizer'
```

- `LocalVisBackend`는 로컬 시스템에서 시각화를 수행하는 백엔드 유형을 의미합니다.
- `SegLocalVisualizer`는 세그멘테이션 결과를 시각화하는 도구를 설정하며, `vis_backends`에서 정의한 백엔드를 사용합니다.

## 로깅 설정

```
log_processor:
  by_epoch: False
log_level: 'INFO'
```

- `by_epoch: False`는 로그가 에폭(epoch) 단위가 아니라 반복(iteration) 단위로 기록되도록 합니다.
- `log_level: 'INFO'`는 로그의 상세 수준을 'INFO'로 설정합니다.

## 학습 재개 설정

```
Load_from: None
resume: False
```

- `load_from: None`은 특정 체크포인트에서 모델을 로드하지 않고 처음부터 학습을 시작하겠다는 의미입니다.
- `resume: False`는 이전에 중단된 학습을 이어서 진행하지 않겠다는 설정입니다.. 
- 만약 True로 설정하면, load_from에 지정된 체크포인트에서 학습 상태(예: 현재 에폭, 최적의 메트릭 값 등)를 함께 로드하여 학습을 계속 이어갈 수 있습니다.
```

이 구성은 프로젝트 설정을 명확하게 설명하며, 개발자가 필요에 따라 쉽게 수정할 수 있도록 도와줍니다.
