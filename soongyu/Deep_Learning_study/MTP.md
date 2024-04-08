# MTP
## MTP pipeline
<img width="622" alt="pipeline" src="https://github.com/sibas-lab/study/assets/100738731/5a590aba-fdda-417c-bee7-e9af554b0479">

## DataSet
프리트레이닝 데이터셋    
DOTA-2.0   
회전된 바운딩 박스 버전을 자른 다음, SAM에 의해 세분화 레이블을 생성   
-> SOTA-RBB 획득
> 원본 SAMRS는 DOTA-2.0 수평 바운딩 박스 버전 사용
> 해당 깃허브 참조

<details>
<summary>사용법 번역 정리</summary>
<div markdown="1">
  
## 사용법
## 환경
#### 다양한 RS 작업에 대한 멀티 태스크 프리트레이닝 및 미세 조정을 지원하기 위해 새 버전의 OpenMMLab 시리즈를 채택
| Package | Version | Package | Version | Package | Version |Package | Version |
| ----- | :-----: | ----- | :-----: | ----- | :-----: | ----- | :-----: |
| Python | 3.8.17 | timm | 0.9.5 | MMEngine | 0.8.4 | MMDetection | 3.1.0 |
| Pytorch | 1.10.0 | OpenCV | 4.8.0 | MMPretrain | 1.2.0| MMRotate | 1.0.0rc1
| Torchvision | 0.10.0 | MMCV | 2.0.0 | MMSegmentation |1.0.0 | Open-CD | 1.1.0 |
#### MMRotate 0.3.4에 대한 환경 구성
| Package | Version | Package | Version | Package | Version |
| ----- | :-----: | ----- | :-----: | ----- | :-----: |
| Python | 3.8.0 | timm | 0.9.2 | MMEngine | 0.10.3 | 
| Pytorch | 1.10.0 | OpenCV | 4.7.0 | MMDetection | 2.28.2 |
| Torchvision | 0.10.0 | MMCV-full | 1.6.1 | MMRotate | 0.3.4 |

FAIR1M-2.0 및 DOTA-V1.0의 다중 스케일 예측에 사용
## 사전 학습 데이터 세트 준비

1. SAMRS 데이터 세트에서 SOTA-RBB와 SIOR 및 FAST 세트를 다운로드
2. SAMRS 데이터 세트의 *.pkl을 COCO *.json로 변환
   
    ```
    python scripts/convert_pkl_json.py
    ```
    
## 멀티태스킹 사전 학습 수행
SLURM으로 MTP를 수행합니다. 다음은 ViT-L + RVSA 사전 학습의 예

```
srun -J mtp -p gpu --gres=dcu:4 --ntasks=32 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python main_pretrain.py \
    --backbone 'vit_l_rvsa' --tasks 'ss' 'is' 'rd' \
    --datasets 'sota' 'sior' 'fast' \
    --batch_size 3 --batch_size_val 3 --workers 8 \
    --save_path [folder path of saved model] \
    --distributed 'True' --end_iter 80000 \
    --image_size 448 --init_backbone 'mae' --port '16003' --batch_mode 'avg' --background 'True' --use_ckpt 'True' --interval 5000
```

훈련은 설정하여 복구 가능 --ft--resume

```
--ft 'True' --resume [path of saved multi-task pretrained model]
```

## 데이터 세트 미세 조정 준비

**Xview**: 다음 기능이 포함

* geojson을 yolo 형식의 레이블로 변환
* 학습 세트와 테스트 세트 나누기
* 클립 이미지 및 yolo 형식 레이블
* yolo 형식 레이블을 COCO 형식으로 변환 *.json
  
**DIOR**: MMDetection에 공급하기 위해 *.xml를 COCO *.json 형식으로 변환

```
python scripts/dior_h_2_coco.py
```

**FAIR1M**: DOTA 형식의 *.txt를 제출에 필요한 *.xml로 변환

```
python scripts/dota_submit_txt_to_fair1m_xml.py --txt_dir [path of *.txt]
```

**SpaceNetv1**: geojson에서 분할 마스크 추출

```
python scripts/process_spacenet.py
```


## 다양한 RS 작업에 대한 미세 조정

**회전 감지를 제외하고 SLURM에서 미세 조정을 수행**

### 씬 분류 (MMPretrain 사용)   
MAE + MTP 사전 훈련된 ViT-L + RVSA를 사용한 EuroSAT 교육 및 검증

```
srun -J mmpretrn -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/vit-rvsa-l-224-mae-mtp_eurosat.py \
--work-dir=/diwang/work_dir/multitask_pretrain/finetune/classification/eurosat/vit-rvsa-l-224-mae-mtp_eurosat \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True
```

### 수평 물체 감지(MMDetection 사용)   
MAE + MTP 사전 훈련된 ViT-L + RVSA의 백본 네트워크와 함께 Faster-RCNN을 사용한 DIOR 교육

```
srun -J mmdet -p gpu --gres=dcu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py \
--work-dir=/diwang/work_dir/multitask_pretrain/finetune/Horizontal_Detection/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior \
--launcher="slurm" 
```

이 후 dection 결과를 테스트하고 생성

```
srun -J mmdet -p gpu --gres=dcu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior.py \
/diwang/work_dir/multitask_pretrain/finetune/Horizontal_Detection/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior/epoch_12.pth \
--work-dir=/diwang/work_dir/multitask_pretrain/finetune/Horizontal_Detection/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior/predict \
--show-dir=/diwang/work_dir/multitask_pretrain/finetune/Horizontal_Detection/dior/faster_rcnn_rvsa_l_800_mae_mtp_dior/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

### 회전된 물체 감지(MMRotate 사용, SLURM 및 GPU 서버 모두에서 실행)

**1. SLURM에서 실행**   
**(MMRotate 1.0.0rc1 사용)** MAE + MTP 사전 훈련된 ViT-L + RVSA의 백본 네트워크와 함께 Oriented-RCNN을 사용한 DIOR-R 교육

```
srun -J mmrot -p gpu --gres=dcu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr \
--launcher="slurm"
```

**(MMRotate 1.0.0rc1 사용)** DIOR-R에서 검출 맵 평가 및 시각화 테스트

```
srun -J mmrot -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/oriented_rcnn_rvsa_l_800_mae_mtp_diorr.py \
/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr/epoch_12.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr/predict \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/diorr/oriented_rcnn_rvsa_l_800_mae_mtp_diorr/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

**(MMRotate 0.3.4 사용)** 데이터 세트가 온라인으로 평가되는 경우 결과를 제출하고 탐지 맵을 시각화하기 위해 FAIR1M-2.0에서 테스트하는 예

```
srun -J mmrot -p gpu --gres=dcu:4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/fair1m/oriented_rcnn_rvsa_l_800_mae_mtp_fair1m20.py \
/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/fair1mv2/oriented_rcnn_rvsa_l_800_mae_mtp_fair1m20/epoch_12.pth --format-only \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/fair1mv2/oriented_rcnn_rvsa_l_800_mae_mtp_fair1m20/predict/show \
--eval-options submission_dir=/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/fair1mv2/oriented_rcnn_rvsa_l_800_mae_mtp_fair1m20/predict/submit \
--launcher="slurm"
```

**2. GPU 서버에서 실행**    
**(MMRotate 1.0.0rc1 사용)** MAE + MTP 사전 훈련된 ViT-L + RVSA의 백본 네트워크와 함께 Oriented-RCNN을 사용하여 DOTA-2.0에 대한 교육

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=40002 --master_addr=1.2.3.4 \
tools/train.py configs/mtp/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20.py \
--work-dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20
```

**(MMRotate 1.0.0rc1 사용)** 온라인 평가 결과를 제출하고 탐지 맵을 시각화하기 위한 DOTA-2.0에 대한 단일 스케일 테스트

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mtp/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20.py \
/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20/epoch_40.pth \
--work-dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20/test \
--show-dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav2/oriented_rcnn_rvsa_l_1024_mae_mtp_dota20/test/vis
```

**(MMRotate 0.3.4 사용)** 온라인 평가 결과를 제출하고 탐지 맵을 시각화하기 위한 DOTA-V1.0의 다중 스케일 테스트

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mtp/dotav1/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10.py \
/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav1/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10/epoch_12.pth --format-only \
--show-dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav1/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10/predict/show \
--eval-options submission_dir=/data/diwang22/work_dir/multitask_pretrain/finetune/Rotated_Detection/dotav1/oriented_rcnn_rvsa_l_1024_mae_mtp_dota10/predict/submit
```

### 의미 체계 구분(MMSegmentation 사용)

MAE + MTP 사전 훈련된 ViT-L + RVSA의 백본 네트워크와 함께 UperNet을 사용하여 SpaceNetv1에서 훈련

```
srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1 \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True
```

정확도 평가 및 예측 맵 생성을 위해 SpaceNetv1에서 테스트

```
srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1.py \
/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/iter_80000.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/predict \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

**온라인 평가**: 온라인 평가 결과 제출 및 예측 맵 생성을 위한 LoveDA 테스트

```
srun -J mmseg -p gpu --gres=dcu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/loveda/rvsa-l-upernet-512-mae-mtp-loveda.py \
/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-l-upernet-512-mae-mtp-loveda/iter_80000.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-l-upernet-512-mae-mtp-loveda/predict \
--out=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-l-upernet-512-mae-mtp-loveda/predict/submit \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-l-upernet-512-mae-mtp-loveda/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

*참고: 추론 후 LoveDA의 예측은 평가 사이트의 요구 사항을 충족하기 위해 수동으로 1을 줄여야 함*

```
python scripts/change_loveda_label.py
```

### 변경 내용 검색(Open-CD 사용)
MAE + MTP 사전 훈련된 ViT-L + RVSA의 백본 네트워크와 함께 UperNet을 사용하여 WHU에서 훈련

```
srun -J opencd -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/whu/rvsa-l-unet-256-mae-mtp_whu.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True
```

정확도 평가 테스트 및 예측 맵 생성

```
srun -J opencd -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/whu/rvsa-l-unet-256-mae-mtp_whu.py \
/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu/epoch_200.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu/predict \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

### 디코더 파라미터 재사용
1. MTP 저장된 가중치의 키를 변경

    ```
    python scripts/change_ckpt.py
    ```

2. 이 후 수정된 웨이트로 훈련
   
    ```
    srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
    python -u tools/train.py configs/mtp/spacenetv1/rvsa-b-upernet-384-mae-mtp-spacenetv1.py \
    --work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-b-upernet-384-mae-mtp-spacenetv1_reuse_decoder \
    --launcher="slurm" \
    --cfg-options 'find_unused_parameters'=True load_from=[path of the revised weights]
    ```
    
나머지 단계는 일반 테스트와 동일
</div>
</details>

<details>
<summary>주요 함수</summary>
<div markdown="1">
  
  <details>
<summary>__init__.py</summary>
<div markdown="1">
  
## Multi-Task_Pretrain/instance_segmentation/__init__.py

```
from .two_stage import *
from .mask_rcnn import *
from .rpn_head import *
from .roi_head import *
from .bbox_head import *
from .mask_head import *
```
- `two_stage`: 두 단계 객체 감지 프레임워크의 주요 구현을 포함
- `mask_rcnn`: Faster R-CNN의 확장인 Mask R-CNN 모델을 구현, 이 모델은 세분화 마스크를 예측하는 브랜치를 추가
- `rpn_head`: 영역 제안 네트워크 (RPN) 헤드의 구현을 포함, 이 헤드는 영역 제안을 생성하는 역할
- `roi_head`: 관심 영역 (ROI) 헤드를 구현, 이 헤드는 제안된 영역에서 바운딩 박스 회귀 및 분류
- `bbox_head`: 바운딩 박스 헤드의 구현을 포함, 이 헤드는 바운딩 박스 좌표와 객체 여부 점수
- `mask_head`: 마스크 헤드를 구현, 이 헤드는 모델이 감지한 객체에 대한 세분화 마스크
> 이름과 의미에서 이해한 각 부분의 역할 세부적으로 학습 예정

</div>
</details>

  <details>
<summary>two_stage.py</summary>
<div markdown="1">
  
## Multi-Task_Pretrain/instance_segmentation/two_stage.py

```
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors import BaseDetector


@MODELS.register_module()
class MTP_IS_TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType = None,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        #self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
```

객체 감지 모델을 구현하는데 사용되는 클래스 *MTP_IS_TwoStageDetector* 정의   
*BaseDetector* 클래스를 상속 받음   
두 단계 객체 감지 모델에서 사용되는 구성 요소들을 포함   
- __init__: 두 단계 객체 감지 모델의 주요 구성 요소인 백본(backbone), 넥(neck), RPN 헤드(rpn_head), ROI 헤드(roi_head) 등을 초기화
- _load_from_state_dict: 가중치를 로드할 때 bbox_head 키를 rpn_head 키로 교체
- extract_feat: 입력 이미지로부터 특징을 추출
- _forward: 네트워크의 순전파 과정을 정의 주로 백본, 넥, RPN 헤드, ROI 헤드의 순전파를 포함하며, 후처리는 미포함
- loss: 입력 이미지와 데이터 샘플의 배치로부터 손실을 계산
- predict: 입력 이미지와 데이터 샘플의 배치로부터 결과 예측, 후처리 수행

</div>
</details>

</div>
</details>
