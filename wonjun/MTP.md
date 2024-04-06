 MTP : 다중 작업 사전 학습(MTP) 패러다임

# 첫번째 폴더 : Figs-pipeline.png
![image](https://github.com/hnk1203/study/assets/82886506/51e5971e-5ef4-4b0c-ab6b-7ba2051e09f6)


-red : 사전 훈련된 모델은 장면분류, 수평/회전 객체감지, 의미론적 분할, 변경감지 등 다양한 RS다운스트림 작업에 대해 미세 조정이 가능한거같음
       근데 우리는 여기서 의미론적 분할 + a 로 가야된다고 생각은 함
-green : 공유 인코더 및 작업별 디코더 아키텍처를 사용하여 다중 작업 지도 사전 학습 수행가능
         여기도 마찬가지로 쓰임새있을거 같은 부분만 별표로 강조해봤음


============================뭔지 잘은 모르겠는데 기본 훈련 코드? 같음===============================

의미론적 분할(MMSegmentation 사용)

MAE + MTP 사전 훈련된 ViT-L + RVSA의 백본 네트워크와 함께 UperNet을 사용하여 SpaceNetv1에 대한 훈련:

srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1 \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True


정확도 평가 및 예측 지도 생성을 위해 SpaceNetv1에서 테스트:

srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/test.py configs/mtp/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1.py \
/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/iter_80000.pth \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/predict \
--show-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-l-upernet-384-mae-mtp-spacenetv1/predict/show \
--launcher="slurm" --cfg-options val_cfg=None val_dataloader=None val_evaluator=None

디코더 매개변수 재사용
미세 조정에서 분할 디코더를 재사용하는 예를 들어보겠습니다.

MTP 저장 가중치의 키를 변경합니다.

python scripts/change_ckpt.py
그런 다음 수정된 가중치로 훈련합니다.

srun -J mmseg -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/spacenetv1/rvsa-b-upernet-384-mae-mtp-spacenetv1.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/spacenetv1/rvsa-b-upernet-384-mae-mtp-spacenetv1_reuse_decoder \
--launcher="slurm" \
--cfg-options 'find_unused_parameters'=True load_from=[path of the revised weights]

===============================================================================


#두번째 폴더 : Multi_Task_Pretrain : 사전훈련하는 곳
-1. instance_segmentation
-- __init__.py : 

         two_stage.py: 두 단계로 객체를 감지하는 알고리즘을 구현한 모듈일 수 있습니다.
         여기서 두 단계는 일반적으로 후보 영역(proposal) 생성과 객체 분류 및 바운딩 박스(bounding box) 조정을 의미합니다.
         
         mask_rcnn.py: Mask R-CNN 알고리즘을 구현한 모듈일 수 있습니다. 이는 객체의 분류, 
         위치 식별 뿐만 아니라 객체의 정확한 픽셀 단위 마스크를 생성하는 데 사용됩니다.
         
         rpn_head.py: Region Proposal Network(RPN)의 핵심 컴포넌트를 구현한 모듈일 수 있습니다. 이는 객체 후보 영역을 식별하는 데 사용됩니다.
         
         roi_head.py: 관심 영역(Region of Interest, ROI) 처리를 위한 핵심 컴포넌트를 구현한 모듈일 수 있습니다.
         
         bbox_head.py: 바운딩 박스 회귀(bounding box regression)와 관련된 기능을 구현한 모듈일 수 있습니다.
         
         mask_head.py: 객체의 마스크 생성과 관련된 기능을 구현한 모듈일 수 있습니다.


-2. semantic_segmentation
-- encoder_decoder.py : 이 코드는 OpenMMLab에서 제공하는 MTP_SS_UperNet이라는 모델 클래스를 정의합니다. 이 모델은 이미지 분할을 수행하는 세그멘테이션 모델입니다.
         #OpenMMLab : 딥러닝 관련 lib 제공, 컴퓨터 비전 관련 업체
         주요인자 :
              decode_head: 디코더 헤드의 설정을 나타내는 딕셔너리입니다. 이 딕셔너리는 UPerHead 라는 디코더 헤드를 사용하며, 해당 헤드의 구성을 지정합니다.
              neck: 넥(추가 모듈)의 설정을 나타내는 옵셔널한 매개변수입니다.
              auxiliary_head: 보조 헤드의 설정을 나타내는 옵셔널한 매개변수입니다.
              train_cfg: 훈련에 대한 설정을 나타내는 옵셔널한 매개변수입니다.
              test_cfg: 테스트에 대한 설정을 나타내는 옵셔널한 매개변수입니다.
              data_preprocessor: 데이터 전처리기의 설정을 나타내는 옵셔널한 매개변수입니다.
              pretrained: 사전 학습된 모델의 경로를 나타내는 옵셔널한 매개변수입니다.
              init_cfg: 모델의 가중치 초기화 설정을 나타내는 옵셔널한 매개변수입니다.

         
######### =================

loss, predict, _forward 메서드는 각각 모델의 훈련, 예측, 순방향 전달을 담당합니다. 또한 슬라이딩 윈도우 방식의 추론과 전체 이미지를 사용한 추론을 지원하며, 데이터 증강을 통한 테스트도 지원합니다.
from typing import List, Optional: 파이썬의 타입 힌팅에 사용되는 모듈들을 불러옵니다. 리스트와 옵셔널한 값의 타입을 지정할 때 사용됩니다.

import torch.nn as nn, import torch.nn.functional as F, from torch import Tensor: 파이토치의 신경망 모듈 및 텐서를 불러옵니다.

from mmseg.registry import MODELS: mmseg 패키지의 모델 레지스트리에서 MODELS를 불러옵니다.

from mmseg.utils import ...: mmseg 패키지에서 여러 유틸리티 함수 및 클래스를 불러옵니다.

@MODELS.register_module(): MTP_SS_UperNet 클래스를 MODELS 레지스트리에 등록합니다.

class MTP_SS_UperNet(BaseSegmentor):: MTP_SS_UperNet 클래스를 정의합니다. 이 클래스는 BaseSegmentor 클래스를 상속합니다.

def __init__(self, ...):: 클래스의 초기화 함수를 정의합니다. 이 함수는 클래스가 생성될 때 호출되며, 모델의 구조와 설정을 초기화합니다.

def _init_decode_head(self, decode_head: ConfigType) -> None:, def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:, def extract_feat(self, inputs: Tensor) -> List[Tensor]:, def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor: 등의 메서드들은 클래스의 내부 동작을 정의합니다. 이들은 주로 특징 추출, 디코더 헤드 초기화, 예측 등의 작업을 수행합니다.

def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:, def predict(self, inputs: Tensor, data_samples: OptSampleList = None) -> SampleList:, def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tensor: 등의 메서드들은 모델의 훈련, 예측, 순방향 전달을 담당합니다.

def slide_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:, def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:, def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor: 등의 메서드들은 추론을 수행합니다. 슬라이딩 윈도우 방식의 추론과 전체 이미지를 사용한 추론이 가능합니다.

def aug_test(self, inputs, batch_img_metas, rescale=True):: 이 메서드는 데이터 증강을 적용한 테스트를 수행합니다.
