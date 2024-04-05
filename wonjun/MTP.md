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


#두번째 폴더 : Multi_Task_Pretrain



