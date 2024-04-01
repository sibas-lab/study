"""
Title: 이미지 분할을 위한 조합 가능한 완전 합성곱 네트워크
Author: Suvaditya Mukherjee
Date created: 2023/06/16
Last modified: 2023/12/25
Description: 완전 합성곱 네트워크를 사용한 이미지 분할.
Accelerator: GPU
"""

"""
## Introduction

이 예제는 Oxford-IIIT Pets 데이터셋에서 이미지 분할을 위한 완전 합성곱 네트워크를 구현하는 단계를 안내합니다.
모델은 2014년 Long 등이 제안한 논문에 기초합니다.
이미지 분할은 컴퓨터 비전에서 가장 일반적이고 기초적인 작업 중 하나로,
이미지 분류 문제를 확장하여 이미지의 각 픽셀에 대해 클래스를 분류합니다.
이 예제에서는 해당 완전 합성곱 분할 아키텍처를 조립하여 이미지 분할을 수행할 수 있습니다.


네트워크는 VGG의 풀링 레이어 출력을 확장하여 업샘플링을 수행하고 최종 결과를 얻습니다.
 VGG19의 3번째, 4번째, 5번째 맥스-풀링 레이어에서 나오는 중간 출력들이 추출되어 다양한 레벨과 배율로 업샘플링되어,
  출력의 형태는 동일하되 각 위치에 픽셀 강도 값 대신 각 픽셀의 클래스가 표시된 최종 출력을 얻습니다.
   네트워크의 다양한 버전을 위해 다른 중간 풀 레이어들이 추출되어 처리됩니다.
    FCN 아키텍처는 품질이 다른 3가지 버전을 가지고 있습니다.

- FCN-32S
- FCN-16S
- FCN-8S
모델의 모든 버전은 사용된 주요 백본의 연속적인 중간 풀 레이어들을 반복적으로 처리함으로써 그들의 출력을 도출합니다. 아래 그림에서 더 나은 이해를 얻을 수 있습니다.

FCN 아키텍처
다이어그램 1: 결합된 아키텍처 버전 (출처: 논문)
이미지 분할에 대해 더 잘 이해하거나 더 많은 사전 훈련된 모델을 찾고 싶다면,
 Hugging Face 이미지 분할 모델 페이지를 방문하거나,
  PyImageSearch 블로그의 의미론적 분할에 관한 글을 참고하세요.
"""

"""
## Setup Imports
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import keras
from keras import ops
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

"""
## Set configurations for notebook variables

실험에 필요한 매개변수를 설정합니다.
선택된 데이터셋에는 이미지당 4개의 클래스가 있습니다.
이 셀에서 하이퍼파라미터를 설정합니다.

시스템이 지원하는 경우 혼합 정밀도 옵션을 사용하여 부하를 줄일 수 있습니다.
이는 계산에서 `32비트 float` 값 대신 대부분의 텐서에 `16비트 float` 값을 사용하게 합니다.
이는 계산 중에 `16비트 float` 텐서를 사용하여 속도를 높이고 정밀도를 손해 보는 대신,
값을 원래의 기본 `32비트 float` 형식으로 저장합니다.
"""

NUM_CLASSES = 4
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
MIXED_PRECISION = True
SHUFFLE = True

# Mixed-precision setting
if MIXED_PRECISION:
    policy = keras.mixed_precision.Policy("mixed_float16")
    keras.mixed_precision.set_global_policy(policy)

"""
## Load dataset

Oxford-IIIT Pets 데이터셋을 사용합니다.
총 7,349개의 샘플과 그들의 분할 마스크를 포함합니다.
학습 및 검증 데이터셋은 각각 3,128개와 552개의 샘플을 가지고 있습니다.
이외에도, 우리의 테스트 분할은 총 3,669개의 샘플을 가집니다.

`batch_size` 매개변수를 설정하여 샘플을 함께 배치하고, `shuffle` 매개변수를 사용하여 샘플을 섞습니다.
"""
# valid_ds
(train_ds, test_ds) = tfds.load(
    "oxford_iiit_pet",
    split=["train", "test"],
    batch_size=BATCH_SIZE,
    shuffle_files=SHUFFLE,
)

"""
## Unpack and preprocess dataset

우리는 학습, 검증, 테스트 데이터셋에 대해 간단한 리사이징 작업을 포함하는 함수를 정의합니다.
마스크에 대해서도 동일한 과정을 수행하여 모양과 크기가 맞도록 합니다.
"""


# Image and Mask Pre-processing
def unpack_resize_data(section):
    image = section["image"]
    segmentation_mask = section["segmentation_mask"]

    resize_layer = keras.layers.Resizing(INPUT_HEIGHT, INPUT_WIDTH)

    image = resize_layer(image)
    segmentation_mask = resize_layer(segmentation_mask)

    return image, segmentation_mask


train_ds = train_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
# valid_ds = valid_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
"""
## Visualize one random sample from the pre-processed dataset

우리는 데이터셋의 테스트 분할에서 무작위 샘플을 시각화하여 효과적인 마스크 영역을 보여줍니다.
데이터셋에 대한 전처리도 수행되었으므로 이미지와 마스크의 크기가 동일합니다.
"""

# Select random image and mask. Cast to NumPy array
# for Matplotlib visualization.

images, masks = next(iter(test_ds))
random_idx = keras.random.uniform([], minval=0, maxval=BATCH_SIZE, seed=10)

test_image = images[int(random_idx)].numpy().astype("float")
test_mask = masks[int(random_idx)].numpy().astype("float")

# Overlay segmentation mask on top of image.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax[0].set_title("Image")
ax[0].imshow(test_image / 255.0)

ax[1].set_title("Image with segmentation mask overlay")
ax[1].imshow(test_image / 255.0)
ax[1].imshow(
    test_mask,
    cmap="inferno",
    alpha=0.6,
)
plt.show()

"""
## Perform VGG-specific pre-processing


keras.applications.
VGG19는 Image-net 스타일의 표준 편차 정규화 체계를 능동적으로 수행하는 preprocess_input 함수의 사용을 요구합니다.
"""


def preprocess_data(image, segmentation_mask):
    image = keras.applications.vgg19.preprocess_input(image)

    return image, segmentation_mask


train_ds = (
    train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=1024)
    .prefetch(buffer_size=1024)
)
# valid_ds = (
#     valid_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
#     .shuffle(buffer_size=1024)
#     .prefetch(buffer_size=1024)
# )
test_ds = (
    test_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=1024)
    .prefetch(buffer_size=1024)
)
"""
## Model Definition

완전 합성곱 네트워크는 `keras.layers.Conv2D` 층, `keras.layers.Dense` 층, `keras.layers.Dropout` 층으로만 구성된 간단한 아키텍처를 자랑합니다.

각 이미지 크기와 동일한 Softmax 합성곱 층을 가지고 픽셀별 예측을 수행하여 직접 비교를 수행할 수 있습니다.

| ![FCN Architecture](https://i.imgur.com/PerTKjf.png) |
| :--: |
| **Diagram 2**: Generic FCN Forward Pass (Source: Paper)|
네트워크에서 정확도와 평균 교차 유니온(Mean-Intersection-over-Union)과 같은 여러 중요한 지표를 찾을 수 있습니다.

"""

"""
### Backbone (VGG-19)


우리는 VGG-19 네트워크를 백본으로 사용합니다.
 이는 논문에서 이 네트워크에 대한 가장 효과적인 백본 중 하나로 제안되었습니다.
  keras.models.Model을 사용하여 네트워크에서 다양한 출력을 추출합니다.
   이어서, 우리는 다이어그램 1을 완벽하게 모방하는 네트워크를 만들기 위해 상단에 층을 추가합니다.
    백본의 keras.layers.Dense 층은 여기에 있는 원본 Caffe 코드에 기반하여 keras.layers.Conv2D 층으로 변환될 것입니다.
     모든 3개의 네트워크는 동일한 백본 가중치를 공유하지만, 그들의 확장에 따라 다른 결과를 가질 것입니다.
      학습 시간 요구 사항을 개선하기 위해 백본을 비학습 가능하게 만듭니다.
       논문에서도 네트워크를 학습 가능하게 만드는 것이 주요 이점을 제공하지 않는다고 언급되었습니다.
"""

input_layer = keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))

# VGG Model backbone with pre-trained ImageNet weights.
vgg_model = keras.applications.vgg19.VGG19(include_top=True, weights="imagenet")

# Extracting different outputs from same model
fcn_backbone = keras.models.Model(
    inputs=vgg_model.layers[1].input,
    outputs=[
        vgg_model.get_layer(block_name).output
        for block_name in ["block3_pool", "block4_pool", "block5_pool"]
    ],
)

# Setting backbone to be non-trainable
fcn_backbone.trainable = False

x = fcn_backbone(input_layer)

# Converting Dense layers to Conv2D layers
units = [4096, 4096]
dense_convs = []

for filter_idx in range(len(units)):
    dense_conv = keras.layers.Conv2D(
        filters=units[filter_idx],
        kernel_size=(7, 7) if filter_idx == 0 else (1, 1),
        strides=(1, 1),
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_initializer=keras.initializers.Constant(1.0),
    )
    dense_convs.append(dense_conv)
    dropout_layer = keras.layers.Dropout(0.5)
    dense_convs.append(dropout_layer)

dense_convs = keras.Sequential(dense_convs)
dense_convs.trainable = False

x[-1] = dense_convs(x[-1])

pool3_output, pool4_output, pool5_output = x

"""
### FCN-32S


마지막 출력을 확장하고, 1x1 합성곱을 수행한 다음,
 32배수로 2D 이중선형 업샘플링을 수행하여 입력과 동일한 크기의 이미지를 얻습니다.
  우리는 keras.layers.Conv2DTranspose보다 keras.layers.UpSampling2D 층을 사용하는데,
   이는 합성곱 연산보다 결정론적인 수학 연산으로서 성능상의 이점을 제공하기 때문입니다.
    또한 논문에서는 업샘플링 파라미터를 학습 가능하게 만드는 것이 이점을 제공하지 않는다고 언급되어 있습니다.
     논문의 원래 실험에서도 업샘플링이 사용되었습니다.
"""

# 1x1 convolution to set channels = number of classes
pool5 = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    padding="same",
    strides=(1, 1),
    activation="relu",
)

# Get Softmax outputs for all classes
fcn32s_conv_layer = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    activation="softmax",
    padding="same",
    strides=(1, 1),
)

# Up-sample to original image size
fcn32s_upsampling = keras.layers.UpSampling2D(
    size=(32, 32),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)

final_fcn32s_pool = pool5(pool5_output)
final_fcn32s_output = fcn32s_conv_layer(final_fcn32s_pool)
final_fcn32s_output = fcn32s_upsampling(final_fcn32s_output)

fcn32s_model = keras.Model(inputs=input_layer, outputs=final_fcn32s_output)

"""
### FCN-16S

FCN-32S에서 나온 풀링 출력은 확장되어 우리 백본의 4단계 풀링 출력과 더해집니다.
 이후에, 우리는 16배로 업샘플링하여 입력과 동일한 크기의 이미지를 얻습니다.
"""

# 1x1 convolution to set channels = number of classes
# Followed from the original Caffe implementation
pool4 = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    padding="same",
    strides=(1, 1),
    activation="linear",
    kernel_initializer=keras.initializers.Zeros(),
)(pool4_output)

# Intermediate up-sample
pool5 = keras.layers.UpSampling2D(
    size=(2, 2),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)(final_fcn32s_pool)

# Get Softmax outputs for all classes
fcn16s_conv_layer = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    activation="softmax",
    padding="same",
    strides=(1, 1),
)

# Up-sample to original image size
fcn16s_upsample_layer = keras.layers.UpSampling2D(
    size=(16, 16),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)

# Add intermediate outputs
final_fcn16s_pool = keras.layers.Add()([pool4, pool5])
final_fcn16s_output = fcn16s_conv_layer(final_fcn16s_pool)
final_fcn16s_output = fcn16s_upsample_layer(final_fcn16s_output)

fcn16s_model = keras.models.Model(inputs=input_layer, outputs=final_fcn16s_output)

"""
### FCN-8S


FCN-16S에서 나온 풀링 출력은 한 번 더 확장되고,
 우리 백본의 3단계 풀링 출력과 추가됩니다.
  이 결과는 우리 입력과 같은 크기의 이미지를 얻기 위해 8배로 업샘플링됩니다.
"""

# 1x1 convolution to set channels = number of classes
# Followed from the original Caffe implementation
pool3 = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    padding="same",
    strides=(1, 1),
    activation="linear",
    kernel_initializer=keras.initializers.Zeros(),
)(pool3_output)

# Intermediate up-sample
intermediate_pool_output = keras.layers.UpSampling2D(
    size=(2, 2),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)(final_fcn16s_pool)

# Get Softmax outputs for all classes
fcn8s_conv_layer = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    activation="softmax",
    padding="same",
    strides=(1, 1),
)

# Up-sample to original image size
fcn8s_upsample_layer = keras.layers.UpSampling2D(
    size=(8, 8),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)

# Add intermediate outputs
final_fcn8s_pool = keras.layers.Add()([pool3, intermediate_pool_output])
final_fcn8s_output = fcn8s_conv_layer(final_fcn8s_pool)
final_fcn8s_output = fcn8s_upsample_layer(final_fcn8s_output)

fcn8s_model = keras.models.Model(inputs=input_layer, outputs=final_fcn8s_output)

"""
### Load weights into backbone


논문에서 밝힌 바와 실험을 통해 알 수 있듯이,
 백본에서 마지막 2개의 완전 연결(Dense) 층의 가중치를 추출하고,
  이 가중치를 이전에 keras.layers.Dense 층에서 keras.layers.Conv2D로 변환한 것과 맞도록 재구성하여 설정하면,
   훨씬 더 좋은 결과와 mIOU 성능의 상당한 향상을 가져옵니다.
"""

# VGG's last 2 layers
weights1 = vgg_model.get_layer("fc1").get_weights()[0]
weights2 = vgg_model.get_layer("fc2").get_weights()[0]

weights1 = weights1.reshape(7, 7, 512, 4096)
weights2 = weights2.reshape(1, 1, 4096, 4096)

dense_convs.layers[0].set_weights([weights1])
dense_convs.layers[2].set_weights([weights2])

"""
## Training

원본 논문에서는 모멘텀을 가진 SGD가 최적의 옵티마이저로 언급되어 있습니다.
그러나 실험을 통해 AdamW가 mIOU와 픽셀별 정확도 측면에서 더 나은 결과를 제공함을 알 수 있었습니다.
[AdamW](https://keras.io/api/optimizers/adamw/)
"""

"""
### FCN-32S
"""

fcn32s_optimizer = keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

fcn32s_loss = keras.losses.SparseCategoricalCrossentropy()

# Maintain mIOU and Pixel-wise Accuracy as metrics
fcn32s_model.compile(
    optimizer=fcn32s_optimizer,
    loss=fcn32s_loss,
    metrics=[
        keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
        keras.metrics.SparseCategoricalAccuracy(),
    ],
)

fcn32s_history = fcn32s_model.fit(train_ds, epochs=EPOCHS)
# , validation_data=valid_ds

"""
### FCN-16S
"""

fcn16s_optimizer = keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

fcn16s_loss = keras.losses.SparseCategoricalCrossentropy()

# Maintain mIOU and Pixel-wise Accuracy as metrics
fcn16s_model.compile(
    optimizer=fcn16s_optimizer,
    loss=fcn16s_loss,
    metrics=[
        keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
        keras.metrics.SparseCategoricalAccuracy(),
    ],
)

fcn16s_history = fcn16s_model.fit(train_ds, epochs=EPOCHS)
# , validation_data=valid_ds

"""
### FCN-8S
"""

fcn8s_optimizer = keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

fcn8s_loss = keras.losses.SparseCategoricalCrossentropy()

# Maintain mIOU and Pixel-wise Accuracy as metrics
fcn8s_model.compile(
    optimizer=fcn8s_optimizer,
    loss=fcn8s_loss,
    metrics=[
        keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False),
        keras.metrics.SparseCategoricalAccuracy(),
    ],
)

fcn8s_history = fcn8s_model.fit(train_ds, epochs=EPOCHS)
# , validation_data=valid_ds
"""
## Visualizations
"""

"""
### Plotting metrics for training run

모든 3개 버전의 모델 간 비교 연구를 수행하여 학습 및 검증 메트릭의 정확도, 손실, 평균 IoU를 추적합니다.
"""

total_plots = len(fcn32s_history.history)
cols = total_plots // 2

rows = total_plots // cols

if total_plots % cols != 0:
    rows += 1

# Set all history dictionary objects
fcn32s_dict = fcn32s_history.history
fcn16s_dict = fcn16s_history.history
fcn8s_dict = fcn8s_history.history

pos = range(1, total_plots + 1)
plt.figure(figsize=(15, 10))

for i, ((key_32s, value_32s), (key_16s, value_16s), (key_8s, value_8s)) in enumerate(
    zip(fcn32s_dict.items(), fcn16s_dict.items(), fcn8s_dict.items())
):
    plt.subplot(rows, cols, pos[i])
    plt.plot(range(len(value_32s)), value_32s)
    plt.plot(range(len(value_16s)), value_16s)
    plt.plot(range(len(value_8s)), value_8s)
    plt.title(str(key_32s) + " (combined)")
    plt.legend(["FCN-32S", "FCN-16S", "FCN-8S"])

plt.show()

"""
### Visualizing predicted segmentation masks

결과를 더 잘 이해하고 보기 위해, 테스트 데이터셋에서 무작위 이미지를 선택하고 각 모델에서 생성된 마스크를 추론하여 봅니다.
참고: 더 나은 결과를 얻으려면 모델을 더 많은 에폭 동안 학습해야 합니다.
"""

images, masks = next(iter(test_ds))
random_idx = keras.random.uniform([], minval=0, maxval=BATCH_SIZE, seed=10)

# Get random test image and mask
test_image = images[int(random_idx)].numpy().astype("float")
test_mask = masks[int(random_idx)].numpy().astype("float")

pred_image = ops.expand_dims(test_image, axis=0)
pred_image = keras.applications.vgg19.preprocess_input(pred_image)

# Perform inference on FCN-32S
pred_mask_32s = fcn32s_model.predict(pred_image, verbose=0).astype("float")
pred_mask_32s = np.argmax(pred_mask_32s, axis=-1)
pred_mask_32s = pred_mask_32s[0, ...]

# Perform inference on FCN-16S
pred_mask_16s = fcn16s_model.predict(pred_image, verbose=0).astype("float")
pred_mask_16s = np.argmax(pred_mask_16s, axis=-1)
pred_mask_16s = pred_mask_16s[0, ...]

# Perform inference on FCN-8S
pred_mask_8s = fcn8s_model.predict(pred_image, verbose=0).astype("float")
pred_mask_8s = np.argmax(pred_mask_8s, axis=-1)
pred_mask_8s = pred_mask_8s[0, ...]

# Plot all results
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

fig.delaxes(ax[0, 2])

ax[0, 0].set_title("Image")
ax[0, 0].imshow(test_image / 255.0)

ax[0, 1].set_title("Image with ground truth overlay")
ax[0, 1].imshow(test_image / 255.0)
ax[0, 1].imshow(
    test_mask,
    cmap="inferno",
    alpha=0.6,
)

ax[1, 0].set_title("Image with FCN-32S mask overlay")
ax[1, 0].imshow(test_image / 255.0)
ax[1, 0].imshow(pred_mask_32s, cmap="inferno", alpha=0.6)

ax[1, 1].set_title("Image with FCN-16S mask overlay")
ax[1, 1].imshow(test_image / 255.0)
ax[1, 1].imshow(pred_mask_16s, cmap="inferno", alpha=0.6)

ax[1, 2].set_title("Image with FCN-8S mask overlay")
ax[1, 2].imshow(test_image / 255.0)
ax[1, 2].imshow(pred_mask_8s, cmap="inferno", alpha=0.6)

plt.show()

"""
## Conclusion

완전 합성곱 네트워크는 이미지 분할 작업에서 다양한 벤치마크에서 강력한 결과를 제공하는 매우 간단한 네트워크입니다.
[SegFormer](https://arxiv.org/abs/2105.15203)와 [DeTR](https://arxiv.org/abs/2005.12872)에서 사용된 
[Attention](https://arxiv.org/abs/1706.03762)과 같은 더 나은 메커니즘의 등장으로,
 이 모델은 알려지지 않은 데이터에 대한 이 작업의 기준점을 찾는 데 있어 빠른 방법을 제공합니다.
"""

"""
## Acknowledgements

이 예제의 초기 리뷰를 제공해 준 [Aritra Roy Gosthipaty](https://twitter.com/ariG23498),
[Ayush Thakur](https://twitter.com/ayushthakur0) 그리고 
[Ritwik Raha](https://twitter.com/ritwik_raha)에게 감사합니다.
또한 [Google Developer Experts](https://developers.google.com/community/experts) 프로그램에 감사드립니다.

"""
