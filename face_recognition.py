from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

def collate_fn(x):
    return x[0]


workers = 0 if os.name == 'nt' else 4
data_dir = './data/test_images'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# Define MTCNN module
# MTCNN은 신경망 및 기타 코드의 모음이므로 내부적으로 필요할 때
# 오브젝트를 복사할 수 있도록 하려면 다음과 같은 방식으로 전달해야 합니다.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)


# Define Inception Resnet V1 module
# 사전 학습된 분류기에 대해 classify=True를 설정합니다.
# 이 모델을 사용하여 임베딩/CNN 특징을 출력하겠습니다.
# 추론의 경우 모델을 평가 모드로 설정하는 것이 중요합니다.
# 자세한 내용은 help(InceptionResnetV1)을 참조하세요.
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# Define a dataset and data loader
# 라벨 인덱스를 나중에 ID 이름에 쉽게 레코딩할 수 있도록 데이터 세트에 idx_to_class 속성을 추가합니다.
dataset = datasets.ImageFolder(data_path)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


# Perfom MTCNN facial detection
# DataLoader 객체를 반복하여 얼굴과 각 얼굴에 대한 관련 감지 확률을 감지합니다.
# 얼굴이 감지된 경우 MTCNN 포워드 메서드는 감지된 얼굴에 맞게 잘린 이미지를 반환합니다.
# 기본적으로 감지된 얼굴은 하나만 반환되며, 감지된 모든 얼굴을 반환하려면
# 위의 MTCNN 객체를 생성할 때 keep_all=True로 설정합니다.
# 잘린 얼굴 이미지가 아닌 경계 상자를 얻으려면 대신 하위 수준 mtcnn.detect() 함수를 호출하면 됩니다.
# 자세한 내용은 help(mtcnn.detect)를 참조하세요.
aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])


# Calculate image embeddings
# MTCNN은 모두 동일한 크기의 얼굴 이미지를 반환하므로 리셋 인식 모듈로 쉽게 일괄 처리할 수 있습니다.
# 여기서는 이미지가 몇 개 밖에 없으므로 단일 배치를 빌드하고 추론을 수행합니다.
# 실제 데이터 세트의 경우, 특히 GPU에서 처리하는 경우에는 코드를 수정하여
# Resnet에 전달되는 배치 크기를 제어해야 합니다.
# 반복 테스트의 경우, 잘린 얼굴이나 경계 상자의 계산을 한 번만 수행하고
# 감지된 얼굴은 나중에 사용할 수 있도록 저장할 수 있으므로 얼굴 감지(MTCNN 사용)와
# 임베딩 또는 분류(InceptionResnetV1 사용)를 분리하는 것이 가장 좋습니다.
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()


# Print distance matrix for classes
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))