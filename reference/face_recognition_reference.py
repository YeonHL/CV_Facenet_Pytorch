# 라이브러리 및 변수 설정:
# 필요한 라이브러리를 가져오고, 데이터 처리와 관련된 변수 및 설정을 초기화합니다.
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4

# GPU/CPU 처리
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# MTCNN(Multi-task Cascaded Convolutional Networks) 모델 초기화:
# 얼굴 탐지 및 정렬을 위해 사용됩니다. 모델의 설정과 장치를 설정합니다.
# 기본 매개변수는 설명용으로 표시되었으나 필요하지 않습니다.
# MTCNN은 신경망 및 기타 코드의 모음이므로 내부적으로 필요할 때 오브젝트 복사를 활성화하려면 다음과 같은 방식으로 전달해야 합니다.
# 자세한 내용은 help(MTCNN)을 참조하세요.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# InceptionResnetV1 초기화:
# 얼굴 임베딩을 추출하기 위한 InceptionResnetV1 모델을 vggface2로 미리 학습된 가중치를 사용하여 초기화합니다.
# 사전 학습된 분류기에 대해 classify=True를 설정합니다. 이 예제에서는 이 모델을 사용하여 임베딩/CNN 특징을 출력하겠습니다.
# Face Recognition을 위해 모델을 평가 모드로 설정합니다.
# 자세한 내용은 help(InceptionResnetV1)을 참조하세요.
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    # 'collate_fn' 함수를 통해 배치 데이터를 생성합니다.
    return x[0]

# 데이터 로딩 및 전처리:
# './data/images' 디렉토리에서 이미지 데이터셋을 로드합니다. 이미지와 해당 레이블을 저장하는 데이터셋을 생성합니다.
dataset = datasets.ImageFolder('./data/images')
# 라벨 인덱스를 나중에 ID 이름에 쉽게 레코딩할 수 있도록 데이터 세트에 idx_to_class 속성을 추가합니다.
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

# DataLoader 설정:
# 데이터셋을 DataLoader로 로드합니다.
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

# 얼굴 탐지 및 얼굴 임베딩 추출:
# 잘린 얼굴 이미지가 아닌 경계 상자를 얻으려면 대신 하위 수준 mtcnn.detect() 함수를 호출하면 됩니다.
# 자세한 내용은 help(mtcnn.detect)를 참조하세요.
aligned = []
names = []
# DataLoader에서 얼굴 이미지와 해당 레이블을 하나씩 가져와서 다음 작업을 수행합니다:
for x, y in loader:
    # MTCNN을 사용하여 얼굴을 탐지하고, 얼굴이 감지된 경우 MTCNN 포워드 메서드는 감지된 얼굴에 맞게 잘린 이미지를 반환합니다.
    # 기본적으로 감지된 얼굴은 하나만 반환되며, 감지된 모든 얼굴을 반환하려면 MTCNN 객체를 생성할 때 keep_all=True로 설정합니다.
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        # 탐지된 얼굴 이미지로부터 얼굴 임베딩을 추출하여 해당 얼굴의 레이블과 함께 리스트에 저장합니다.
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])


# 얼굴 간의 거리 계산 및 출력:
# MTCNN은 모두 동일한 크기의 얼굴 이미지를 반환하므로 리셋 인식 모듈로 쉽게 일괄 처리할 수 있습니다.
# 여기서는 이미지가 몇 개 밖에 없으므로 단일 배치를 빌드하고 추론을 수행합니다.
# 실제 데이터 세트의 경우, 특히 GPU에서 처리하는 경우에는 코드를 수정하여 Resnet에 전달되는 배치 크기를 제어해야 합니다.
# 반복 테스트의 경우, 잘린 얼굴이나 경계 상자의 계산을 한 번만 수행하고 감지된 얼굴은 나중에 사용할 수 있도록 저장할 수 있으므로
# 얼굴 감지(MTCNN 사용)와 임베딩 또는 분류(InceptionResnetV1 사용)를 분리하는 것이 가장 좋습니다.

# 추출된 얼굴 임베딩을 사용하여 얼굴 간의 거리(유사도)를 계산합니다.
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
# 계산된 거리는 데이터프레임으로 출력됩니다.
# 출력된 데이터프레임은 얼굴 간의 유사도를 나타내며, 인덱스와 컬럼은 얼굴 이미지의 레이블로 구성됩니다.
print(pd.DataFrame(dists, columns=names, index=names))