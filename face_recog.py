# 폐쇄망 환경을 위해 커스텀하여 사용 (모델 파일은 README를 참고하여 수동 다운)

import cv2
import models.inception_resnet_v1 as resnetv1
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import torch
import os


# 학습과 인식의 Model 설정값 통일을 위해 구현
from models.setting import Setting
setting = Setting()


# 안면 인식을 위한 workers
workers = 0 if os.name == 'nt' else 4

# MTCNN 호출 (keep_all은 프레임 내 모든 얼굴을 가져오는 옵션)
mtcnn = MTCNN(device=setting.device)
# 사용해야 할 코드: mtcnn = MTCNN(keep_all=True, device=setting.device)

# 저장된 모델 불러오기 (InceptionResnetV1)
if not os.path.exists(setting.model_path):
    raise FileNotFoundError("모델 학습을 먼저 수행하세요.")
else:
    resnet = torch.load(setting.model_path, map_location=setting.device)

# # TODO: 클래스 이름 불러오기
# class_names = ['person1', 'person2', 'person3', ...]
#
# class FaceRecog:
#     def __init__(self):
#         # TODO: 어떤 값이 필요할지 생각해보기
#         pass
#
#     def take_frame(self):
#         # TODO: 전송 받은 프레임을 가져오는 코드
#         # 아래는 예시 코드
#         frame_path = 'input_frame.png'
#         frame = cv2.imread(frame_path)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     def face_recog(self, frame):
#         # TODO: 입력 받은 frame에 안면 인식을 수행해서 결과를 반환하는 코드
#         boxes, _ = mtcnn.detect(frame)
#
#         if boxes is not None:
#             for box in boxes:
#                 box = box.astype(int)
#
#                 # Extract the detected face
#                 face = frame[box[1]:box[3], box[0]:box[2]]
#
#                 # Preprocess the face for recognition
#                 face = cv2.resize(face, (160, 160))
#                 face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float()
#                 face = (face - 127.5) / 128.0
#
#                 # Perform face recognition using the InceptionResnetV1 model
#                 embeddings = resnet(face)
#
#                 # Assuming you have a function to match embeddings to class names
#                 recognized_name = match_embedding_to_name(embeddings, class_names)
#
#                 print(f"Detected face! Recognized as: {recognized_name}")
#         else:
#             print("No faces detected in the frame.")


if __name__ == '__main__':
    def collate_fn(x):
        # 'collate_fn' 함수를 통해 배치 데이터를 생성합니다.
        return x[0]


    # 데이터 로딩 및 전처리:
    # './data/images' 디렉토리에서 이미지 데이터셋을 로드합니다. 이미지와 해당 레이블을 저장하는 데이터셋을 생성합니다.
    dataset = datasets.ImageFolder(setting.image_path)
    # 라벨 인덱스를 나중에 ID 이름에 쉽게 레코딩할 수 있도록 데이터 세트에 idx_to_class 속성을 추가합니다.
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

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
    aligned = torch.stack(aligned).to(setting.device)
    embeddings = resnet(aligned).detach().cpu()

    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    # 계산된 거리는 데이터프레임으로 출력됩니다.
    # 출력된 데이터프레임은 얼굴 간의 유사도를 나타내며, 인덱스와 컬럼은 얼굴 이미지의 레이블로 구성됩니다.
    print(pd.DataFrame(dists, columns=names, index=names))