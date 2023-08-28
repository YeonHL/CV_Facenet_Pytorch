from facenet_pytorch import InceptionResnetV1
import torch
import os

# 학습과 인식의 Model 설정값 통일을 위해 구현
from models.setting import Setting
model = Setting()

# 안면 인식을 위한 workers
workers = 0 if os.name == 'nt' else 4

# 얼굴 탐지 및 정렬을 위해 사용
mtcnn = model.mtcnn

# TODO: resnet 처리하기
# 모델 아키텍처 설정
resnet = InceptionResnetV1(pretrained=model.pretrained).eval().to(model.device)

# 저장된 모델 가중치 불러오기
if not os.path.exists(model.save_path):
    raise FileNotFoundError("모델 학습을 먼저 수행하세요.")
else:
    loaded_state_dict = torch.load(model.save_path)
resnet.load_state_dict(loaded_state_dict)
