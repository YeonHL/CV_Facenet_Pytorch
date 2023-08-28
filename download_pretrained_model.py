import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms

from models.setting import Setting
setting = Setting()

dataset = datasets.ImageFolder(setting.image_path, transform=transforms.Resize((512, 512)))

# 데이터셋의 각 이미지에 대한 경로와 해당 이미지의 클래스 레이블을 저장한 리스트
# 데이터 경로를 원본 데이터 경로에서 잘려진(cropped) 데이터 경로로 변경하는 작업을 수행
dataset.samples = [
    (p, p.replace(setting.image_path, setting.image_path + '_cropped'))
    for p, _ in dataset.samples
]

# 모델 생성
resnet = InceptionResnetV1(
    classify=setting.classify,
    pretrained=setting.pretrained,
    num_classes=len(dataset.class_to_idx)
).to(setting.device)

# 모델 저장
torch.save(resnet.state_dict(), 'inception_resnet.pth')
