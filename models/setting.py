from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, _utils
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os

class Setting:
    def __init__(self):
        # 공통 설정값

        # 실행 경로
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # 모델 저장 경로
        self.model_path = os.path.join(self.root_path, "models", "trained_model.pt")

        # 현재 장비 인식 (GPU / CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # TODO: 임베딩을 위한 목록 파일 경로

        # 모델 학습용 설정값

        # 학습에 사용할 이미지 데이터의 디렉토리 경로
        self.image_path = os.path.join(self.root_path,"data",'images')

        os.makedirs(os.path.join(self.root_path, 'checkpoints'), exist_ok=True)
        self.checkpoint_path = os.path.join(self.root_path,"checkpoints","checkpoint.pth")

        # vggface2 모델 경로
        self.vgg_path = os.path.join(self.root_path,"models",'20180402-114759-vggface2.pt')

        # casia-webface 모델 경로
        self.casia_path = os.path.join(self.root_path,"models",'20180408-102900-casia-webface.pt')

        # 한 번의 학습에 사용할 데이터 샘플의 수
        self.batch_size = 32

        # 전체 데이터셋을 몇 번 반복해서 학습할 것인지를 결정하는 변수
        self.start_epoch = 0
        self.end_epochs = 8

        # 학습률
        self.learning_rate = 0.001

        # 학습률 스케줄러
        # 5번째와 10번째 epoch 이후에 학습률을 조정합니다.
        self.scheduler_epoch = [5,10]

        # Writer의 현재 반복 횟수
        self.writer_iteration = 0

        # Writer의 로그 기록 반복 간격
        self.writer_interval = 10


        # 모델 다운용 설정값
        # Pretrained 모델 선택 (vggface2, casia-webface)
        # 미리 학습된 특성을 활용하여 얼굴 임베딩을 더욱 풍부하게 생성
        # 선택하지 않고 classify가 True라면 num_classes를 반드시 설정해야 한다.
        self.pretrained = 'vggface2'

        # 수동으로 입력할 경우 데이터셋의 클래스 수와 동일하게 설정
        # Pretrained 모델을 사용한다고 가정하여 None 설정
        self.num_classes = None

        # 모델을 분류(classification)용으로 사용할 것인지를 나타내는 매개변수
        # True로 설정하면 분류를 위한 출력 레이어가 모델에 포함
        self.classify = True

if __name__ == '__main__':
    pass