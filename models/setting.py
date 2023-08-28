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

        # 모델 저장 경로
        self.save_path = './models/best_model.pth'

        # 현재 장비 인식 (GPU / CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # MTCNN은 얼굴 탐지 및 정렬을 위해 사용합니다.
        # image_size: 얼굴 이미지를 어떤 크기로 변환할지를 결정하는 매개변수, 160으로 설정되어 있으므로, 입력 이미지는 160x160 크기로 변환
        # margin: 얼굴 주위에 추가할 여유 공간(margin)을 결정하는 매개변수, 0이면 얼굴 주위에 여유 공간이 추가되지 않습니다.
        # min_face_size: 감지할 수 있는 최소 얼굴 크기를 결정, 픽셀 단위이며 가로, 세로 모두 해당
        # thresholds: 얼굴 감지 단계에서 사용되는 임계값(threshold)
        # factor: 이미지 스케일을 조정하기 위한 스케일 팩터(scale factor)
        # post_process: 후처리 과정을 수행할 지 여부를 결정하는 변수, True일 때 후처리 과정을 수행
        # device: 얼굴 감지 모델이 실행될 디바이스를 결정, 위에서 설정한 device 변수를 사용하여 설정합니다.
        # 자세한 내용은 help(MTCNN) 참고하기
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )


        # 모델 학습용 설정값

        # 학습에 사용할 이미지 데이터의 디렉토리 경로
        self.image_path = './data/images'

        # 한 번의 학습에 사용할 데이터 샘플의 수
        self.batch_size = 32

        # 전체 데이터셋을 몇 번 반복해서 학습할 것인지를 결정하는 변수
        self.epochs = 8

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

    # MTCNN 삭제 (사용 후 GPU 메모리 사용량을 줄이기 위해 모델 삭제)
    def del_mtcnn(self):
        del self.mtcnn

