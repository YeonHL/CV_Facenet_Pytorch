from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os


# 학습과 인식의 Model 설정값 통일을 위해 구현
from models.setting import Setting
setting = Setting()


workers = 0 if os.name == 'nt' else 8


# 이미지 데이터를 불러오기 위해 PyTorch의 ImageFolder 데이터셋 클래스를 사용
# data_dir에서 데이터를 로드하고, transforms.Resize((512, 512))를 통해 이미지 크기를 512x512로 조정
dataset = datasets.ImageFolder(setting.image_path, transform=transforms.Resize((512, 512)))

# 데이터셋의 각 이미지에 대한 경로와 해당 이미지의 클래스 레이블을 저장한 리스트
# 데이터 경로를 원본 데이터 경로에서 잘려진(cropped) 데이터 경로로 변경하는 작업을 수행
dataset.samples = [
    (p, p.replace(setting.image_path, setting.image_path + '_cropped'))
        for p, _ in dataset.samples
]


# 데이터셋을 배치로 나누어 로딩하기 위해 PyTorch의 DataLoader를 사용
# num_workers, batch_size, 그리고 collate_fn 등을 설정
# collate_fn은 배치를 형성하는 과정에서 사용할 함수로, 이 경우 training.collate_pil 함수 사용
loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=setting.batch_size,
    collate_fn=training.collate_pil
)

# DataLoader를 통해 배치 단위로 이미지 데이터와 레이블을 반복적으로 가져옵니다.
# MTCNN 모델을 사용하여 얼굴을 감지하고, 해당 얼굴 이미지를 y 경로에 저장합니다.
# 현재 진행 중인 배치의 번호와 전체 배치의 수를 출력, \r은 커서를 맨 앞으로 옮겨서 덮어쓰기를 통해 진행 상황을 출력
for i, (x, y) in enumerate(loader):
    setting.mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

# del mtcnn: GPU 메모리 사용량을 줄이기 위해 MTCNN 모델을 삭제합니다.
setting.del_mtcnn()


# TODO: resnet 처리하기
# Define Inception Resnet V1 module
# 자세한 내용은 help(InceptionResnetV1)를 참고하세요.
# num_classes=len(dataset.class_to_idx): 입니다. 합니다.

# 얼굴 임베딩을 생성하는데 사용되는 InceptionResNetV1 모델 초기화
# 데이터셋마다 num_classes가 달라지므로 하나의 모델 파일만 사용하는 것이 불가능
# 폐쇄망에서 사용하기 위해선
resnet = InceptionResnetV1(
    classify=setting.classify,
    pretrained=setting.pretrained,
    num_classes=len(dataset.class_to_idx)
    # 모델을 지정한 디바이스(GPU 또는 CPU)로 옮깁니다.
).to(setting.device)


# Define optimizer, scheduler, dataset, and dataloader
# Adam 옵티마이저를 생성하여 InceptionResNetV1 모델의 파라미터를 최적화
# 학습률(learning rate)은 0.001로 설정하고 있습니다.
optimizer = optim.Adam(resnet.parameters(), setting.learning_rate)

# 학습률 스케줄러를 생성
# MultiStepLR은 지정된 epoch 수에 따라 학습률을 조정하는 스케줄러
# 여기서는 5번째와 10번째 epoch 이후에 학습률을 조정합니다.
scheduler = MultiStepLR(optimizer, setting.scheduler_epoch)

# 여러 개의 이미지 변환 함수를 조합하여 하나의 변환 파이프라인을 생성
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),

    # 이미지 픽셀 값을 정규화하는 함수
    fixed_image_standardization
])

# 얼굴 이미지 데이터셋을 불러옵니다. 이미지 변환으로 위에서 정의한 변환 파이프라인 trans을 사용합니다.
dataset = datasets.ImageFolder(setting.image_path + '_cropped', transform=trans)

# 데이터셋 내의 이미지 인덱스를 담은 배열
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)

# 전체 데이터셋 중 학습용과 검증용으로 분할된 이미지 인덱스 배열
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]


# 각각 학습 데이터와 검증 데이터를 로드하는 DataLoader
# SubsetRandomSampler를 사용하여 지정된 인덱스를 기반으로 데이터를 로드
# 학습과 검증 데이터가 무작위로 섞이며 배치 단위로 데이터를 가져옵니다.
train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=setting.batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=setting.batch_size,
    sampler=SubsetRandomSampler(val_inds)
)


# Define loss and evaluation functions
# 크로스 엔트로피 손실 함수를 초기화, 분류 문제에서 주로 사용되며, 신경망 출력과 실제 레이블 간의 손실을 계산
loss_fn = torch.nn.CrossEntropyLoss()

# 모델 평가를 위한 여러 메트릭(metric)을 포함하는 딕셔너리
metrics = {
    # 학습 속도(fps)를 측정하기 위한 training.BatchTimer() 인스턴스를 생성, 학습 과정에서 배치 당 속도를 측정하는데 사용
    'fps': training.BatchTimer(),

    # 정확도(accuracy)를 계산하기 위한 training.accuracy 함수를 설정, 주어진 모델의 예측과 실제 레이블 간의 일치 정도를 계산
    'acc': training.accuracy
}


# Train model
# TensorBoard를 사용하여 학습 과정을 시각화하기 위한 SummaryWriter를 생성
writer = SummaryWriter()
writer.iteration = setting.writer_iteration
writer.interval = setting.writer_interval


print('\n\nInitial')
print('-' * 10)

# 모델을 평가 모드로 설정
resnet.eval()

# 학습 또는 검증 단계를 실행하는 함수, 주어진 모델, 손실 함수, 데이터 로더, 옵티마이저, 스케줄러, 메트릭, 디바이스 등을 활용하여
# 한 에폭 동안의 학습 또는 검증을 수행, 결과는 TensorBoard에도 기록
# 검증 데이터셋에서의 초기 검증 결과를 기록
validation_loss = training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=setting.device,
    writer=writer
)

# 반복문을 통해 각 에폭(epoch)마다 학습 및 검증 단계를 반복하면서 모델을 학습

# 가장 낮은 검증 손실을 가진 모델을 저장하기 위한 변수
best_val_loss = validation_loss
best_epoch = 0

for epoch in range(setting.epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, setting.epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=setting.device,
        writer=writer
    )

    resnet.eval()
    validation_loss = training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=setting.device,
        writer=writer
    )

    # 가장 낮은 검증 손실을 가진 모델을 저장
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_epoch = epoch
        torch.save(resnet.state_dict(), setting.save_path)

# 학습 및 로깅이 끝난 후 SummaryWriter를 닫습니다.
writer.close()