# 자체 데이터 세트에서 InceptionResnetV1 모델을 미세 조정하는 방법을 보여줍니다.
# 이는 대부분 표준 파이토치 트레이닝 패턴을 따릅니다.

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os


# Define run parameters
# 데이터 세트는 VGGFace2/ImageNet 스타일의 디렉토리 레이아웃을 따라야 합니다.
# data_dir을 미세 조정하려는 데이터 세트의 위치로 수정합니다.

# 학습에 사용할 이미지 데이터의 디렉토리 경로
data_dir = '../data/images'

# 한 번의 학습에 사용할 데이터 샘플의 수
batch_size = 32

# 전체 데이터셋을 몇 번 반복해서 학습할 것인지를 결정하는 변수
epochs = 8

# 데이터를 로딩하는 데 사용할 스레드의 수
# os.name == 'nt'는 현재 운영체제가 Windows인지를 체크
# Windows에서는 멀티스레딩이 일부 제한될 수 있어서 0으로 지정
workers = 0 if os.name == 'nt' else 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# Define MTCNN module
# 자세한 내용은 help(MTCNN) 참고하기

# image_size: 얼굴 이미지를 어떤 크기로 변환할지를 결정하는 매개변수, 160으로 설정되어 있으므로, 입력 이미지는 160x160 크기로 변환
# margin: 얼굴 주위에 추가할 여유 공간(margin)을 결정하는 매개변수, 0이면 얼굴 주위에 여유 공간이 추가되지 않습니다.
# min_face_size: 감지할 수 있는 최소 얼굴 크기를 결정, 픽셀 단위이며 가로, 세로 모두 해당
# thresholds: 얼굴 감지 단계에서 사용되는 임계값(threshold)
# factor: 이미지 스케일을 조정하기 위한 스케일 팩터(scale factor)
# post_process: 후처리 과정을 수행할 지 여부를 결정하는 변수, True일 때 후처리 과정을 수행
# device: 얼굴 감지 모델이 실행될 디바이스를 결정, 위에서 설정한 device 변수를 사용하여 설정합니다.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)


# Perfom MTCNN facial detection
# 데이터로더 객체를 반복하여 잘린 얼굴을 얻습니다.


# 이미지 데이터를 불러오기 위해 PyTorch의 ImageFolder 데이터셋 클래스를 사용
# data_dir에서 데이터를 로드하고, transforms.Resize((512, 512))를 통해 이미지 크기를 512x512로 조정
dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))

# 데이터셋의 각 이미지에 대한 경로와 해당 이미지의 클래스 레이블을 저장한 리스트
# 데이터 경로를 원본 데이터 경로에서 잘려진(cropped) 데이터 경로로 변경하는 작업을 수행
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
]

# 데이터셋을 배치로 나누어 로딩하기 위해 PyTorch의 DataLoader를 사용
# num_workers, batch_size, 그리고 collate_fn 등을 설정
# collate_fn은 배치를 형성하는 과정에서 사용할 함수로, 이 경우 training.collate_pil 함수 사용
loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

# DataLoader를 통해 배치 단위로 이미지 데이터와 레이블을 반복적으로 가져옵니다.
# MTCNN 모델을 사용하여 얼굴을 감지하고, 해당 얼굴 이미지를 y 경로에 저장합니다.
# 현재 진행 중인 배치의 번호와 전체 배치의 수를 출력, \r은 커서를 맨 앞으로 옮겨서 덮어쓰기를 통해 진행 상황을 출력
for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

# Remove mtcnn to reduce GPU memory usage
# del mtcnn: GPU 메모리 사용량을 줄이기 위해 MTCNN 모델을 삭제합니다.
del mtcnn


# Define Inception Resnet V1 module
# 자세한 내용은 help(InceptionResnetV1)를 참고하세요.
# num_classes=len(dataset.class_to_idx): 입니다. 데이터셋의 클래스 수와 동일하게 설정합니다.
# 얼굴 임베딩을 생성하는데 사용되는 InceptionResNetV1 모델 초기화
resnet = InceptionResnetV1(
    # 모델을 분류(classification)용으로 사용할 것인지를 나타내는 매개변수
    # True로 설정하면 분류를 위한 출력 레이어가 모델에 포함
    classify=True,

    # 미리 학습된 가중치를 사용하도록 지정하는 매개변수
    # 'vggface2'로 설정하면 VGGFace2 데이터셋에서 학습된 가중치를 사용
    # 미리 학습된 특성을 활용하여 얼굴 임베딩을 더욱 풍부하게 생성
    pretrained='vggface2',

    # 분류 레이어의 출력 클래스 수를 결정하는 변수
    # 모델을 지정한 디바이스(GPU 또는 CPU)로 옮깁니다.
    num_classes=len(dataset.class_to_idx)
).to(device)


# Define optimizer, scheduler, dataset, and dataloader
# Adam 옵티마이저를 생성하여 InceptionResNetV1 모델의 파라미터를 최적화
# 학습률(learning rate)은 0.001로 설정하고 있습니다.
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# 학습률 스케줄러를 생성
# MultiStepLR은 지정된 epoch 수에 따라 학습률을 조정하는 스케줄러
# 여기서는 5번째와 10번째 epoch 이후에 학습률을 조정합니다.
scheduler = MultiStepLR(optimizer, [5, 10])

# 여러 개의 이미지 변환 함수를 조합하여 하나의 변환 파이프라인을 생성
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),

    # 이미지 픽셀 값을 정규화하는 함수
    fixed_image_standardization
])

# 얼굴 이미지 데이터셋을 불러옵니다. 이미지 변환으로 위에서 정의한 변환 파이프라인 trans을 사용합니다.
dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)

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
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
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

# writer의 현재 반복 횟수와 로그를 기록할 반복 간격을 설정
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)

# 모델을 평가 모드로 설정
resnet.eval()

# 학습 또는 검증 단계를 실행하는 함수, 주어진 모델, 손실 함수, 데이터 로더, 옵티마이저, 스케줄러, 메트릭, 디바이스 등을 활용하여
# 한 에폭 동안의 학습 또는 검증을 수행, 결과는 TensorBoard에도 기록
# 검증 데이터셋에서의 초기 검증 결과를 기록
validation_loss = training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

# 반복문을 통해 각 에폭(epoch)마다 학습 및 검증 단계를 반복하면서 모델을 학습

# 가장 낮은 검증 손실을 가진 모델을 저장하기 위한 변수
best_val_loss = validation_loss
best_epoch = 0

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    validation_loss = training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    # 가장 낮은 검증 손실을 가진 모델을 저장
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_epoch = epoch
        torch.save(resnet.state_dict(), 'best_model.pth')

# 학습 및 로깅이 끝난 후 SummaryWriter를 닫습니다.
writer.close()