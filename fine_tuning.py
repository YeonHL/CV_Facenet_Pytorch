# 다음 예제는 InceptionResnetV1 모델을 사용자의 데이터셋에 맞게 미세 조정하는 방법을 보여줍니다. 이는 주로 표준 PyTorch 학습 패턴을 따를 것입니다.

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

# 라이브러리 및 변수 설정:
# 코드 시작 부분에서 필요한 라이브러리를 가져오고, 데이터 디렉토리 및 학습 설정에 필요한 변수를 초기화합니다.
# 데이터 세트는 VGGFace2/ImageNet 스타일의 디렉토리 레이아웃을 따라야 합니다. data_dir을 미세 조정하려는 데이터 세트의 위치로 수정합니다.

data_dir = './data/test_images'

# 한 번의 학습 단계에서 사용되는 데이터 배치의 크기
batch_size = 32

# 전체 데이터셋을 학습하는 데 필요한 에폭(epoch) 수 에폭은 전체 데이터셋을 한 번 훑는 것을 의미
epochs = 8

# 데이터를 불러오고 전처리하는 데 사용되는 워커(작업자)의 수
# PyTorch의 DataLoader는 병렬화를 통해 데이터를 효율적으로 로드하는데, workers 수는 병렬 처리에 사용되는 프로세스 수를 결정
# Windows 환경에서는 멀티프로세싱이 일부 문제를 일으킬 수 있어서 0으로 설정하여 비활성화하는 경우가 종종 있습니다.
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# MTCNN (Multi-task Cascaded Convolutional Networks) 초기화:
# 얼굴을 탐지하고 잘라내기 위한 MTCNN 모델을 초기화합니다. 이 모델은 얼굴 감지와 얼굴 정렬을 수행합니다. 얼굴을 탐지한 후 이미지를 해당 경로에 저장하는 작업을 수행합니다.
# See help(MTCNN) for more details.

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# 데이터 로딩 및 전처리:
# 이미지 데이터를 로드하고 전처리합니다. 입력 이미지 크기를 조정하고, 데이터를 크롭한 이미지를 새로운 디렉토리에 저장합니다.
# 데이터로더 객체를 반복하여 잘린 얼굴을 얻습니다.

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
]

# DataLoader 설정:
# 전처리된 데이터를 DataLoader로 로드합니다. 학습 데이터와 검증 데이터를 랜덤하게 샘플링하여 배치로 나누어줍니다.

loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)


for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

# Remove mtcnn to reduce GPU memory usage
del mtcnn

# 모델 초기화:
# InceptionResnetV1 모델을 초기화합니다. 이 모델은 얼굴 특징을 추출하고, 얼굴 인식을 수행하는 역할을 합니다.
# See help(InceptionResnetV1) for more details.

resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(dataset.class_to_idx)
).to(device)

# 옵티마이저 및 스케줄러 설정:
# 모델의 학습을 위한 옵티마이저와 학습률 스케줄러를 초기화합니다.

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

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

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

# TensorBoard 작성기 초기화 및 사용 (SummaryWriter):
# 학습 중에 모델의 성능 및 메트릭을 기록하기 위해 TensorBoard 작성기를 초기화하고 사용합니다. 학습 및 검증 과정에서 얻은 정보를 TensorBoard에 기록하여 모니터링하고 시각화할 수 있습니다.

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

# 학습 및 검증:
# 초기화된 모델과 설정을 사용하여 학습과 검증을 진행합니다. 주요 루프에서는 다음 작업이 반복됩니다:
#
# 검증 데이터를 사용하여 초기 모델의 성능을 확인합니다.
# 주어진 에폭 수 만큼 모델을 학습시킵니다.
# 학습된 모델을 검증 데이터를 사용하여 평가합니다.
# 각 에폭의 결과 및 성능 메트릭은 TensorBoard에 기록됩니다.

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
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()