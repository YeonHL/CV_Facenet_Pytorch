import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, training
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# 가상의 데이터셋 클래스 생성
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 가상의 데이터셋 경로와 라벨 설정
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
labels = [0, 1, ...]  # 각 이미지의 라벨

# 데이터 전처리를 위한 변환 설정
data_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# CustomDataset 인스턴스 생성
dataset = CustomDataset(image_paths, labels, transform=data_transform)

# 데이터로더 생성
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 얼굴 인식 모델 생성
model = InceptionResnetV1(pretrained='vggface2').train()

# 학습 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# 학습
for epoch in range(10):  # 예시로 10 에폭으로 설정
    for images, labels in dataloader:
        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

# 학습이 완료된 모델을 이용하여 인식 수행
mtcnn = MTCNN(image_size=160)
recognizer = training.InceptionResnetV1(num_classes=len(labels))

# 모델 로드
recognizer.load_state_dict(model.state_dict())

# 인식할 이미지 경로 설정
test_image_path = "path/to/test_image.jpg"
test_image = Image.open(test_image_path).convert("RGB")

# 이미지에서 얼굴 추출
boxed_faces = mtcnn.detect(test_image)

# 추출된 얼굴들을 인식 모델에 적용
for i, (box, prob, landmarks) in enumerate(boxed_faces):
    face = test_image.crop(box).resize((160, 160))
    face = data_transform(face).unsqueeze(0)
    embedding = recognizer(face)

    # 가장 유사한 라벨 찾기
    predicted_label = torch.argmax(embedding)

    print(f"Detected face {i + 1} is recognized as person: {predicted_label}")