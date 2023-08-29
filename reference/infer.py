import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image

# 얼굴 인식 모델 생성
mtcnn = MTCNN(image_size=160)
recognizer = InceptionResnetV1(pretrained='vggface2').eval()


# 이미지에서 얼굴 추출
def extract_faces(image_path):
    image = Image.open(image_path).convert("RGB")
    boxed_faces = mtcnn.detect(image)
    faces = []

    for box, _, _ in boxed_faces:
        face = image.crop(box).resize((160, 160))
        faces.append(face)

    return faces


# 얼굴 인식
def recognize_faces(faces):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    recognized_people = []

    for face in faces:
        face_tensor = data_transform(face).unsqueeze(0)
        embedding = recognizer(face_tensor)

        # 이 부분에서 실제로는 어떤 방식으로 인식 결과를 처리할지 정해야 합니다.
        # 여기서는 임시로 랜덤한 라벨을 생성하여 출력하도록 했습니다.
        predicted_label = torch.randint(high=5, size=(1,)).item()  # 임의의 라벨 생성
        recognized_people.append(predicted_label)

    return recognized_people


# 테스트 이미지 경로 설정
test_image_path = "path/to/test_image.jpg"

# 얼굴 추출 및 인식
faces = extract_faces(test_image_path)
recognized_people = recognize_faces(faces)

# 인식된 결과 출력
for i, recognized_label in enumerate(recognized_people):
    print(f"Detected face {i + 1} is recognized as person: {recognized_label}")