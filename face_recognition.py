# 폐쇄망 환경을 위해 커스텀하여 사용 (모델 파일은 README를 참고하여 수동 다운)

from models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import os
import pickle

# 학습과 인식의 Model 설정값 통일을 위해 구현
from models.setting import Setting
setting = Setting()

with open(setting.pickle_path, 'rb') as f:
    loaded_data = pickle.load(f)

test_dataset = datasets.ImageFolder(setting.test_image_path)
test_dataset.idx_to_class = {i: c for c, i in test_dataset.class_to_idx.items()}

class FaceRecog:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True, device=setting.device)
        self.resnet = InceptionResnetV1(
            classify=setting.classify,
            pretrained=setting.pretrained,
            num_classes=len(test_dataset.class_to_idx)
            # 모델을 지정한 디바이스(GPU 또는 CPU)로 옮깁니다.
        ).to(setting.device)

        self.resnet.load_state_dict(torch.load(setting.model_path))
        self.resnet.eval()

        self.known_embeddings = loaded_data['known_embeddings']
        self.names = loaded_data['names']

    def take_frame(self):
        # TODO: 전송 받은 프레임을 가져오는 코드
        # 아래는 예시 코드
        from PIL import Image
        image_path = './data/test.jpg'
        frame = Image.open(image_path).convert('RGB')

        return frame

    def face_recog(self, frame):
        input_aligned = []

        # 이미지 프레임에서 얼굴 탐지
        faces = self.mtcnn(frame)

        # TODO: 인식된 얼굴의 수가 전체 얼굴 수보다 많을 경우 threshold 값을 낮추는 코드 작성
        if faces is not None:
            for face in faces:
                input_aligned.append(face)
            # 탐지된 얼굴들의 임베딩 추출
            input_aligned = torch.stack(input_aligned).to(setting.device)
            input_embeddings = self.resnet(input_aligned).detach().cpu()

            # 얼굴 간의 거리(유사도) 계산
            dists = [[(e1 - e2).norm().item() for e2 in input_embeddings] for e1 in self.known_embeddings]

            # 유사도가 1보다 작은 얼굴들의 인덱스 찾기
            similar_faces = set()
            for i, row in enumerate(dists):
                for j, dist in enumerate(row):
                    if dist < setting.threshold:  # 유사도가 1보다 작을 경우
                        similar_faces.add(self.names[i])

            print(self.names)
            print(dists)
            print(similar_faces)

        else:
            print("얼굴이 탐지되지 않았습니다.")


if __name__ == '__main__':
    facerecog = FaceRecog()
    frame = facerecog.take_frame()
    facerecog.face_recog(frame)
