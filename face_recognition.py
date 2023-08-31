# 폐쇄망 환경을 위해 커스텀하여 사용 (모델 파일은 README를 참고하여 수동 다운)

from models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import os

# 학습과 인식의 Model 설정값 통일을 위해 구현
from models.setting import Setting
setting = Setting()

class FaceRecog:
    def __init__(self):
        # 안면 인식을 위한 workers
        workers = 0 if os.name == 'nt' else 4

        # MTCNN 호출 (keep_all은 프레임 내 모든 얼굴을 가져오는 옵션)
        mtcnn = MTCNN(device=setting.device)

        # TODO: 학습 데이터와 인식 데이터의 폴더 수가 다르면 에러가 발생한다. 예외 처리 등의 방법 찾기
        # 데이터 로딩 및 전처리:
        # './data/train_images' 디렉토리에서 이미지 데이터셋을 로드합니다. 이미지와 해당 레이블을 저장하는 데이터셋을 생성합니다.
        dataset = datasets.ImageFolder(setting.recog_image_path)
        # 라벨 인덱스를 나중에 ID 이름에 쉽게 레코딩할 수 있도록 데이터 세트에 idx_to_class 속성을 추가합니다.
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

        # 저장된 모델 불러오기 (InceptionResnetV1)
        if not os.path.exists(setting.model_path):
            raise FileNotFoundError("모델 학습을 먼저 수행하세요.")
        else:
            self.resnet = InceptionResnetV1(
                classify=setting.classify,
                pretrained=setting.pretrained,
                num_classes=len(dataset.class_to_idx)
                # 모델을 지정한 디바이스(GPU 또는 CPU)로 옮깁니다.
            ).to(setting.device)
            # TODO: 모델 학습 당시 가중치보다 너무 많이 변했다. 원인 확인
            self.resnet.load_state_dict(torch.load(setting.model_path))
            self.resnet.eval()

        # DataLoader 설정:
        # 데이터셋을 DataLoader로 로드합니다.
        loader = DataLoader(dataset, collate_fn=self.collate_fn, num_workers=workers)

        # 얼굴 탐지 및 얼굴 임베딩 추출:
        # 잘린 얼굴 이미지가 아닌 경계 상자를 얻으려면 대신 하위 수준 mtcnn.detect() 함수를 호출하면 됩니다.
        # 자세한 내용은 help(mtcnn.detect)를 참조하세요.
        aligned = []
        self.names = []

        # DataLoader에서 얼굴 이미지와 해당 레이블을 하나씩 가져와서 다음 작업을 수행합니다:
        for x, y in loader:
            # MTCNN을 사용하여 얼굴을 탐지하고, 얼굴이 감지된 경우 MTCNN 포워드 메서드는 감지된 얼굴에 맞게 잘린 이미지를 반환합니다.
            # 기본적으로 감지된 얼굴은 하나만 반환되며, 감지된 모든 얼굴을 반환하려면 MTCNN 객체를 생성할 때 keep_all=True로 설정합니다.
            x_aligned = mtcnn(x)
            if x_aligned is not None:
                # 탐지된 얼굴 이미지로부터 얼굴 임베딩을 추출하여 해당 얼굴의 레이블과 함께 리스트에 저장합니다.
                aligned.append(x_aligned)
                self.names.append(dataset.idx_to_class[y])

        # 얼굴 간의 거리 계산 및 출력:
        # MTCNN은 모두 동일한 크기의 얼굴 이미지를 반환하므로 리셋 인식 모듈로 쉽게 일괄 처리할 수 있습니다.
        # 여기서는 이미지가 몇 개 밖에 없으므로 단일 배치를 빌드하고 추론을 수행합니다.
        # 실제 데이터 세트의 경우, 특히 GPU에서 처리하는 경우에는 코드를 수정하여 Resnet에 전달되는 배치 크기를 제어해야 합니다.
        # 반복 테스트의 경우, 잘린 얼굴이나 경계 상자의 계산을 한 번만 수행하고 감지된 얼굴은 나중에 사용할 수 있도록 저장할 수 있으므로
        # 얼굴 감지(MTCNN 사용)와 임베딩 또는 분류(InceptionResnetV1 사용)를 분리하는 것이 가장 좋습니다.

        # 추출된 얼굴 임베딩을 사용하여 얼굴 간의 거리(유사도)를 계산합니다.
        known_aligned = torch.stack(aligned).to(setting.device)
        self.known_embeddings = self.resnet(known_aligned).detach().cpu()

        self.mtcnn = MTCNN(keep_all=True, device=setting.device)

    def collate_fn(self, x):
        # 'collate_fn' 함수를 통해 배치 데이터를 생성합니다.
        return x[0]

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