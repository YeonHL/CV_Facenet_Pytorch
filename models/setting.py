import torch
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


        # 인식용 설정값
        os.makedirs(os.path.join(self.root_path, 'data', 'test_images'), exist_ok=True)
        self.test_image_path = os.path.join(self.root_path, "data", 'test_images')

        self.pickle_path = os.path.join(self.root_path, "models", "recog_pickle.pkl")

        self.threshold = 1.5


        # 모델 학습용 설정값

        # 학습에 사용할 이미지 데이터의 디렉토리 경로
        os.makedirs(os.path.join(self.root_path, 'data','train_images'), exist_ok=True)
        self.train_image_path = os.path.join(self.root_path, "data", 'train_images')

        # 체크포인트 파일을 저장할 디렉토리 경로
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

# TODO: Setting 조회 부분 완성하기
def menu():
    print("\n확인하고 싶은 설정의 번호를 입력해주세요.")
    print("1. 학습 관련 설정")
    print("2. 인식 관련 설정\n")
    print("======== 공통 설정 ========")
    print("3. root_path: 파일을 실행한 디렉토리 경로")
    print("4. model_path: 학습한 모델 파일 경로")
    print("5. device: 실행되고 있는 장치 정보 (GPU/CPU)")
    print("프로그램을 종료하고 싶을 경우 q를 입력해주세요.")

if __name__ == '__main__':
    setting = Setting()
    want_to_see = ""

    while not want_to_see == "q":


        want_to_see = input()

        if want_to_see == "1":
            print("\n======== 학습 관련 설정 ========")
            print("1. batch_size: 한 번의 학습에 사용할 데이터 샘플의 수")
            print("2. end_epochs: 전체 데이터셋을 몇 번 반복해서 학습할지 결정하는 변수")
            print("3. learning_rate: 모델 학습률")
            print("4. scheduler_epoch: 몇 번째 epoch 이후 학습률을 조정할지 결정하는 변수")
            print("5. writer_interval: Writer의 로그 기록 반복 횟수")
            print("6. train_image_path: 학습에 사용할 이미지들의 디렉토리 경로")
            print("7. checkpoint_path: 체크포인트 파일이 저장될 디렉토리 경로")
            print("8. vgg_path: 사전 학습된 VGGFace2 모델 파일 경로")
            print("9. casia_path: 사전 학습된 Casia-Webface 모델 파일 경로")
        elif want_to_see == "2":
            print("\n======== 인식 관련 설정 ========")
            print("1. recog_image_path: 인식에 사용할 이미지들의 디렉토리 경로")
            # 인식 관련 설정 정보 출력 코드 추가
        else:
            print("잘못된 입력입니다. 다시 시도하세요.")

        want_to_see = input()

    print("프로그램을 종료합니다.")