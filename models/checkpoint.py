import torch

# 학습과 인식의 Model 설정값 통일을 위해 구현
from models.setting import Setting

setting = Setting()

def save_checkpoint(model, optimizer, loss, epoch):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch
    }
    torch.save(checkpoint, setting.checkpoint_path)

def load_checkpoint(model, optimizer):
    checkpoint = torch.load(setting.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    setting.start_epoch = checkpoint['epoch']
    return model, optimizer, loss

if __name__ == '__main__':
    pass