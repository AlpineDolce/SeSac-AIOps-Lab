import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# --- 1. 모델 및 하이퍼파라미터 정의 ---

# FastAPI 서버에서는 이 파일의 ImageClassifier 클래스와 가중치 파일(trained_model_weights.pth)만 필요합니다.
# 아래의 학습 코드는 별도의 train.py 파일로 분리하거나, if __name__ == "__main__": 블록 안에서만 실행되도록 하여
# FastAPI 서버가 실행될 때 학습 코드가 실행되지 않도록 하는 것이 좋습니다.

# 장치 설정 (GPU 사용 가능하면 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터
batch_size = 64
learning_rate = 0.001
epochs = 10 # 데모를 위해 epoch 수를 줄임
MODEL_PATH = 'trained_model_weights.pth'

# 완전연결신경망 모델 정의
class ImageClassifier(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=500, num_class=10):
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # 이미지를 1차원 벡터로 변환
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- 2. 데이터 준비 (학습 시에만 필요) ---

def get_data_loaders(batch_size):
    """MNIST 데이터셋을 위한 DataLoader를 생성하고 반환합니다."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# --- 3. 모델 학습 및 평가 (학습 시에만 필요) ---

def train_model(model, train_loader, criterion, optimizer, epochs):
    """모델을 학습시킵니다."""
    model.train() # 모델을 학습 모드로 설정
    print("모델 학습을 시작합니다...")
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    print("모델 학습 완료.")

def evaluate_model(model, test_loader):
    """학습된 모델을 평가합니다."""
    model.eval() # 모델을 평가 모드로 설정
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"테스트셋 정확도: {accuracy:.2f}%")
    return accuracy

# --- 4. FastAPI를 위한 예측 함수 ---

def load_model(model_path):
    """저장된 모델 가중치를 불러옵니다."""
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 추론을 위해 평가 모드로 설정
    return model

def predict(model, image_tensor):
    """단일 이미지에 대해 예측을 수행합니다."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

# --- 5. 스크립트 직접 실행 시 학습 수행 ---

if __name__ == "__main__":
    # 모델 인스턴스 생성
    model = ImageClassifier().to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 데이터 로더 가져오기
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, epochs)
    
    # 모델 평가
    evaluate_model(model, test_loader)
    
    # 학습된 모델 가중치 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"학습된 모델을 '{MODEL_PATH}'에 저장했습니다.")

    # --- 저장된 모델 로드 및 예측 예시 ---
    print("\n--- 예측 예시 ---")
    # 모델 로드
    loaded_model = load_model(MODEL_PATH)
    
    # 테스트 데이터셋에서 이미지 하나 가져오기
    example_image, example_label = next(iter(test_loader))
    example_image_tensor = example_image[0] # 첫번째 이미지
    
    # 예측 수행
    prediction = predict(loaded_model, example_image_tensor)
    
    print(f"실제 레이블: {example_label[0].item()}")
    print(f"모델 예측: {prediction}")
