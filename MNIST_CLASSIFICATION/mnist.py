###############################################################
# 모듈 불러오기
###############################################################

# pytorch 가져오기
import torch
# pytorch의 딥러닝 기본 구성 요소인 torch.nn 모듈을 nn으로 지정하여 불러오기
import torch.nn as nn
# 딥러닝에 자주 사용되는 함수가 포함된 모듈 torch.nn.functional을 F로 지정하여 불러오기
import torch.nn.functional as F
# 최적화 알고리즘을 포함한 torch.optim 모듈을 optim으로 지정하여 불러오기
import torch.optim as optim
# 딥러닝에서 자주 사용되는 데이터셋과 모델 구조 및 이미지 변환 기술을 가지고 있는 Torchvision 모듈에서 dataset과 transforms 함수만 불러오기
from torchvision import datasets, transforms

from matplotlib import pyplot as plt

###############################################################
# 분석환경 설정
###############################################################

# 현재 GPU가 사용 가능하면 1 아니면 0
is_cuda = torch.cuda.is_available()
# device 변수에 모델과 사용하는 데이터에 어떤 장비를 사용할지 지정
device = torch.device('cuda' if is_cuda else 'cpu')

print('Current cuda device is', device)

# HyperParameter 지정

# 모델 가중치를 한 번 업데이트 시킬때 사용되는 샘플 단위 개수 (=미니 배치 사이즈)
batch_size = 50
# 학습 데이터를 모두 사용하여 학습하는 기본 단위 횟수 (=Epoch 수)
epoch_num = 15
# 가중치 업데이트의 정도(=Learning Rate(학습률))
learning_rate = 0.0001

###############################################################
# MNIST 데이터 불러오기
###############################################################

# root : MNIST 데이터를 저장할 물리적 공간
# train : True/False의 논리값. 데이터를 학습용으로 사용할 것 인지?
# download : root 옵션으로 지정한 위치에 데이터를 저장할 것 인지?
# transform : MNIST 데이터를 저장함과 동시에 전처리를 할 수 있는 옵션.
# Pytorch는 입력으로 Tensor를 사용하므로
# 이미지를 Tensor로 변형하는 전처리인 transforms.ToTensor()를 사용함
train_data = datasets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=False,
                           transform=transforms.ToTensor())

print('number of training data: ', len(train_data))
print('number of test data: ', len(test_data))

###############################################################
# MNIST 데이터 확인하기
###############################################################

# 학습데이터, 정답
image, label = train_data[0]

###############################################################
# MNIST 데이터 확인하기
###############################################################

# MNIST는 단일 채널로 [1,28,28] 3차원 텐서
# squeeze() 함수를 통해 크기가 1인 차원을 제거
# [1,28,28] => [28,28]
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title('label : %s' % label)
plt.show()

###############################################################
# 미니 배치 구성하기
###############################################################

# dataset : 미니 배치로 구성할 데이터
# batch_size : 미니 배치의 사이즈. 본 예제에서는 사전에 50으로 지정된 상태
# shuffle : 데이터의 순서를 랜덤으로 섞어 미니 배치를 구성할지 여부 결정
# 시계열 데이터가 아닌 경우, 딥러닝이 데이터의 순서를 학습하지 못하도록 데이터를
# 섞는 과정이 필수
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

first_batch = train_loader.__iter__().__next__()

print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))

# 60000개의 학습 데이터에 50의 배치 사이즈 사용 => 1200개의 미니 배치
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))

# 미니 배치는 두 가지의 요소로 구성
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)),
                                     len(first_batch)))

# 미니 배치의 첫 번째 요소는 [50,1,28,28]의 4차원 Tensor
# 각 요소는 Batch Size, Channel, Width, Height를 나타냄
# 데이터가 여러개 쌓이며 차원이 하나 더 추가되는 것
print('{:15s} | {:<25s} | {}'.format('first_batch[0]',
                                     str(type(first_batch[0])),
                                     first_batch[0].shape))

# 50 크기의 벡터로, 미니 배치의 정답이 저장되어 있음
print('{:15s} | {:<25s} | {}'.format('first_batch[1]',
                                     str(type(first_batch[1])),
                                     first_batch[1].shape))


###############################################################
# 모델 학습
###############################################################

# __init__을 이용하여 모델에 사용되는 가중치 형태를 정의. 이때 이전 Layer의 출력 크기와 직후 Layer의 입력 크기는 같아야 함
# Feature Map 사이즈 공식
# O = [ (I+2P-F) / S ] + 1
# I : 이미지 크기
# P : Padding 크기
# F : Filter 사이즈
# S : Stride 크기

# nn.Module 클래스를 상속받은 CNN 클래스 정의
class CNN(nn.Module):
    # __init__을 통해 모델에서 사용된느 가중치를 정의
    def __init__(self):
        # super() 함수를 통해 nn.Moudle 클래스의 속성을 상속받고 초기화
        super(CNN, self).__init__()

        # 첫 번째 Conv Layer인 conv1 정의.
        # nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1)
        # in_channels : 입력 Tensor의 채널 크기
        # out_channels : 출력 채널의 Tensor 크기
        # kernel_size : Filter의 크기. Scalar로 지정시 가로 세로가 같은 2D Filter 생성. 3 => 3X3 필터
        # stride : Filter가 움직이는 단위. padding 옵션이 없을 때 기본 padding = 0
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # 두 번째 Conv Layer인 conv2 정의
        # conv1의 Output과 conv2의 Input Channel이 같아야 함
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # 0.25 확률의 Dropout 정의
        self.dropout1 = nn.Dropout2d(0.25)

        # 0.5 확률의 Dropout 정의
        self.dropout2 = nn.Dropout2d(0.5)

        # 첫 번째 Fully-connected Layer 정의. 9216 크기의 벡터를 128 크기의 벡터로 변환하는 가중치 설계
        self.fc1 = nn.Linear(9216, 128)

        # 두 번째 Fully-connected Layer 정의. 128 크기의 벡터를 MNIST의 클래스 개수인 10 크기의 벡터로 변환하는 가중치 설계
        self.fc2 = nn.Linear(128, 10)

    # 입력 이미지와 정의한 가중치를 이용한 Feed Forward 연산 정의
    def forward(self, x):
        # 입력 이미지를 conv1 레이어를 통과
        x = self.conv1(x)

        # ReLU 활성 함수를 적용. 활성 함수는 단순 연산이기 때문에 __init__에서 정의한 학습 가중치가 없음
        x = F.relu(x)

        # conv2 레이어 통과
        x = self.conv2(x)

        # ReLU 적용
        x = F.relu(x)

        # (2X2) 크기의 Filter로 Max Pooling 적용. Pooling Layer는 단순 연산이므로 학습할 가중치가 없음
        x = F.max_pool2d(x, 2)

        # 사전에 정의한 0.25 확률의 dropout1 적용
        x = self.dropout1(x)

        # Torch.flatten 함수를 통해 Fully-connected Layer를 통과하기 전, 고차원의 Tensor를 1차원의 벡터로 변환
        # 2개의 Conv Layer와 1번의 MaxPooling으로 만들어진 [64,12,12] 크기의 3차원 tensor가 9216크기의 벡터로 변환됨
        x = torch.flatten(x, 1)

        # 9216 => 128 크기의 벡터로 학습하는 fc1을 통과
        x = self.fc1(x)

        # ReLU 활성 함수를 적용
        x = F.relu(x)

        # 사전에 정의한 0.5 확률의 dropout2 적용
        x = self.dropout2(x)

        # 두 번째 Fully-Connected Layer인 fc2를 통과하며 벡터의 사이즈가 128 => 10으로 줄어듬
        x = self.fc2(x)

        # 최종 출력값으로 log-softmax 연산. log-softmax는 Softmax 대비 연산 속도 향상 효과를 가짐
        output = F.log_softmax(x, dim=1)
        return output


###############################################################
# Optimizer 및 손실 함수 정의
###############################################################

# CNN 클래스를 이용해 model이라는 인스턴스 생성. 이때, 코드 상단에서 지정한 연산 장비(GPU or CPU) device를 인식
model = CNN().to(device)

# 손실 함수를 최소로 하는 가중치를 찾기 위해 Adam 알고리즘의 optimizer를 지정
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# MNIS는 다중 클래스 분류 문제이므로 교차 엔트로피(Cross Entropy)를 손실 함수로 설정
criterion = nn.CrossEntropyLoss()

# 설계한 CNN 모형 확인하기

print(model)

###############################################################
# 모델 학습
###############################################################

# CNN 클래스가 저장된 model 인스턴스를 학습 모드로 실행
model.train()

# 반복 학습 중 손실 함수 현황을 확인하기 위한 학습 횟수를 나타내는 보조 인덱스 저장
i = 0

# 미리 지정해둔 Epoch 수만큼 반복 학습 for문 선언
for epoch in range(epoch_num):

    # 학습 데이터를 batch_size로 나눈 만큼 반복 수행되며, train_loader는 매 시행마다 미니 배치의 데이터와 정답을 data와 target에 할당
    for data, target in train_loader:

        # 미니 배치의 데이터를 기존에 지정한 장비 device에 할당
        data = data.to(device)

        # 미니 배치의 정답을 기존에 지정한 장비 device에 할당
        target = target.to(device)

        # 학습을 시작하기 전에 이전 반복 시행에서 저장된 optimizer의 Gradient 초기화
        optimizer.zero_grad()

        # 미니 배치 데이터를 모델에 통과시키는 Feed Forward 연산을 결과값을 계산
        output = model(data)

        # 계산된 결과값과 실제 정답으로 손실 함수 계산
        loss = criterion(output, target)

        # 손실 함수를 통해 Gradient 계산
        loss.backward()

        # Gradient를 바탕으로 모델의 가중치 업데이트
        optimizer.step()

        # 1000 번째 시행마다 손실함수를 확인
        if i % 1000 == 0:
            # 손실 함수 출력
            print('Train Step: {}\tLoss: {:.3f}'.format(i, loss.item()))

        # 학습 완료 시 보조 인덱스 1 올리기
        i += 1

###############################################################
# 모델 평가
###############################################################

# 평가 모드를 실행. eval() 함수 호출시 Dropout이 적용되지 않고, Batch-Normalization도 평가 모드로 전환
model.eval()

# 정답 개수를 지정할 correct를 초기화
correct = 0

# 테스트 데이터를 batch_size로 나눈 만큼 반복 수행. test_loader는 매 시행마다 미니 배치의 데이터와 정답을 data와 target에 저장
for data, target in test_loader:

    # 미니 배치의 데이터를 기존에 지정한 장비 device에 할당
    data = data.to(device)

    # 미니 배치의 정답을 기존에 지정한 장비 device에 할당
    target = target.to(device)

    # 미니 배치 데이터를 모델에 통과시켜 결과값을 계산
    output = model(data)

    # Log-Softmax 값이 가장 큰 인덱스를 예측값으로 지정
    prediction = output.data.max(1)[1]

    # 실제 정답과 예측값이 같으면 True, 다르면 False인 논리 값으로 구성된 벡터를 더함. 즉, 미니 배치 중 정답 개수를 구하고 반복 시행마다 누적하여 더함
    correct += prediction.eq(target.data).sum()

# 전체 테스트 데이터 중 맞춘 개수의 비율을 통해 정확도를 계산하여 출력
print('Test set: Accuracy: {:.2f}%'.format(100 * correct /
                                           len(test_loader.dataset)))
