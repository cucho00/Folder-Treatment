# gcn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Local_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, out_channels):
        super(Local_GCN, self).__init__()
        # 3-layer GCN 진행 = 총 3번의 GCN 연산을 수행하여 결과를 도출
        # 시작노드에서 총 3번의 이웃정보(3 hop) 집계를 수행
            # 데이터는 트리구조 그래프의 모든 노드가 포함됨으로,
            # 각 노드와 그 노드에 연결된 다른 노드간의 이웃정보를 고려하여 결과를 도출한다.
            # 이때 유향 간선과 같은 간선의 특징은 무시되며 노드와 간선이 연결되어있는가 만으로 hop이 결정된다.
        self.conv1 = GCNConv(in_channels, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.conv3 = GCNConv(hidden2, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)  # 마지막에는 softmax or cross_entropy로 해석
        return x
    
# 모델 학습을 위한 함수
    # 아래의 4개의 값을 입력
    # 모델, 학습 데이터, 학습 반복 횟수, 학습 시 가중치 변환 수치
        # epoches = 학습 반복 횟수 (5번 정도가 적당, 초반엔 10번도 상관없음)     
        # lr = 가중치 변환 값 (처음엔 크게, 이후 점 점 줄여가며 자세한 가중치 조절)
def train_fn(model, data, epochs=10, lr=0.01):                  
    model.train()   # 학습 모드로 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # 모델의 파라미터(가중치)를 업데이트할 옵티마이저 정의
            # Adam은 모멘텀 + 학습률 자동 조절을 활용하는 대표적인 옵티마이저
    criterion = torch.nn.CrossEntropyLoss()
        # 모델의 출력과 정답 레이블의 차이를 계산하여 손실값(loss)를 만드는 손실 함수 정의
            # 분류 문제에서 많이 사용하는 CrossEntropyLoss는 
            # 소프트 맥스(Softmax) + dmadml fhrm dneh(NLL)을 합친 것

    # epoches 만큼 전체 데이터셋 학습 반복
        # 기울기 초기화
        # 데이터셋에서 문제(data.x, data.edge_index)를 모델에 입력 후, 결과 도출
        # 모델의 결과와 문제의 정답(data.y)비교 후, 손실값 계산(정답 오차)
        # 손실값을 이용해 가중치 계산하여 모델의 모든 가중치를 수정(학습)
    for epoch in range(epochs):
        optimizer.zero_grad()                       
            # 이전 epoch에서 계산된 gradient(기울기)를 초기화
            # PyTorch는 기본적으로 gradient를 누적시키므로 매 epoch마다 초기화 해야 함
        out = model(data.x, data.edge_index)        
            # 모델에 data.x(노드 특성), data.edge_index(간선 정보)를 입력
            # out에 각 노드에 대한 예측 결과(logit)가 저장됨
        loss = criterion(out, data.y)  
            # 모델의 예측 결과(out)과 정답(data.y)를 비교하여 손실값 계산
        loss.backward()
            # 손실을 기준으로 모든 파라미터에 대해 기울기를 자동으로 계산
        optimizer.step()
            # 계산된 gradient를 기반으로 모델 파라미터를 업데이트 (실제 학습이 이루어짐)

    last_loss = loss.item()  # 마지막 loss 저장

    return last_loss   # 마지막 epoch의 손실값 반환

# 모델 저장 함수
def save_model(model, path: str):
    torch.save(model.state_dict(), path)

# 저장된 모델 불러오기
def load_model(model, path: str, map_location=None):
    sd = torch.load(path, map_location=map_location)
    model.load_state_dict(sd)
    return model

# 실제 모델 사용 함수
def predict_fn(model, data, return_prob=False):
    model.eval()  # 추론 모드
    with torch.no_grad():   # 연산 그래프 사용 안함 
        # with 안의 코드 수행중에만 torch.no_grad() 효과가 적용됨
        # 학습이 아닌 추론 모드이기 때문에 역전파 알고리즘을 사용할 필요가 없음 (메모리 절약 + 속도 향상)

        out = model(data.x, data.edge_index)   # [N, C] 노드별 예측 결과 (model의 forward 함수를 실행) 
        pred = out.argmax(dim=1)               # [N] 클래스 인덱스 (결과에서 예측값 추출)

        if return_prob: # 확률까지 계산해서 반환 / 클래스 번호만 반환 선택
            probs = F.softmax(out, dim=1)      # [N, C] 클래스 번호의 노드마다의 확률 계산
            return pred, probs      # 노드별 확률 반환
        else:
            return pred     # 클래스 번호(예측 값) 반환
                # 단순히 숫자 하나로 결과를 받을거면 이걸로 반환