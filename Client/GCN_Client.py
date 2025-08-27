# client.py (for Flower)
import flwr as fl
from flwr.common import GetParametersIns, GetParametersRes, Parameters, Status, Code, FitIns, FitRes, EvaluateIns, EvaluateRes
import io
import numpy as np
import torch
from sklearn.metrics import accuracy_score  # 정확도 계산

from Local_GCN_Module import Local_GCN, train_fn, predict_fn
from Folder_Tree import build_graph_from_folder


# 평가 함수 (evaluate)
def evaluate_model(model, data):
    # 데이터셋을 사용하여 모델을 평가 (예: 정확도)
    model.eval()  # 평가 모드로 설정
    predictions = model(data.x, data.edge_index)  # 모델의 예측값 계산
    accuracy = accuracy_score(data.y.cpu(), predictions.argmax(dim=1).cpu())  # 정확도 계산
    return accuracy

# 연합학습 Client 클래스
class GCNClient(fl.client.Client):
    def __init__(self, model, train_fn, PyG_data):
        self.model = model
        self.train_fn = train_fn    # 클라이언트 로컬 학습 함수
        self.data = PyG_data        # 학습 데이터 (PyG 데이터 셋)
        #self.data_len = data_len   # 해당 클라이언트의 가중치 영향력 설정 (연합학습에서의 영향력)


    # 현재 모델 가중치 전송
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        state_dict = self.model.state_dict()
        tensors = []
        for param in state_dict.values():
            arr = param.cpu().numpy()
            buffer = io.BytesIO()
            # np.save로 저장해야 Flower 서버에서 문제 없이 np.load로 읽는다!
            np.save(buffer, arr, allow_pickle=False)
            tensors.append(buffer.getvalue())
        parameters = Parameters(tensor_type="numpy", tensors=tensors)
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=parameters
        )
        

    # 서버에서 받은 가중치로 모델 업데이트
    def set_parameters(self, parameters: Parameters):
        state_dict = self.model.state_dict()
        param_keys = list(state_dict.keys())
        # parameters.tensors는 bytes의 리스트
        for k, v_bytes in zip(param_keys, parameters.tensors):
            # bytes → numpy → tensor
            arr = np.load(io.BytesIO(v_bytes), allow_pickle=False)
            tensor = torch.from_numpy(arr)
            state_dict[k] = tensor
        self.model.load_state_dict(state_dict)


    # 모델 학습 후, 업데이트 된 가중치 서버로 반환
    def fit(self, ins: FitIns) -> FitRes:
        self.set_parameters(ins.parameters)                 # 서버에서 받은 가중치로 초기화
        loss = self.train_fn(self.model, self.data)            # 로컬 학습 (폴더 그래프 기반), 손실값 받아오기
        params = self.get_parameters(GetParametersIns(config={}))
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=params.parameters,
            num_examples=self.data.x.size(0),
            metrics={"loss": loss} 
        )


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self.set_parameters(ins.parameters)
        accuracy = evaluate_model(self.model, self.data)
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=0.0,  # 실제로 손실 값을 넣을 것!
            num_examples=self.data.x.size(0),
            metrics={"accuracy": accuracy}
        )






# 클라이언트 실행 예시
if __name__ == "__main__":

    # [PyG 데이터 셋 생성]
    data = build_graph_from_folder(r"H:\Project_work\Roots(TestFolder)", label=2, with_label=False)           # 해당 폴더의 데이터셋 생성 (추론용)
        # 데이터 셋으로 만들 폴더 위치, 폴더 구조의 정리 방식(0~2 사이, 추론용 데이터셋에는 의미 없음), 추론용 데이터셋 생성(False)
    print(data.x.shape)

    # [모델 정의]
    print(torch.cuda.is_available())    # GPU 사용중인지 확인
    model = Local_GCN(in_channels=data.x.shape[1], hidden1=32, hidden2=32, out_channels=3)
        # GCN AI 모델 객체(인스턴스) 생성
            # in_channels = 그래프안의 노드 특징의 갯수 (모듈과 데이터의 첫번째 배열의 입력값이 일치해야 함)
            # out_channels = 각 노드별로 0~2(3)까지의 클래스에 값이 매겨짐 (모든 노드의 가장 큰 값들을 추출해 가장 많은 값을 찾으면 됨)

    
    # [조건에 따라 모델 호출]
    model_path = "initial_gcn_model.pth"        # 저장된 모델 파일 이름
    try:
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)                                            # 저장된 모델 호출
        print(f"Loaded weights from {model_path}")                              # 해당 모델 파일에서 가중치 가져왔다고 알림
    except FileNotFoundError:
        print(f"No existing weights at {model_path}, training from scratch.")   # 해당 위치에 모델 파일이 없다고 알림


    # [추론] 
    pred = predict_fn(model, data, return_prob=False)       # 추론 수행 (모델 실행, 학습 X)
    counts = torch.bincount(pred, minlength=3).tolist()     # pred(리스트)의 각 값이 얼마나 존재하는지 리스트 형태로 정리
    final_class = counts.index(max(counts))                 # counts(리스트)의 값 중 가장 많은 값 추출 (이게 진짜 추론의 결과)
    print(final_class)


# 추론 완료 후 사용자가 추론의 결과를 받아들일지 다른 정리방법을 선택할지 값을 받아와서 label 값에 넣는다.
# 해당 학습데이터를 본래 모델에만 학습할 것인지, 연합학습으로 가중치를 전송할 것인지 선택한다.


    # [PyG 데이터 셋 생성]
    data = build_graph_from_folder(r"H:\Project_work\Roots(TestFolder)", label=2, with_label=True)           # 해당 폴더의 데이터셋 생성 (학습용)
        # 데이터 셋으로 만들 폴더 위치, 폴더 구조의 정리 방식(0~2 사이, 사용자가 선택), 학습용 데이터셋 생성(True)
    print(data.x.shape)


    # [일반 학습, 모델 저장]
    loss = train_fn(model, data)                                # 학습 시도
    
    torch.save(model.state_dict(), "initial_gcn_model.pth")     # 모델 저장
    print(f"Saved weights to {model_path} (after local train), loss={loss:.6f}")    # 해당 위치에 모델 파일 저장 및, 학습 후 가중치 알림


    # [연합 학습, 모델 저장]
        # Server와 연합학습을 위한 연결 설정
        # 클라이언트는 모든 준비를 하고 서버와 연결하여 서버의 요청을 대기한다.
        # 서버가 요청하면 클라이언트는 학습하고 그 결과인 가중치를 서버에 반환한다.
    fl.client.start_client(
        server_address="localhost:8080",
        client=GCNClient(model, train_fn, data)
    )

    torch.save(model.state_dict(), "initial_gcn_model.pth")             # 모델 저장


# 여긴 그냥 참고용 주석

        # model = 학습할 GCN 모델 객체(인스턴스)
        # data = PyG 데이터 객체 (학습 데이터)
        # train_fn = 모델과 데이터를 이용해 실제로 로컬 학습을 수행하는 함수
        # 클라이언트가 가진 데이터의 양 (샘플 수)

    print ("이건 테스트 문구")