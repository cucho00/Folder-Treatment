# server.py
import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

# 서버가 가지고 있는 모델의 가중치를 클라이언트에게 제공
# 클라이언트는 전달받은 가중치를 자신이 가진 모델에 적용
# 이후, 클라이언트가 학습을 진행하면 진행한 모델의 가중치를 서버에 다시 전달
# 서버는 여러 클라이언트가 보내준 가중치를 평균화 하여 새 모델 생성
# 여기까지가 한 라운드 (이를 여러번 반복할 횟수 설정)


# 학습 메트릭 집계 함수 (여기서는 평균값을 계산)
def fit_metrics_aggregation_fn(metrics):
    # metrics: List[Tuple[num_examples, metrics_dict]]
    # ex: [(20, {"loss": 0.32}), (17, {"loss": 0.21}), ...]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"loss": 0.0}
    loss = sum(num_examples * m["loss"] for num_examples, m in metrics) / total_examples
    return {"loss": loss}

# 평가 메트릭 집계 함수 (여기서는 평균값을 계산)
def evaluate_metrics_aggregation_fn(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    return {"accuracy": accuracy}


def make_strategy():
    return FedAvg(
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=0,   # 평가 비활성 권장
        fraction_fit=1.0,
        fraction_evaluate=0.0,    # 평가 요청 안 함
        evaluate_fn=None,         # 서버 글로벌 평가 끔
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

def make_config():
    return ServerConfig(
        num_rounds=1,      # 테스트는 짧게
        round_timeout=120, # 영구 대기 방지
    )


"""

# 클라이언트에서 전달된 가중치 평균을 내서 서버 모델 업데이트 (기본 전략)
strategy = FedAvg(              
    #fraction_fit=0.5,           # 전체 클라이언트 중 50%만 학습 참여
    #fraction_eval=0.5,          # 평가에 참여할 클라이언트 비율

    min_fit_clients=1,           # 최소 2명 이상은 학습에 참여해야 함
    min_available_clients=1,     # 최소 2명 이상 클라이언트가 연결되어 있어야 함
    min_evaluate_clients=1,
    
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # 평가 메트릭 집계
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn  # 학습 메트릭 집계
)

start_server(
    server_address="localhost:8080",                
        # 서버 주소 및 포트번호 (클라이언트에서 동일하게 사용해야 함)
    config=fl.server.ServerConfig(num_rounds=5),    
        # 총 연합 학습 라운드 수
    strategy=strategy
        # 이 설정으로 클라이언트 가중치 수집 방식, 서버 업데이트 방식, 평가 시점 등을 결정
)


# 서버는 자체 모델을 보유하지 않음
# GlobalModel은 실제 모델이 아니라 
# 학습된 가중치들의 값으로 만들어낸 평균화된 가중치 값을 의미


if __name__ == "__main__":
    start_server()
"""