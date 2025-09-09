import threading
import time
import os
import flwr as fl
from flwr.server import start_server
from flwr.server.strategy import FedAvg
from GCN_Server import make_strategy, make_config

# ... (메트릭 집계 함수 등은 그대로)

def stop_server_after(delay_sec=120):  # 예시: 2분(120초)ㅁ
    time.sleep(delay_sec)
    print("\n[Flower 서버 자동 종료] {}초 경과".format(delay_sec))
    # os._exit(0)  # 강제종료(아주 안전하게 끊김)
    # 또는
    import signal
    os.kill(os.getpid(), signal.SIGINT)  # Ctrl+C 신호 보내기

if __name__ == "__main__":
    # 종료 타이머 스레드 시작

    print("SERVER_READY", flush=True)


    t = threading.Thread(target=stop_server_after, args=(300,), daemon=True)  # 300초=5분, daemon=True는 메인 프로세스 죽으면 같이 죽음
    t.start()

    # 서버 실행
    try:
        start_server(
            server_address="localhost:8080",
            config=make_config(),
            strategy=make_strategy(),
        )
    except Exception as e:
        print(f"[SERVER ERROR] {e}", file=sys.stderr, flush=True)
        raise    
