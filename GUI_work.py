# main.py
import os, sys, socket
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # 지금 실행중인 파이선 소스 파일의 폴더 경로(절대 경로)
PROJECT_ROOT = BASE_DIR                         # GUI_work.py가 이미 루트에 있으니까 그냥 BASE_DIR이 곧 루트
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print("[DEBUG] PROJECT_ROOT =", PROJECT_ROOT)

import torch
import flwr as fl

import Client
from GUI_work2 import ImageGroupDialog
"""
from Client.Folder_Tree import build_graph_from_folder
from Client.Local_GCN_Module import Local_GCN, train_fn, predict_fn, save_model, load_model
from Client.GCN_Client_before import GCNClient  
"""

from PyQt5.QtCore import Qt, QProcess, QByteArray, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget,
    QPlainTextEdit, QMessageBox, QCheckBox, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QDialog, QComboBox, QDialogButtonBox, QFormLayout
)

#MODEL_PATH    = BASE_DIR / ".." / "Client" / "initial_gcn_model.pth"
#CLIENT_SCRIPT = BASE_DIR / ".." / "Client" / "GCN_Client.py"
#SERVER_SCRIPT = BASE_DIR / ".." / "Server" / "GCN_Server_Run.py"
MODEL_PATH = (BASE_DIR / "Client" / "initial_gcn_model.pth").resolve() 
CLIENT_SCRIPT = (BASE_DIR / "Client" / "GCN_Client.py").resolve() 
SERVER_SCRIPT = (BASE_DIR / "Server" / "GCN_Server_Run.py").resolve()
PYTHON_BIN = sys.executable


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GCN Folder Assistant (PyQt5)")
        self.resize(1000, 640)

        self.model = None
        self.server_ready = False
        self.selected_dir: Path = None
        self.lbl_final_label = QLabel("최종 라벨: (미정)")
        self.server_proc: QProcess = None
        self.last_infer_major = None        # 최근 추론 다수결 결과(학습 라벨 기본값)

        self.dataset = None                 # PyG 데이터셋 클래스 객체
        self.PyG_data = None                # PyG 데이터 

        # ---------- 메인 레이아웃 ----------
        root = QWidget()
        root_l = QVBoxLayout(root)
        self.setCentralWidget(root)

        # ===== Title =====
        title = QLabel("이미지 유사도 분석기")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        # ---------- 상단 바 ----------
        top = QWidget()
        top_l = QHBoxLayout(top)
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        btn_pick   = QPushButton("폴더 선택")
        btn_infer  = QPushButton("추론 실행")
        btn_train  = QPushButton("학습")
        self.chk_fed = QCheckBox("연합학습 참여")
        btn_next = QPushButton("다음 (중복 이미지 정리)")
        btn_next.setEnabled(False)  # 초기엔 비활성화

        top_l.addWidget(QLabel("선택 폴더:"))
        top_l.addWidget(self.path_edit, 1)
        top_l.addWidget(btn_pick)
        top_l.addWidget(btn_infer)
        top_l.addWidget(btn_train)
        top_l.addWidget(self.chk_fed)
        top_l.addWidget(btn_next)
        top_l.addWidget(self.lbl_final_label, stretch=1)

        # ---------- 로그 ----------
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        # ---------- 전체 적용 ----------- 
        root_l.addWidget(title)
        root_l.addWidget(top)
        root_l.addWidget(self.log, 1)


        # 보관(활성/비활성 제어용)
        self.btn_infer = btn_infer
        self.btn_train = btn_train
        self.btn_next = btn_next

        self.infer_done = False  # 추론 완료 여부 플래그

        # ---------- 시그널 ----------          (버튼과 실행되는 함수를 연결하는 매핑 과정)
        btn_pick.clicked.connect(self.pick_folder)
        btn_infer.clicked.connect(self.run_inference)
        btn_train.clicked.connect(self.run_training)
        self.chk_fed.toggled.connect(self._update_buttons)
        btn_next.clicked.connect(self.open_dedupe_window)

        self._update_buttons()

    # ---- 유틸 ----           (여러 함수에서 반복해서 사용되는 편의 함수를 따로 정의)
    def _log_print(self, s: str):                                        # 로그 생성
        self.log.appendPlainText(s)

    def _update_buttons(self):                                           # 버튼 활성화
        has_dir = self.selected_dir is not None
        
        # 추론과 (로컬/연합)학습 버튼은 폴더만 선택하면 활성화
        self.btn_infer.setEnabled(has_dir)      
        self.btn_train.setEnabled(has_dir)
            
        # "다음" 버튼은 폴더가 있고, 추론까지 끝났을 때만 활성화
        self.btn_next.setEnabled(has_dir and getattr(self, "infer_done", False))

    def _guard_dir(self) -> bool:                                       # 폴더가 선택된 상태인지 확인
        if not self.selected_dir:
            QMessageBox.warning(self, "경고", "먼저 폴더를 선택하세요.")
            return False
        return True

    def _build_data(self, with_label: bool, label: int = 0):            # 선택된 폴더 → PyG 데이터셋 변환
        if not self.selected_dir:
            raise RuntimeError("폴더가 선택되지 않았습니다.")
        if self.dataset is None:
            self.dataset = Client.PyG_Dataset(str(self.selected_dir))   # dataset 클래스 객체가 없다면 생성
        return self.dataset.build_graph_from_folder(label=label, with_label=with_label)     # dataset 클래스 객체를 통해 PyG 데이터셋 생성


    def _ensure_model(self, in_channels: int, out_channels: int = 3):   # 모델 생성 또는 기존 모델 로드
        if getattr(self, "model", None) is None:
            self._log_print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
            self.model = Client.Local_GCN(in_channels=in_channels, hidden1=32, hidden2=32, out_channels=out_channels,)

            # 기존 가중치 로드(있다면)
            if MODEL_PATH.exists():
                try:
                    Client.load_model(self.model, str(MODEL_PATH), map_location="cpu")
                    self._log_print(f"[INFO] Loaded weights from {MODEL_PATH}")
                except Exception as e:
                    self._log_print(f"[WARN] Failed to load weights: {e}")
            else:
                self._log_print("[INFO] No existing weights, starting fresh")
        return self.model

    def _majority_from_pred(self, pred, num_classes: int):              # 추론 결과 계산 (추론 결과 리스트에서 가장 많이 나온 값 추출)
        counts = torch.bincount(pred, minlength=num_classes).tolist()
        return counts.index(max(counts))

    def _is_port_open(self, host: str, port: int, timeout: float = 1.5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False


    # ---------- 핸들러 함수 ----------

    def pick_folder(self):      
        """ 폴더 선택 """
        d = QFileDialog.getExistingDirectory(self, "분석할 폴더 선택", str(Path.home()))
        if d:
            self.selected_dir = Path(d)
            self.path_edit.setText(str(self.selected_dir))
            self._log_print(f"[선택 폴더] {self.selected_dir}")

            self.dataset = Client.PyG_Dataset(str(self.selected_dir))       # 선택한 폴더로 PyG 데이터 셋 클래스 객체 생성

        self._update_buttons()

    def run_inference(self):
        """ 추론 실행 """
        if not self._guard_dir():
            return
        try:
            # 1) 추론용 데이터 (라벨 불필요)
            data = self._build_data(with_label=False)
            self._log_print(f"[Infer] data.x shape = {tuple(data.x.shape)}")

            # 2) 모델 생성/로드
            model = self._ensure_model(in_channels=data.x.shape[1], out_channels=3)

            # 3) 추론
            pred = Client.predict_fn(model, data, return_prob=False)       # 결과 추론 (리스트)
            maj = self._majority_from_pred(pred, num_classes=3)     # 평균 계산 (리스트에서 가장 많은 값)
            self.last_infer_major = maj

            # 4) 로그 출력
            counts = torch.bincount(pred, minlength=3).tolist()
            self._log_print(f"[Infer] class counts = {counts}")
            self._log_print(f"[Infer] majority class = {maj}")

        except Exception as e:
            QMessageBox.critical(self, "추론 오류", str(e))
            self._log_print(f"[ERROR] Inference failed: {e}")
            return                      # 에러 발생시 다이얼로그 진행하지 않도록 조기 반환  

        # 추론 성공 시, 다음 진행
        class_names = ["Class 0", "Class 1", "Class 2"]  # 원하는 이름으로 변경 가능(예: ["정리","보관","삭제"])
        dlg = InferenceReviewDialog(counts, maj, class_names, self)
        res = dlg.exec_()

        if res == QDialog.Accepted:
            chosen = dlg.selected_label()
            self.last_infer_major = chosen
            self._log_print(f"[Infer] user-selected label = {chosen} ({class_names[chosen]})")
            if hasattr(self, "lbl_final_label"):
                self.lbl_final_label.setText(f"최종 라벨: {chosen} ({class_names[chosen]})")
        else:
            # 취소 시 다수결 기본값 유지
            self.last_infer_major = maj
            self._log_print(f"[Infer] (사용자 취소) majority used = {maj} ({class_names[maj]})")
            if hasattr(self, "lbl_final_label"):
                self.lbl_final_label.setText(f"최종 라벨: {maj} ({class_names[maj]})")

        # 추론 완료 시 플래그 on + 버튼 갱신
        self.infer_done = True
        self._update_buttons()      # 버튼 상태 갱신 (다음 버튼 활성화)


    def run_training(self):
        """ 학습 수행 """
        if not self._guard_dir():
            return

        # 라벨 결정: 추론 다수결 결과가 있으면 사용, 없으면 0 기본
        label = self.last_infer_major if self.last_infer_major is not None else 0
        try:
            data = self._build_data(with_label=True, label=label)                       # 학습용 PyG 데이터셋 생성
            self._log_print(f"[Train] data.x shape = {tuple(data.x.shape)}, label={label}")

            model = self._ensure_model(in_channels=data.x.shape[1], out_channels=3)         # 모델 객체 생성

            if not self.chk_fed.isChecked():
                # -------- 로컬 학습 --------
                loss = Client.train_fn(model, data)
                Client.save_model(model, str(MODEL_PATH))  
                self._log_print(f"[Local Train] done, loss={loss:.6f}")
                self._log_print(f"[Model] Saved to {MODEL_PATH}")
            else:
                # -------- 연합학습 -------- 
                host, port = "localhost", 8080
                if not self._is_port_open(host, port):                                  # 연합학습을 위한 포트 열림 확인
                    QMessageBox.information(
                        self, "연합학습 서버 필요",
                        f"서버({host}:{port})가 꺼져 있습니다.\n"
                        f"서버 프로그램을 먼저 실행한 뒤 다시 시도해 주세요."
                    )
                    return

                self.btn_train.setEnabled(False)  # 중복 클릭 방지 (선택)
                self._log_print("[FL] Starting Flower client in background thread...")

                # 스레드 시작
                self.fl_thread = FL_Client("localhost:8080", self.model, Client.train_fn, data, self)
                self.fl_thread.finished_ok.connect(self._on_fl_done)                        # 완료 시 모델 저장/로그
                self.fl_thread.failed.connect(self._on_fl_failed)                           # 실패 시 안내
                self.fl_thread.finished.connect(lambda: self.btn_train.setEnabled(True))    # 종료 시 버튼 복구
                self.fl_thread.finished_ok.connect(lambda: self._log_print("[FL] client finished"))     # 스레드 완료 시 로그
                self.fl_thread.failed.connect(lambda msg: self._log_print(f"[FL][ERROR] {msg}"))        # 스레드 실패 시 로그
                self.fl_thread.start() 

        except Exception as e:
            QMessageBox.critical(self, "학습 오류", str(e))
            self._log_print(f"[ERROR] Training failed: {e}")

    # 스레드 시작 후 곧바로가 아니라 스레드가 정상/비정상적으로 종료되었을 때, 동작되어야 하기 때문에 이렇게 따로 함수로 구성
    def _on_fl_done(self):
        Client.save_model(self.model, str(MODEL_PATH))
        self._log_print(f"[FL] finished. Saved final weights to {MODEL_PATH}")

    def _on_fl_failed(self, msg: str):
        QMessageBox.critical(self, "연합학습 실패", msg)
        self._log_print(f"[FL][ERROR] {msg}")

    # ======================================================================
    def open_dedupe_window(self):
        if not self._guard_dir():
            return

        # PyG 그래프/메타 생성 (분석용)
        try:
            # 폴더 선택시 만든 self.dataset 객체가 없을 시, 생성
            if self.dataset is None:
                self.dataset = Client.PyG_Dataset(str(self.selected_dir))
            
            # PyG 데이터셋 미생성 시, 1회 생성
            if getattr(self.dataset, "G", None) is None or self.dataset.G.number_of_nodes() == 0:
                self.dataset.build_graph_from_folder(with_label=False)

        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 분석 실패: {e}")
            return
        
        # 분석 결과를 표시하는 다이얼로그
        dlg = ImageGroupDialog(self.dataset, parent=self)
        dlg.exec_()


class FL_Client(QThread):
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, server_addr, model, train_fn, data, parent=None):
        super().__init__(parent)
        self.server_addr = server_addr
        self.model = model
        self.train_fn = train_fn
        self.data = data

    def run(self):
        try:
            client = Client.GCNClient(self.model, self.train_fn, self.data)                # 연합학습 클라이언트 설정
            fl.client.start_client(server_address=self.server_addr, client=client)  # 연합학습 클라이언트 수행
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


""" 추론 결과 확인 및 다른 정리방식 선택 """
class InferenceReviewDialog(QDialog):
    def __init__(self, counts, majority_idx, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("추론 결과 확인")
        self.setModal(True)

        # 설명 라벨
        info = QLabel(
            "모델 추론 요약\n"
            f"- counts: {counts}\n"
            f"- majority: {majority_idx} - {class_names[majority_idx]}"
        )
    
        # 사용자 선택 콤보박스
        self.combo = QComboBox(self)
        for i, name in enumerate(class_names):
            self.combo.addItem(f"{i} - {name}", i)
        self.combo.setCurrentIndex(majority_idx)  # 기본값: 다수결

        # OK / Cancel 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # 레이아웃 (창에서의 위젯 배치 구조)
        lay = QFormLayout(self)
        lay.addRow(info)
        lay.addRow("최종 라벨 선택:", self.combo)
        lay.addRow(buttons)

    def selected_label(self) -> int:
        return self.combo.currentData()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
