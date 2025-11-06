# main.py (v2 - 고도화된 레이아웃 적용)
import os, sys, socket, shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print("[DEBUG] PROJECT_ROOT =", PROJECT_ROOT)

import torch
import flwr as fl

import Client
from GUI_work2 import ImageGroupDialog # v2 코드를 사용해야 합니다.

from PyQt6.QtCore import Qt, QProcess, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget,
    QPlainTextEdit, QMessageBox, QCheckBox, QDialog, QComboBox, 
    QDialogButtonBox, QFormLayout, QFrame, QSizePolicy
)

MODEL_PATH = (BASE_DIR / "Client" / "initial_gcn_model.pth").resolve()
PYTHON_BIN = sys.executable

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GCN Folder Assistant")
        self.resize(1000, 700) # 창 크기 확대

        self.model = None
        self.selected_dir: Path = None
        self.last_infer_major = None
        self.dataset = None
        self.infer_done = False

        # ---------- 1. 메인 레이아웃 (사이드바 + 컨텐츠) ----------
        root = QWidget()
        self.setCentralWidget(root)
        
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0) # 여백 없음
        main_layout.setSpacing(0)

        # ---------- 2. 좌측 사이드바 ----------
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar") # QSS용 ID
        sidebar.setFixedWidth(280)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(10)

        # 사이드바 - 제목
        title = QLabel("Folder Treatment")
        title.setObjectName("titleLabel") # QSS용 ID
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 사이드바 - 폴더 선택 그룹
        folder_group = QWidget()
        folder_layout = QVBoxLayout(folder_group)
        folder_layout.setContentsMargins(0,0,0,0)
        folder_layout.addWidget(QLabel("선택된 폴더:"))
        
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("폴더를 선택해주세요...")
        folder_layout.addWidget(self.path_edit)
        
        btn_pick = QPushButton("폴더 찾아보기")
        folder_layout.addWidget(btn_pick)

        # 사이드바 - 연합 학습
        self.chk_fed = QCheckBox("연합학습 참여")
        
        # 사이드바 - 최종 라벨
        self.lbl_final_label = QLabel("최종 라벨: (미정)")
        self.lbl_final_label.setObjectName("statusLabel") # QSS용 ID
        self.lbl_final_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_final_label.setWordWrap(True)

        # 사이드바 레이아웃에 추가
        sidebar_layout.addWidget(title)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(folder_group)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(self.chk_fed)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.lbl_final_label)
        sidebar_layout.addStretch(1) # 나머지 공간 밀어내기

        # ---------- 3. 우측 컨텐츠 영역 (탭) ----------
        content_area = QFrame()
        content_area.setObjectName("ContentArea") # QSS용 ID
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # 탭 위젯 생성
        self.tabs = QTabWidget()
        self.tabs.setObjectName("TabWidget") # QSS용 ID
        content_layout.addWidget(self.tabs)

        # 탭 1: 분석 및 학습
        tab_analyze = QWidget()
        tab_analyze_layout = QVBoxLayout(tab_analyze)
        tab_analyze_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        tab_analyze_layout.setContentsMargins(20, 20, 20, 20)
        
        analyze_title = QLabel("1. 분석 및 학습")
        analyze_title.setObjectName("tabTitle") # QSS용 ID
        self.btn_infer = QPushButton("추론 실행 (폴더 분석)")
        self.btn_train = QPushButton("학습 (선택된 라벨로)")
        
        tab_analyze_layout.addWidget(analyze_title)
        tab_analyze_layout.addSpacing(15)
        tab_analyze_layout.addWidget(self.btn_infer)
        tab_analyze_layout.addSpacing(10)
        tab_analyze_layout.addWidget(self.btn_train)
        tab_analyze_layout.addStretch(1)

        # 탭 2: 중복 파일 정리
        tab_dedupe = QWidget()
        tab_dedupe_layout = QVBoxLayout(tab_dedupe)
        tab_dedupe_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        tab_dedupe_layout.setContentsMargins(20, 20, 20, 20)

        dedupe1_title = QLabel("1. 중복 파일 정리")
        dedupe1_title.setObjectName("tabTitle")
        self.btn_next = QPushButton("중복/유사 이미지 정리 시작")
        self.btn_next.setObjectName("SpecialButton") # QSS용 ID (특별 버튼)
        
        dedupe2_title = QLabel("2. 파일 구조 정리")
        dedupe2_title.setObjectName("tabTitle")
        self.btn_last = QPushButton("라벨대로 파일 정리 시작")
        self.btn_last.setObjectName("SpecialButton") # QSS용 ID (특별 버튼)
        
        tab_dedupe_layout.addWidget(dedupe1_title)
        tab_dedupe_layout.addSpacing(15)
        tab_dedupe_layout.addWidget(self.btn_next)
        tab_dedupe_layout.addStretch(1)

        tab_dedupe_layout.addWidget(dedupe2_title)
        tab_dedupe_layout.addSpacing(15)
        tab_dedupe_layout.addWidget(self.btn_last)
        tab_dedupe_layout.addStretch(1)


        # 탭 3: 시스템 로그
        tab_log = QWidget()
        tab_log_layout = QVBoxLayout(tab_log)
        tab_log_layout.setContentsMargins(10, 10, 10, 10)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        tab_log_layout.addWidget(self.log)

        # 탭 위젯에 탭 추가
        self.tabs.addTab(tab_analyze, "  1. 분석/학습  ")
        self.tabs.addTab(tab_dedupe, "  2. 파일 정리  ")
        self.tabs.addTab(tab_log, "  3. 시스템 로그  ")

        # ---------- 4. 전체 레이아웃 적용 ----------
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_area, 1) # 1 = 남은 공간 모두 차지

        # ---------- 5. 스타일시트 로드 ----------
        self._load_stylesheet()
        
        # ---------- 6. 시그널 연결 ----------
        btn_pick.clicked.connect(self.pick_folder)
        self.btn_infer.clicked.connect(self.run_inference)
        self.btn_train.clicked.connect(self.run_training)
        self.btn_next.clicked.connect(self.open_dedupe_window)
        self.btn_last.clicked.connect(self.organize_files)
        # self.chk_fed.toggled.connect(self._update_buttons) # 로직이 바뀌었으므로 일단 보류

        self._update_buttons()

    # ---- 스타일시트 로더 ----
    def _load_stylesheet(self):
        qss_file_path = BASE_DIR / "style.qss"
        try:
            with open(qss_file_path, "r", encoding="utf-8") as f:
                style_sheet_content = f.read()
            self.setStyleSheet(style_sheet_content)
            self._log_print(f"[INFO] Loaded stylesheet from {qss_file_path}")
        except FileNotFoundError:
            self._log_print(f"[WARN] Stylesheet file not found at {qss_file_path}")
        except Exception as e:
            self._log_print(f"[ERROR] Failed to load stylesheet: {e}")

    # ---- 유틸 함수 (기존과 동일/유사) ----
    def _log_print(self, s: str):
        self.log.appendPlainText(s)
        self.tabs.setCurrentWidget(self.tabs.widget(2)) # 로그 발생 시 로그 탭으로 자동 전환

    def _update_buttons(self):
        has_dir = self.selected_dir is not None
        
        self.btn_infer.setEnabled(has_dir)
        self.btn_train.setEnabled(has_dir)
        self.btn_next.setEnabled(has_dir and self.infer_done)
        self.btn_last.setEnabled(has_dir and self.infer_done)

        # 탭 활성화/비활성화
        self.tabs.setTabEnabled(0, has_dir) # 분석/학습
        self.tabs.setTabEnabled(1, has_dir and self.infer_done) # 파일 정리
        
        if not has_dir:
            self.tabs.setCurrentWidget(self.tabs.widget(2)) # 로그 탭으로
            self.lbl_final_label.setText("최종 라벨: (폴더 선택 대기)")
        if has_dir and not self.infer_done:
            self.tabs.setCurrentWidget(self.tabs.widget(0)) # 분석 탭으로
            
    def _guard_dir(self) -> bool:
        if not self.selected_dir:
            QMessageBox.warning(self, "경고", "먼저 폴더를 선택하세요.")
            return False
        return True

    def _build_data(self, with_label: bool, label: int = 0):
        if not self.selected_dir:
            raise RuntimeError("폴더가 선택되지 않았습니다.")
        if self.dataset is None:
            self.dataset = Client.PyG_Dataset(str(self.selected_dir))
        return self.dataset.build_graph_from_folder(label=label, with_label=with_label)

    def _ensure_model(self, in_channels: int, out_channels: int = 3):
        if getattr(self, "model", None) is None:
            self._log_print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
            self.model = Client.Local_GCN(in_channels=in_channels, hidden1=32, hidden2=32, out_channels=out_channels,)
            if MODEL_PATH.exists():
                try:
                    Client.load_model(self.model, str(MODEL_PATH), map_location="cpu")
                    self._log_print(f"[INFO] Loaded weights from {MODEL_PATH}")
                except Exception as e:
                    self._log_print(f"[WARN] Failed to load weights: {e}")
            else:
                self._log_print("[INFO] No existing weights, starting fresh")
        return self.model

    def _majority_from_pred(self, pred, num_classes: int):
        counts = torch.bincount(pred, minlength=num_classes).tolist()
        return counts.index(max(counts))

    def _is_port_open(self, host: str, port: int, timeout: float = 1.5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _collect_files(self, only_existing: bool = True):   # 데이터셋에서 파일만 수집
        G = self.dataset.G
        out = []
        for i in G.nodes:
            try:
                x = G.nodes[i]["x"]  # tensor([is_file, size_kb, depth, mtime])

                if int(x[0].item()) != 1:   # 폴더 파일 구분값에서 0(폴더)라면
                    continue                # 해당 try 구문을 진행하지 않고 넘김

                path = self.dataset.idx_to_path[i]      # 해당(index에 해당하는) 파일의 전체 경로
                name = G.nodes[i].get("name", os.path.basename(path))   # 해당 파일 이름
                ext  = G.nodes[i].get("ext", os.path.splitext(name)[1].lower()) # 해당 파일의 확장자
                stem = os.path.splitext(name)[0]    # 해당 파일의 순수 이름(확장제 제거)
                mtime = float(x[3].item())  # 가장 최근 수정시간

                exists = os.path.exists(path)   # 해당 위치의 파일이 현재 존재하는지 확인 (파일이 있다면 True)
                rec = {"idx": i, "path": path, "name": name, "stem": stem,
                    "ext": ext, "mtime": mtime, "exists": exists}
                if (not only_existing) or exists:   # 해당 위치에 파일이 있거나, 없더라도only_existing이 True로 되어있다면 상관없이 모두 리스트에 포함
                    out.append(rec)                 # 파일이 있다면 해당파일의 딕셔너리를 리스트에 추가
            except Exception:
                continue
        return out          # 리스트 반환

    # ---------- 핸들러 함수 (기존과 동일) ----------

    def pick_folder(self):      
        d = QFileDialog.getExistingDirectory(self, "분석할 폴더 선택", str(Path.home()))
        if d:
            self.selected_dir = Path(d)
            self.path_edit.setText(str(self.selected_dir))
            self._log_print(f"[선택 폴더] {self.selected_dir}")
            self.dataset = Client.PyG_Dataset(str(self.selected_dir))
            self.infer_done = False # 새 폴더 선택 시 추론 상태 초기화
            self.lbl_final_label.setText("최종 라벨: (미정)")
        self._update_buttons()

    def run_inference(self):
        if not self._guard_dir(): return
        try:
            self._log_print("[Infer] 추론용 데이터 빌드 시작...")
            data = self._build_data(with_label=False)
            self._log_print(f"[Infer] data.x shape = {tuple(data.x.shape)}")
            model = self._ensure_model(in_channels=data.x.shape[1], out_channels=3)
            
            self._log_print("[Infer] 추론 실행...")
            pred = Client.predict_fn(model, data, return_prob=False)
            maj = self._majority_from_pred(pred, num_classes=3)
            self.last_infer_major = maj

            counts = torch.bincount(pred, minlength=3).tolist()
            self._log_print(f"[Infer] class counts = {counts}")
            self._log_print(f"[Infer] majority class = {maj}")

        except Exception as e:
            QMessageBox.critical(self, "추론 오류", str(e))
            self._log_print(f"[ERROR] Inference failed: {e}")
            return

        class_names = ["확장자 분류", "마지막 수정 기준", "동일 이름 구분"] # 이름 예시
        dlg = InferenceReviewDialog(counts, maj, class_names, self)
        res = dlg.exec()

        if res == QDialog.DialogCode.Accepted:
            chosen = dlg.selected_label()
            self.last_infer_major = chosen
            self._log_print(f"[Infer] user-selected label = {chosen} ({class_names[chosen]})")
        else:
            self.last_infer_major = maj
            self._log_print(f"[Infer] (사용자 취소) majority used = {maj} ({class_names[maj]})")
        
        self.lbl_final_label.setText(f"최종 라벨: {self.last_infer_major} ({class_names[self.last_infer_major]})")
        self.infer_done = True
        self._update_buttons()
        self.tabs.setCurrentWidget(self.tabs.widget(1)) # 정리 탭으로 자동 이동

    def run_training(self):
        if not self._guard_dir(): return
        label = self.last_infer_major if self.last_infer_major is not None else 0
        try:
            data = self._build_data(with_label=True, label=label)
            self._log_print(f"[Train] data.x shape = {tuple(data.x.shape)}, label={label}")
            model = self._ensure_model(in_channels=data.x.shape[1], out_channels=3)

            if not self.chk_fed.isChecked():
                self._log_print("[Train] 로컬 학습 시작...")
                loss = Client.train_fn(model, data)
                Client.save_model(model, str(MODEL_PATH))  
                self._log_print(f"[Local Train] done, loss={loss:.6f}")
                self._log_print(f"[Model] Saved to {MODEL_PATH}")
                QMessageBox.information(self, "로컬 학습 완료", f"로컬 학습 완료!\nLoss: {loss:.6f}")
            else:
                host, port = "localhost", 8080
                if not self._is_port_open(host, port):
                    QMessageBox.warning(self, "연합학습 서버 필요", f"서버({host}:{port})가 꺼져 있습니다.")
                    return

                self.btn_train.setEnabled(False)
                self._log_print("[FL] Flower 클라이언트 시작...")
                self.fl_thread = FL_Client("localhost:8080", self.model, Client.train_fn, data, self)
                self.fl_thread.finished_ok.connect(self._on_fl_done)
                self.fl_thread.failed.connect(self._on_fl_failed)
                self.fl_thread.finished.connect(lambda: self.btn_train.setEnabled(True))
                self.fl_thread.finished_ok.connect(lambda: self._log_print("[FL] client finished"))
                self.fl_thread.failed.connect(lambda msg: self._log_print(f"[FL][ERROR] {msg}"))
                self.fl_thread.start() 

        except Exception as e:
            QMessageBox.critical(self, "학습 오류", str(e))
            self._log_print(f"[ERROR] Training failed: {e}")

    def _on_fl_done(self):
        Client.save_model(self.model, str(MODEL_PATH))
        self._log_print(f"[FL] finished. Saved final weights to {MODEL_PATH}")
        QMessageBox.information(self, "연합 학습 완료", "연합 학습 1 라운드 완료!")

    def _on_fl_failed(self, msg: str):
        QMessageBox.critical(self, "연합학습 실패", msg)
        self._log_print(f"[FL][ERROR] {msg}")

    def open_dedupe_window(self):
        if not self._guard_dir(): return
        try:
            if self.dataset is None:
                self.dataset = Client.PyG_Dataset(str(self.selected_dir))
            if getattr(self.dataset, "G", None) is None or self.dataset.G.number_of_nodes() == 0:
                self._log_print("[Dedupe] 분석용 데이터셋 빌드...")
                self.dataset.build_graph_from_folder(with_label=False)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 분석 실패: {e}")
            return
        
        self._log_print("[Dedupe] 중복 분석 창 표시...")
        dlg = ImageGroupDialog(self.dataset, parent=self)
        dlg.exec()


    def organize_files(self):
        label_class = self.last_infer_major
        file_data = self._collect_files()

        if not file_data:
            self._log_print("[INFO] 정리할 파일이 없습니다.")
            return
        
        for f in file_data[:10]:  # 상위 10개만 확인
            print("mtime =", f.get("mtime"), "type =", type(f.get("mtime")))

        match label_class:
            case 0:
                self._log_print("[정리 방식] 확장자 기준 정리 수행")
                base_dir = Path(self.selected_dir)
                target_root = base_dir / "확장자 기준 정리"
                target_root.mkdir(exist_ok=True)

                # 확장자별 파일 분류 (딕셔너리속 리스트에 해당 파일의 전체경로 저장)
                ext_groups = {}
                for file in file_data:
                    ext = file["ext"].lower() if file["ext"] else "_NOEXT"  # ext가 있다면 해당 값을, ext가 비어있다면("") _NOEXT 값을 저장
                    path = file["path"]
                    if ext not in ext_groups:
                        ext_groups[ext] = []
                    ext_groups[ext].append(path)

                MIN_COUNT = 3
                misc_folder = target_root / "그 외"
                misc_folder.mkdir(parents=True, exist_ok=True)
                for exts, files in ext_groups.items():
                    # 확장자별 폴더 경로
                    folder_name = exts[1:].upper() if exts.startswith(".") else exts.upper()   # 확장자라면 폴더이름은 앞의 점을 제거한 확장자 문구, 나머지는 무시
                    target_dir = target_root / folder_name

                    if len(files) < MIN_COUNT:      # 3개 미만인 경우,
                        target_dir = misc_folder        # 그외에 해당하는 경우 폴더 경로 변경           
                    target_dir.mkdir(exist_ok=True)  # 확장자 폴더 생성 (이미 있다면 무시)

                    # 파일 복사
                    for file_path in files:
                        src = Path(file_path)   # 문자열 형태의 폴더 경로를 Path 객체로 변경 (여러 연산 활용 가능)
                        if not src.exists():    # 파일 존재 여부 확인
                            continue  # 이미 삭제된 파일은 건너뜀
                        try:
                            shutil.copy2(str(src), str(target_dir / src.name))      # 복사할 파일(src), 복사될 위치()
                            self._log_print(f"[COPY] {src.name} -> {target_dir}")   # 로그에 기록
                        except Exception as e:
                            self._log_print(f"[WARN] 복사 실패: {src} ({e})")         # 로그에 기록
                    
            case 1:
                self._log_print("[정리 방식] 마지막 수정 날짜 기준 정리 수행")
                base_dir = Path(self.selected_dir)
                target_root = base_dir / "수정날짜 기준 정리"
                target_root.mkdir(exist_ok=True)

                # 수정날짜별 파일 분류 (딕셔너리속 리스트에 해당 파일의 전체경로 저장)
                mtime_groups = {}
                for file in file_data:
                    mtime = float(file["mtime"])     # 파일의 mtime값을 저장
                    path = file["path"]             
                    try:
                        key = datetime.fromtimestamp(mtime).strftime("%Y-%m") if mtime > 0 else "_UNKNOWN"
                    except Exception:
                        key = "_UNKNOWN"
                    if key not in mtime_groups:
                        mtime_groups[key] = []      # 딕셔너리에 해당 키 생성
                    mtime_groups[key].append(path)

                for keys, files in mtime_groups.items():
                    # 수정시간별 폴더 경로
                    target_dir = target_root / keys
                    target_dir.mkdir(exist_ok=True)

                    # 파일 복사
                    for file_path in files:
                        src = Path(file_path)   # 문자열 형태의 폴더 경로를 Path 객체로 변경 (여러 연산 활용 가능)
                        if not src.exists():    # 파일 존재 여부 확인
                            continue  # 이미 삭제된 파일은 건너뜀
                        try:
                            shutil.copy2(str(src), str(target_dir / src.name))      # 복사할 파일(src), 복사될 위치()
                            self._log_print(f"[COPY] {src.name} -> {target_dir}")   # 로그에 기록
                        except Exception as e:
                            self._log_print(f"[WARN] 복사 실패: {src} ({e})")         # 로그에 기록

            case 2:
                self._log_print("[정리 방식] 동일 이름 파일 그룹화 정리 수행")
                base_dir = Path(self.selected_dir)
                target_root = base_dir / "동일한 파일이름 기준 정리"
                target_root.mkdir(exist_ok=True)

                # 동일한 파일이름별 파일 분류 (딕셔너리속 리스트에 해당 파일의 전체경로 저장)
                samefile_groups = {}
                for file in file_data:
                    path = Path(file["path"])     # 파일의 전체경로 값을 저장
                    if not path.exists():
                        continue
                    
                    name = path.stem              # 전체파일경로 -> 확장자를 제거한 파일이름
                    key = name[:5] if len(name) >= 5 else name  # 앞 5글자(5글자 미만은 그대로)

                    if key not in samefile_groups:
                        samefile_groups[key] = []      # 딕셔너리에 해당 키 생성
                    samefile_groups[key].append(path)

                for key, files in samefile_groups.items():
                    # 파일 복사
                    for file_path in files:
                        src = Path(file_path)   # 문자열 형태의 폴더 경로를 Path 객체로 변경 (여러 연산 활용 가능)
                        if not src.exists():    # 파일 존재 여부 확인
                            continue  # 이미 삭제된 파일은 건너뜀
                        try:
                            parent_name = src.parent.name       # 부모 폴더 이름 추출
                            stem, suf = src.stem, src.suffix    # 파일 이름과 확장자 추출

                            if len(files) == 1:
                                # 원래 폴더 구조 유지: target_root 기준 상대경로
                                rel_path = src.parent.relative_to(base_dir) 
                                    # 파일의 부모폴더가 base_dir 하위에 있을 경우 base_dir 이후의 경로만 반환
                                        # (base_dir을 기준으로 해당 파일의 상대 경로 반환)
                                dst_dir = target_root / rel_path
                            else:
                                dst_dir = target_root / key

                            dst_dir.mkdir(parents=True, exist_ok=True)  # 폴더 생성

                            new_name = f"{stem}--({parent_name}){suf}"  # "파일이름--(부모폴더).확장자" 구조의 새 파일이름 생성
                            shutil.copy2(str(src), str(dst_dir / new_name))      # 복사할 파일(src), 복사될 위치()
                            self._log_print(f"[COPY] {src.name} -> {dst_dir}")    # 로그에 기록
                        except Exception as e:
                            self._log_print(f"[WARN] 복사 실패: {src} ({e})")       # 로그에 기록

        os.startfile(target_root)       # 정리된 폴더 파일탐색기에 열기
        QApplication.quit()             # 프로그램 종료



# (FL_Client 스레드 클래스와 InferenceReviewDialog 클래스는 기존과 동일하게 유지)
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
            client = Client.GCNClient(self.model, self.train_fn, self.data)
            fl.client.start_client(server_address=self.server_addr, client=client)
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))

class InferenceReviewDialog(QDialog):
    def __init__(self, counts, majority_idx, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("추론 결과 확인")
        self.setModal(True)
        info = QLabel(f"모델 추론 요약 (다수결: {majority_idx}번)\n- Counts: {counts}")
        self.combo = QComboBox(self)
        for i, name in enumerate(class_names):
            self.combo.addItem(name, i) # userData로 인덱스 저장
        self.combo.setCurrentIndex(majority_idx)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QFormLayout(self)
        lay.addRow(info)
        lay.addRow("최종 라벨 선택:", self.combo)
        lay.addRow(buttons)
    def selected_label(self) -> int:
        return self.combo.currentData() # userData에서 인덱스 반환

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())