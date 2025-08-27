# main.py
import os, sys, json, random
from pathlib import Path
from PyQt5.QtCore import Qt, QProcess, QByteArray
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget,
    QPlainTextEdit, QMessageBox, QCheckBox, QTreeWidget, QTreeWidgetItem,
    QGroupBox
)

BASE_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = BASE_DIR / ".." / "Server" / "GCN_Server_Run.py"
CLIENT_SCRIPT = BASE_DIR / ".." / "Client" / "GCN_Client.py"
PYTHON_BIN = sys.executable

def default_result_json(selected_dir: Path) -> Path:
    return selected_dir / "gcn_output" / "results.json"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GCN Folder Assistant (PyQt5)")
        self.resize(1100, 750)

        self.selected_dir: Path = None
        self.server_ready = False
        self.server_proc: QProcess = None
        self.client_proc: QProcess = None
        self.current_seed = random.randint(0, 1_000_000)
        self.result_json_path: Path = None

        # --- 상단 바 ---
        top = QWidget(); top_l = QHBoxLayout(top)
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        self.chk_upload = QCheckBox("데이터셋을 서버에 전송")

        btn_pick   = QPushButton("폴더 선택")
        btn_server = QPushButton("서버 시작")
        btn_client = QPushButton("클라이언트 실행")
        btn_reroll = QPushButton("다른 결과")
        btn_apply  = QPushButton("적용")

        top_l.addWidget(QLabel("선택 폴더:"))
        top_l.addWidget(self.path_edit, 1)
        top_l.addWidget(self.chk_upload)
        top_l.addWidget(btn_pick)
        top_l.addWidget(btn_server)
        top_l.addWidget(btn_client)
        top_l.addWidget(btn_reroll)
        top_l.addWidget(btn_apply)

        # --- 결과 탭(추천안 미리보기) ---
        self.results_tabs = QTabWidget()
        self._show_placeholder()

        # --- 로그 ---
        logs = QGroupBox("로그"); logs_l = QHBoxLayout(logs)
        self.server_log = QPlainTextEdit(); self.server_log.setReadOnly(True)
        self.client_log = QPlainTextEdit(); self.client_log.setReadOnly(True)
        self.server_log.setPlaceholderText("서버 로그")
        self.client_log.setPlaceholderText("클라이언트 로그")
        logs_l.addWidget(self.server_log, 1)
        logs_l.addWidget(self.client_log, 1)

        # --- 메인 레이아웃 ---
        central = QWidget(); root = QVBoxLayout(central)
        root.addWidget(top)
        root.addWidget(self.results_tabs, 3)
        root.addWidget(logs, 2)
        self.setCentralWidget(central)

        # --- 시그널 ---
        btn_pick.clicked.connect(self.pick_folder)
        btn_server.clicked.connect(self.start_server)
        btn_client.clicked.connect(self.run_client_once)
        btn_reroll.clicked.connect(self.reroll_results)
        btn_apply.clicked.connect(self.apply_current_result)

        self._update_buttons()

    # ---------- 유틸 ----------
    def _show_placeholder(self):
        while self.results_tabs.count(): self.results_tabs.removeTab(0)
        ph = QPlainTextEdit(); ph.setReadOnly(True)
        ph.setPlainText("아직 결과가 없습니다.\n[클라이언트 실행]을 눌러 결과를 생성하세요.")
        self.results_tabs.addTab(ph, "결과 없음")

    def _update_buttons(self):
        has_dir = self.selected_dir is not None
        can_client = self.server_ready and has_dir

        for w in self.findChildren(QPushButton):
            t = w.text()
            if t in ("클라이언트 실행", "다른 결과"):
                w.setEnabled(can_client)
            elif t == "적용":
                w.setEnabled(can_client and self.results_tabs.count() > 0 and
                            self.results_tabs.tabText(0) != "결과 없음")
            elif t == "서버 시작":
                w.setEnabled(not self.server_ready)
            elif t == "폴더 선택":
                w.setEnabled(True)

    def _log_srv(self, s): 
        if s: self.server_log.appendPlainText(s.rstrip("\n"))
    def _log_cli(self, s): 
        if s: self.client_log.appendPlainText(s.rstrip("\n"))

    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "분석할 폴더 선택", str(Path.home()))
        if d:
            self.selected_dir = Path(d)
            self.path_edit.setText(str(self.selected_dir))
            self._log_cli(f"[선택 폴더] {self.selected_dir}")
        self._update_buttons()

    # ---------- 서버 ----------
    def start_server(self):
        if not SERVER_SCRIPT.exists():
            QMessageBox.critical(self, "오류", f"서버 스크립트를 찾을 수 없습니다:\n{SERVER_SCRIPT}")
            return
        if self.server_proc and self.server_proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "서버", "이미 실행 중입니다.")
            return

        self.server_proc = QProcess(self)
        self.server_proc.setProgram(PYTHON_BIN)
        self.server_proc.setArguments([str(SERVER_SCRIPT)])
        self.server_proc.setWorkingDirectory(str(SERVER_SCRIPT.parent))
        self.server_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.server_proc.readyReadStandardOutput.connect(self._on_srv_out)
        self.server_proc.finished.connect(self._on_srv_fin)

        self._log_srv(f">>> 서버 시작: {PYTHON_BIN} {SERVER_SCRIPT}")
        self.server_proc.start()

    def _on_srv_out(self):
        text = bytes(self.server_proc.readAllStandardOutput()).decode(errors="ignore")
        self._log_srv(text)
        if "SERVER_READY" in text and not self.server_ready:
            self.server_ready = True
            self._log_srv("[READY] 서버가 준비되었습니다.")
            self._update_buttons()

    def _on_srv_fin(self, code, status):
        self._log_srv(f"[서버 종료] code={code}")
        self.server_ready = False
        self._update_buttons()

    # ---------- 클라이언트 ----------
    def run_client_once(self, seed=None):
        if not (self.server_ready and self.selected_dir):
            QMessageBox.information(self, "안내", "서버 실행 및 폴더 선택 후 이용하세요.")
            return
        if not CLIENT_SCRIPT.exists():
            QMessageBox.critical(self, "오류", f"클라이언트 스크립트를 찾을 수 없습니다:\n{CLIENT_SCRIPT}")
            return

        self.client_proc = QProcess(self)
        self.client_proc.setProgram(PYTHON_BIN)

        if seed is None: seed = self.current_seed
        else: self.current_seed = seed

        self.result_json_path = default_result_json(self.selected_dir)
        self.result_json_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            str(CLIENT_SCRIPT),
            "--input", str(self.selected_dir),
            "--out", str(self.result_json_path),
            "--seed", str(seed)
        ]
        if self.chk_upload.isChecked():
            args.append("--upload")

        self.client_proc.setArguments(args)
        self.client_proc.setWorkingDirectory(str(CLIENT_SCRIPT.parent))
        self.client_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.client_proc.readyReadStandardOutput.connect(self._on_cli_out)
        self.client_proc.finished.connect(self._on_cli_fin)

        self._log_cli(f">>> 클라이언트 실행: {PYTHON_BIN} {' '.join(args)}")
        self.client_proc.start()

    def _on_cli_out(self):
        text = bytes(self.client_proc.readAllStandardOutput()).decode(errors="ignore")
        self._log_cli(text)
        for line in text.splitlines():
            if line.startswith("RESULT_JSON:"):
                p = line.split("RESULT_JSON:", 1)[1].strip()
                if p: self.result_json_path = Path(p)

    def _on_cli_fin(self, code, status):
        self._log_cli(f"[클라이언트 종료] code={code}")
        if self.result_json_path and self.result_json_path.exists():
            with open(self.result_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._populate_results(data)
        else:
            QMessageBox.warning(self, "주의", "결과 JSON을 찾지 못했습니다.")
        self._update_buttons()

    # ---------- 결과 탭 ----------
    def _populate_results(self, data: dict):
        while self.results_tabs.count(): self.results_tabs.removeTab(0)
        props = data.get("proposals", [])
        if not props:
            self._show_placeholder()
            return
        for i, prop in enumerate(props):
            name = prop.get("name", f"추천안 {i+1}")
            tree = prop.get("tree", {})
            view = self._build_tree_view(tree)
            self.results_tabs.addTab(view, name)
        self.results_tabs.setCurrentIndex(0)

    def _build_tree_view(self, d: dict) -> QTreeWidget:
        w = QTreeWidget(); w.setHeaderLabels(["폴더/파일"]); w.setColumnCount(1)
        def rec(parent, sub):
            for k, v in sub.items():
                it = QTreeWidgetItem([str(k)])
                parent.addChild(it)
                if isinstance(v, dict):
                    rec(it, v)
        rec(w.invisibleRootItem(), d)
        w.expandToDepth(1)
        return w

    # ---------- 다른 결과 ----------
    def reroll_results(self):
        if not (self.server_ready and self.selected_dir):
            QMessageBox.information(self, "안내", "서버 실행 및 폴더 선택 후 이용하세요.")
            return
        new_seed = random.randint(0, 1_000_000)
        self._log_cli(f"[다른 결과] seed={new_seed}")
        self.run_client_once(seed=new_seed)

    # ---------- 적용 ----------
    def apply_current_result(self):
        if not (self.server_ready and self.selected_dir):
            QMessageBox.information(self, "안내", "서버 실행 및 폴더 선택 후 이용하세요.")
            return
        if self.results_tabs.count() == 0 or self.results_tabs.tabText(0) == "결과 없음":
            QMessageBox.information(self, "안내", "적용할 결과가 없습니다.")
            return

        idx = self.results_tabs.currentIndex()
        ret = QMessageBox.question(
            self, "적용 확인",
            f"현재 탭(추천안 {idx+1})을 적용할까요?\n적용 전 백업을 권장합니다.",
            QMessageBox.Yes | QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return

        proc = QProcess(self)
        proc.setProgram(PYTHON_BIN)
        args = [
            str(CLIENT_SCRIPT),
            "--apply", str(idx),
            "--input", str(self.selected_dir),
            "--out", str(self.result_json_path if self.result_json_path else default_result_json(self.selected_dir))
        ]
        if self.chk_upload.isChecked():
            args.append("--upload")
        proc.setArguments(args)
        proc.setWorkingDirectory(str(CLIENT_SCRIPT.parent))
        proc.setProcessChannelMode(QProcess.MergedChannels)

        def on_out():
            t = bytes(proc.readAllStandardOutput()).decode(errors="ignore")
            self._log_cli(t)
        def on_fin(code, s):
            self._log_cli(f"[적용 완료] code={code}")
            QMessageBox.information(self, "완료", "적용이 완료되었습니다.")
            proc.deleteLater()

        proc.readyReadStandardOutput.connect(on_out)
        proc.finished.connect(on_fin)

        self._log_cli(f">>> 적용 실행: {PYTHON_BIN} {' '.join(args)}")
        proc.start()

if __name__ == "__main__":
    print("test")
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
