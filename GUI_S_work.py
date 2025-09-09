# GUI_S_work.py
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit, QMessageBox
from PyQt5.QtCore import QProcess

PYTHON_BIN = sys.executable
SERVER_SCRIPT = (Path(__file__).resolve().parent / "Server" / "GCN_Server_Run.py").resolve()

class ServerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flower Server GUI")
        self.resize(600, 400)

        self.proc: QProcess | None = None

        # 버튼 + 로그창
        self.btn_start = QPushButton("서버 시작")
        self.btn_stop = QPushButton("서버 종료")
        self.btn_stop.setEnabled(False)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.log)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 연결
        self.btn_start.clicked.connect(self.start_server)
        self.btn_stop.clicked.connect(self.stop_server)

    def start_server(self):
        if not SERVER_SCRIPT.exists():
            QMessageBox.critical(self, "오류", f"서버 스크립트를 찾을 수 없습니다:\n{SERVER_SCRIPT}")
            return
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "알림", "이미 실행 중입니다.")
            return

        self.proc = QProcess(self)
        self.proc.setProgram(PYTHON_BIN)
        self.proc.setArguments([str(SERVER_SCRIPT)])
        self.proc.setWorkingDirectory(str(SERVER_SCRIPT.parent))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_output)
        self.proc.finished.connect(self.on_finish)

        self.log.appendPlainText(f">>> 서버 시작: {PYTHON_BIN} {SERVER_SCRIPT}")
        self.proc.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_server(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.terminate()
            self.log.appendPlainText("[STOP] 서버 종료 요청 보냄...")
        else:
            self.log.appendPlainText("[INFO] 서버가 실행 중이 아님")

    def on_output(self):                # 로그 출력
        if not self.proc:
            return
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="ignore")
        self.log.appendPlainText(text)

    def on_finish(self, code, status):  # 연합학습 서버 프로세스가 종료될 때 호출
        self.log.appendPlainText(f"[서버 종료] code={code}")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ServerWindow()
    win.show()
    sys.exit(app.exec_())