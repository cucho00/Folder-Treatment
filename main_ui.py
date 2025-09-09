import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QScrollArea, QHBoxLayout, QGroupBox, QCheckBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from send2trash import send2trash

# ✅ app.py에 정의된 함수 가져오기
from app import analyze_folder

class ImageGroupApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("이미지 유사도 분석기")
        self.setMinimumSize(800, 600)

        self.folder_path = ""

        self.layout = QVBoxLayout(self)

        # ===== Title =====
        title = QLabel("이미지 유사도 분석기")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        # ===== Controls =====
        controls = QHBoxLayout()
        self.path_display = QLabel(" 분석할 폴더를 선택해주세요")
        controls.addWidget(self.path_display)

        btn_browse = QPushButton("폴더 열기")
        btn_browse.clicked.connect(self.browse_folder)
        controls.addWidget(btn_browse)

        btn_analyze = QPushButton("분석 시작")
        btn_analyze.clicked.connect(self.run_analysis)
        controls.addWidget(btn_analyze)

        self.layout.addLayout(controls)

        # ===== Results =====
        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)

        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_area.setWidget(self.result_container)

        self.layout.addWidget(self.result_area)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if folder:
            self.folder_path = folder
            self.path_display.setText(f" 선택된 폴더: {folder}")

    def run_analysis(self):
        if not self.folder_path:
            self.path_display.setText(" 먼저 폴더를 선택하세요.")
            return

        self.path_display.setText(" 분석 중입니다. 잠시만 기다려주세요...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = analyze_folder(self.folder_path)
            self.display_result(result)
            self.path_display.setText(f" 분석 완료: {self.folder_path}")
        except Exception as e:
            self.path_display.setText(f" 분석 중 오류 발생: {str(e)}")
            self.display_result(None)
        finally:
            QApplication.restoreOverrideCursor()

    def display_result(self, result):
        # 기존 위젯 초기화
        for i in reversed(range(self.result_layout.count())):
            widget = self.result_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if not result:
            self.result_layout.addWidget(QLabel("❌ 결과가 없습니다."))
            return

        # 텍스트 그룹 / 이미지 그룹 표시
        self.add_group_section(" 텍스트 기반 유사 그룹", result.get("text_groups", []))
        self.add_group_section(" 이미지 기반 유사 그룹", result.get("image_groups", []))

    def add_group_section(self, title, groups):
        section = QGroupBox(title)
        section_layout = QVBoxLayout()

        if groups:
            for idx, group in enumerate(groups, 1):
                group_box = QGroupBox(f" 그룹 {idx} ({len(group)}개)")
                group_layout = QVBoxLayout()

                for fname, sim in group:
                    h_layout = QHBoxLayout()
                    
                    # 체크박스
                    checkbox = QCheckBox(f"{fname} (유사도: {sim:.1f}%)")
                    checkbox.setChecked(False)
                    h_layout.addWidget(checkbox)
                    
                    group_layout.addLayout(h_layout)

                group_box.setLayout(group_layout)
                section_layout.addWidget(group_box)
        else:
            section_layout.addWidget(QLabel(" 그룹 없음"))

        section.setLayout(section_layout)
        self.result_layout.addWidget(section)

        # ✅ 선택 항목 휴지통 버튼
        btn_delete_selected = QPushButton("선택 항목 휴지통으로 보내기")
        btn_delete_selected.clicked.connect(lambda _, s=section: self.delete_selected(s))
        section_layout.addWidget(btn_delete_selected)

    def delete_selected(self, section):
        # 섹션 안의 체크박스 확인 후 삭제
        for layout_index in range(section.layout().count()):
            item = section.layout().itemAt(layout_index)
            widget = item.widget()
            if isinstance(widget, QGroupBox):
                group_layout = widget.layout()
                for i in range(group_layout.count()):
                    h_layout_item = group_layout.itemAt(i)
                    h_layout = h_layout_item.layout()
                    if h_layout:
                        for j in range(h_layout.count()):
                            w = h_layout.itemAt(j).widget()
                            if isinstance(w, QCheckBox) and w.isChecked():
                                # 파일 이름 추출
                                fname = w.text().split(" ")[0]
                                file_path = os.path.join(self.folder_path, fname)
                                if os.path.exists(file_path):
                                    send2trash(file_path)
                                    w.setText(w.text() + " ✅ 삭제됨")
                                    w.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImageGroupApp()
    win.show()
    sys.exit(app.exec())