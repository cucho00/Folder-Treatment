import os
# PyQt6 모듈로 변경
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget,
    QGroupBox, QHBoxLayout, QCheckBox, QApplication, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from send2trash import send2trash


class ImageGroupDialog(QDialog):
    """
    중복/유사 이미지 결과 표시 & 선택 삭제 다이얼로그 (v2.1)
    - "삭제 버튼" 버그 수정
    - "전체 선택" 기능 포함
    """
    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("중복/유사 이미지 그룹")
        self.resize(900, 700)
        self.setModal(True)

        self.dataset = dataset
        self.all_checkboxes = [] # 모든 체크박스 관리를 위한 리스트

        # ===== Title =====
        title = QLabel("중복/유사 이미지 분석기")
        title.setObjectName("titleLabel") # QSS용 ID
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ===== Results =====
        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)
        self.result_area.setObjectName("ResultScrollArea")

        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_area.setWidget(self.result_container)

        # ===== 메인 레이아웃 =====
        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addWidget(self.result_area, 1)
        
        # ===== 전체 삭제 버튼 (하단에 고정) =====
        # (display_result에서 이쪽으로 이동)
        self.btn_delete_selected = QPushButton("선택된 항목 휴지통으로 보내기")
        self.btn_delete_selected.setObjectName("SpecialButton") # QSS용 ID
        self.btn_delete_selected.clicked.connect(self.delete_selected)
        self.btn_delete_selected.setEnabled(False) # <--- 기본값은 비활성화
        lay.addWidget(self.btn_delete_selected) # <--- 레이아웃에 바로 추가

        # QSS 적용 (부모 창의 스타일시트를 상속받음)
        if parent:
            self.setStyleSheet(parent.styleSheet())
            
        # ===== 분석 즉시 실행 =====
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.dataset.get_images_extract()
            self.display_result(result) # <--- 이 함수가 버튼을 활성화시킬 것임
        except Exception as e:
            self.result_layout.addWidget(QLabel(f" 분석 중 오류: {e}"))
            # (오류 발생 시 버튼은 비활성화 상태 유지)
            return
        finally:
            QApplication.restoreOverrideCursor()


    def display_result(self, result):
        self.all_checkboxes.clear()
        for i in reversed(range(self.result_layout.count())):
            widget = self.result_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # (버그 수정) result가 없어도 함수가 종료되지 않음
        if not result:
            self.result_layout.addWidget(QLabel(" 결과가 없습니다."))
            # (버튼은 비활성화 상태 유지)
            return

        # 텍스트 그룹 / 이미지 그룹 표시
        self.add_group_section(" 텍스트 기반 유사 그룹", result.get("text_groups", []))
        self.add_group_section(" 이미지 기반 유사 그룹", result.get("image_groups", []))

        # --- 중요: 버튼 활성화 ---
        # add_group_section을 통해 all_checkboxes 리스트가 채워짐
        if self.all_checkboxes:
            self.btn_delete_selected.setEnabled(True) # <--- 결과가 있으면 버튼 활성화
        else:
            self.result_layout.addWidget(QLabel("중복되거나 유사한 파일이 없습니다."))
            # (버튼은 비활성화 상태 유지)


    def add_group_section(self, title, groups):
        # 섹션 그룹박스 (QGroupBox)
        section = QGroupBox(title)
        section_layout = QVBoxLayout()

        if groups:
            for g_idx, group in enumerate(groups, 1):
                group_box = QGroupBox(f"그룹 {g_idx} ({len(group)}개)")
                group_box.setObjectName("InnerGroup") # QSS용 ID
                group_layout = QVBoxLayout()
                
                # --- "전체 선택" 체크박스 추가 ---
                chk_all = QCheckBox("이 그룹 전체 선택")
                group_checkboxes = [] # 이 그룹에 속한 체크박스 리스트
                
                for idx, sim in group:
                    full_path = self.dataset.idx_to_path[idx]
                    fname = os.path.basename(full_path)
                    
                    h_layout = QHBoxLayout()
                    checkbox = QCheckBox(f"{fname} (유사도: {sim:.1f}%)")
                    checkbox.setChecked(False)
                    checkbox.setProperty("full_path", full_path)
                    
                    h_layout.addWidget(checkbox)
                    group_layout.addLayout(h_layout)
                    
                    group_checkboxes.append(checkbox) # 리스트에 추가
                    self.all_checkboxes.append(checkbox) # 전체 리스트에도 추가

                # "전체 선택" 시그널 연결
                chk_all.toggled.connect(lambda checked, boxes=group_checkboxes: [
                    box.setChecked(checked) for box in boxes if box.isEnabled()
                ])
                
                # 그룹 레이아웃의 맨 위에 '전체 선택' 추가
                group_layout.insertWidget(0, chk_all) 
                group_box.setLayout(group_layout)
                section_layout.addWidget(group_box)
        else:
            section_layout.addWidget(QLabel("  그룹 없음"))

        section.setLayout(section_layout)
        self.result_layout.addWidget(section)


    def delete_selected(self):
        deleted_count = 0
        for w in self.all_checkboxes:
            if isinstance(w, QCheckBox) and w.isChecked() and w.isEnabled():
                full_path = w.property("full_path")
                if os.path.exists(full_path):
                    try:
                        send2trash(full_path)
                        w.setText(w.text() + "  삭제됨")
                        w.setEnabled(False)
                        deleted_count += 1
                    except Exception as e:
                        w.setText(w.text() + f"  삭제 실패: {e}")
        
        # 삭제 후 사용자에게 알림
        if deleted_count > 0:
            QMessageBox.information(self, "삭제 완료", f"{deleted_count}개의 파일을 휴지통으로 보냈습니다.")
        else:
            QMessageBox.warning(self, "알림", "선택된 파일이 없습니다.")
            
# --- 단독 실행용 테스트 코드 ---
if __name__ == "__main__":
    import sys
    
    class MockDataset:
        def __init__(self):
            self.idx_to_path = [
                "/fake/path/image1.jpg", "/fake/path/image2.png", "/fake/path/image3.jpg",
                "/fake/path/text_image1.png", "/fake/path/text_image2.jpg"
            ]
        def get_images_extract(self):
            # --- 테스트 시나리오 ---
            # 1. 그룹이 있는 경우 (정상)
            return {
                "text_groups": [[(3, 100.0), (4, 85.1)]],
                "image_groups": [[(0, 100.0), (1, 95.5), (2, 90.0)]]
            }
            # 2. 그룹이 없는 경우 (버튼 비활성화 테스트)
            # return {"text_groups": [], "image_groups": []}
            # 3. 결과가 아예 없는 경우 (버튼 비활성화 테스트)
            # return None

    print("GUI_work2.py 단독 실행 모드 (디자인 테스트)")
    app = QApplication(sys.argv)
    
    # --- 가짜 부모 창 및 QSS (테스트용) ---
    mock_parent = QWidget()
    mock_parent.setStyleSheet("""
        QWidget { background-color: #F8F9FB; font-family: 'Noto Sans KR'; font-size: 14px; }
        QLabel#titleLabel { font-size: 24px; font-weight: 600; color: #2C3E50; }
        QGroupBox { 
            background-color: #FFFFFF; border: 1px solid #E0E0E0; 
            border-radius: 8px; margin-top: 10px; padding: 10px;
        }
        QGroupBox::title { 
            subcontrol-origin: margin; subcontrol-position: top left; 
            left: 10px; padding: 0 5px; background-color: #FFFFFF;
            color: #3498DB; font-size: 15px; font-weight: 600;
        }
        QGroupBox#InnerGroup { background-color: #FDFEFE; }
        QPushButton { 
            background-color: #3498DB; color: white; border: none; 
            border-radius: 6px; padding: 8px 14px;
        }
        QPushButton:hover { background-color: #5DADE2; }
        QPushButton#SpecialButton { background-color: #E74C3C; }
        QPushButton#SpecialButton:hover { background-color: #EC7063; }
        QPushButton:disabled { background-color: #D5DBDB; }
        QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #BDC3C7; border-radius: 4px; }
        QCheckBox::indicator:checked { 
            background-color: #3498DB; border-color: #3498DB; 
            image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17L4 12"></path></svg>');
        }
    """)
    
    dialog = ImageGroupDialog(MockDataset(), parent=mock_parent)
    dialog.show()
    sys.exit(app.exec())