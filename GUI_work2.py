import os, platform, shutil
from datetime import datetime
from send2trash import send2trash

# PyQt6 모듈로 변경
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget,
    QGroupBox, QHBoxLayout, QCheckBox, QApplication, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontMetrics



class ImageGroupDialog(QDialog):
    """
    중복/유사 이미지 결과 표시 & 선택 삭제 다이얼로그 (v2.1)
    - "삭제 버튼" 버그 수정
    - "전체 선택" 기능 포함
    """
    def __init__(self, dataset, ft_root, parent=None, selected_exts=None):
        super().__init__(parent)
        self.setWindowTitle("중복/유사 이미지 그룹")
        self.resize(900, 700)
        self.setModal(True)

        self.dataset = dataset              # 정리할 파일들 데이터셋
        self.ft_root = ft_root              # 정리될 위치 폴더 이름
        self.all_checkboxes = []            # 모든 체크박스 관리를 위한 리스트
        self.selected_exts = selected_exts  # 선택한 확장자 목록 (이미지 중복 제거)

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
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)   # 커서가 로딩 아이콘(모래시계)으로 바뀜
        try:
            result = self.dataset.get_images_extract(self.selected_exts)  # AI 분석 진행 (app.py 파일의 PyG_Dataset.get_images_extract() 호출)
            self.display_result(result) # <--- 이 함수가 버튼을 활성화시킬 것임 (UI에 표시)
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


    # 유사도 높은 각 파일들의 상태 표시
    def _make_item_row(self, full, name, parent, mtime_fmt, size_fmt, sim, widths=None, mono_numeric=False):
        row = QWidget()                     # 파일 1개를 표현할 행 위젯
        h = QHBoxLayout(row)                # row의 내부 레이아웃
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(10)

        # 체크박스 (파일의 전체경로가 설정)
        cb = QCheckBox()
        cb.setProperty("full_path", full)
        cb.setToolTip(full)
        # 나머지 옵션 (파일이름, 부모 폴더, 수정 시간, 파일크기, 유사도)
        name_lbl = QLabel(name)
        folder_lbl = QLabel(parent or "-")
        mtime_lbl = QLabel(mtime_fmt)
        size_lbl = QLabel(size_fmt)
        sim_lbl  = QLabel(f"{sim:.1f}%")

        for lbl in (mtime_lbl, size_lbl, sim_lbl):      # 모든 라벨 우측정렬, 세로기준선은 중앙에 위치
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)   

        # 숫자/날짜 열 모노스페이스 적용(수정날짜, 파일크기, 유사도와 같은 숫자 항목은 각 글자별 크기 폭이 동일하게 지정)
        if mono_numeric:
            mono = QFont("Consolas")  # OS에 맞는 모노폰트로 바꿔도 됨
            for lbl in (mtime_lbl, size_lbl, sim_lbl):
                lbl.setFont(mono)
        # 세 항목간 고정 폭 적용 (모든 행 동일)
        if widths:
            mtime_lbl.setFixedWidth(widths["date"])
            size_lbl.setFixedWidth(widths["size"])
            sim_lbl.setFixedWidth(widths["sim"])

        # 가로로 나열(h) [CB] [이름(늘어남)] [폴더(늘어남)] [수정] [크기] [유사도]
        h.addWidget(cb)
        h.addWidget(name_lbl, 2)    # 가변
        h.addWidget(folder_lbl, 1)  # 가변
        h.addWidget(mtime_lbl)      # 고정
        h.addWidget(size_lbl)       # 고정
        h.addWidget(sim_lbl)        # 고정

        self.group_checkboxes.append(cb) # 리스트에 추가
        self.all_checkboxes.append(cb) # 전체 리스트에도 추가
        return row

    # 파일크기 정형화
    def _human_size_kb(self, kb: float) -> str:
        if kb is None:
            return "-"
        s = float(kb) * 1024.0
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if s < 1024.0:
                return f"{s:.1f} {unit}"
            s /= 1024.0
        return f"{s:.1f} PB"
    
    def _calc_column_widths(self, sample_font: QFont) -> dict:
        fm = QFontMetrics(sample_font)
        # 대충 최대치를 커버하는 템플릿 문자열로 계산
        w_date = fm.horizontalAdvance("2025-12-31 23:59")
        w_size = fm.horizontalAdvance("9999.9 MB")   # KB/MB 판단해 여유 있게
        w_sim  = fm.horizontalAdvance("100.0%")
        # 여유 패딩
        return {"date": w_date + 8, "size": w_size + 8, "sim": w_sim + 8}
    
    def _make_header_row(self, widths: dict):
        row = QWidget()
        row.setObjectName("HeaderRow")  # 스키마 헤더 디자인 적용
        h = QHBoxLayout(row)
        h.setContentsMargins(6,2,6,2); h.setSpacing(10)

        h.addWidget(QLabel(""), 0)              # 체크박스 칸
        lab_name = QLabel("파일이름");  lab_folder = QLabel("부모폴더")
        lab_date = QLabel("최근 수정"); lab_size = QLabel("크기"); lab_sim = QLabel("유사도")
        lab_name.setObjectName("HdrName")       # 파일 이름 디자인
        lab_folder.setObjectName("HdrFolder")   # 부모 폴더 이름 디자인

        # 정렬/굵기
        for lab in (lab_name, lab_folder, lab_date, lab_size, lab_sim):
            lab.setStyleSheet("font-weight:600;")
        for lab in (lab_date, lab_size, lab_sim):
            lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        h.addWidget(lab_name, 2)
        h.addWidget(lab_folder, 1)

        lab_date.setFixedWidth(widths["date"])
        lab_size.setFixedWidth(widths["size"])
        lab_sim.setFixedWidth(widths["sim"])

        h.addWidget(lab_date)
        h.addWidget(lab_size)
        h.addWidget(lab_sim)
        return row

    def add_group_section(self, title, groups):
        section = QGroupBox(title)                  # 섹션 그룹박스 (QGroupBox)
        section_layout = QVBoxLayout()

        if groups:
            widths = self._calc_column_widths(self.font())              # 한 번만 폭 계산(현재 폰트 기준)

            # 파일에 대한 각 항목 헤더(스키마)
            header = self._make_header_row(widths)
            section_layout.addWidget(header)

            for g_idx, group in enumerate(groups, 1):
                group_box = QGroupBox(f"그룹 {g_idx} ({len(group)}개)")
                group_box.setObjectName("InnerGroup") # QSS용 ID
                group_layout = QVBoxLayout()
                
                # --- "전체 선택" 체크박스 추가 ---
                chk_all = QCheckBox("이 그룹 전체 선택")
                self.group_checkboxes = []               # 이 그룹에 속한 체크박스 리스트
                group_layout.addWidget(chk_all)     # 체크박스 전체선택을 추가 (가장 먼저)

                # 그룹 아이템은 (idx, sim, full, name, parent, mtime, size_kb) 형태
                for item in group:
                    try:
                        idx, sim, full, name, parent, mtime, size_kb = item[:7]
                    except Exception:
                        # 혹시 일부 항목이 빠져있을 경우 안전하게 처리
                        continue
                    try:
                        mtime_fmt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")    # 날짜 포멧팅
                    except Exception:   
                        mtime_fmt = "-"
                    size_fmt = self._human_size_kb(size_kb)                                     # 크기 포멧팅
                    
                    row = self._make_item_row(full, name, parent, mtime_fmt, size_fmt, sim,
                                              widths=widths, mono_numeric=True)
                    group_layout.addWidget(row)         

                # "전체 선택" 시그널 연결
                def on_all_toggled(checked, boxes=self.group_checkboxes):
                    for box in boxes:
                        if box.isEnabled():
                            box.setChecked(checked)
                chk_all.toggled.connect(on_all_toggled)

                # 그룹 레이아웃의 맨 위에 '전체 선택' 추가
                group_box.setLayout(group_layout)
                section_layout.addWidget(group_box)
        else:
            section_layout.addWidget(QLabel("  그룹 없음"))

        section.setLayout(section_layout)
        self.result_layout.addWidget(section)


    def delete_selected(self):
        os_name = platform.system()  # "Windows" | "Linux" | "Darwin"
        tr_index = 1  # 파일 일련번호 시작값
        deleted_count = 0

        match os_name:
            # -------------------- Windows --------------------
            case "Windows":
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
                                w.setText(w.text() + f"  [ 삭제 실패 ]: {e}")

            # -------------------- macOS / Linux / 기타 OS --------------------
            case _:
                trash_dir = os.path.join(self.ft_root, "Trash--(FT)")   # 휴지통 폴더이름 생성
                os.makedirs(trash_dir, exist_ok=True)                   # 휴지통 폴더 생성

                for w in self.all_checkboxes:
                    if isinstance(w, QCheckBox) and w.isChecked() and w.isEnabled():
                        full_path = w.property("full_path")                                 # 파일이름이 있는 체크박스 객체(w)에서 파일 전체 경로 추출
                        if os.path.exists(full_path):
                            try:
                                name, ext = os.path.splitext(os.path.basename(full_path))   # 파일 전체경로에서 파일이름을 추출 -> 전체 파일이름에서 파일이름과 확장자를 추출
                                new_name = f"{name}--(tr_{tr_index}){ext}"                  # 휴지통으로 옮겨진 파일 이름 생성
                                shutil.move(full_path, os.path.join(trash_dir, new_name))   # 파일 이동
                                w.setText(w.text() + f"  (삭제됨 → {new_name})")
                                w.setEnabled(False)
                                deleted_count += 1
                                tr_index += 1
                            except Exception as e:
                                w.setText(w.text() + f"  [ 삭제 실패 ]: {e}")

        # 삭제 후 사용자에게 알림
        if deleted_count > 0:
            QMessageBox.information(self, "  [ 삭제 완료 ]", f"{deleted_count}개의 파일을 휴지통으로 보냈습니다.")
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