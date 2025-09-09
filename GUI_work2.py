import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget,
    QGroupBox, QHBoxLayout, QCheckBox, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from send2trash import send2trash



class ImageGroupDialog(QDialog):
    """
    중복/유사 이미지 결과 표시 & 선택 삭제 다이얼로그
    - dataset: PyG_Dataset 인스턴스 (idx_to_path 사용)
    """

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("중복/유사 이미지 그룹")
        self.resize(900, 600)
        self.setModal(True)

        self.dataset = dataset

        # ===== Title =====
        title = QLabel("이미지 유사도 분석기")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        # ===== Results =====
        self.result_area = QScrollArea()
        self.result_area.setWidgetResizable(True)

        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_area.setWidget(self.result_container)

        lay = QVBoxLayout(self)                 # 레이아웃 생성
        lay.addWidget(title)                    # 레이아웃에 요소(위젯) 추가
        lay.addWidget(self.result_area, 1)      # 레이아웃에 요소(위젯) 추가

        # ===== 분석 즉시 실행 =====
        QApplication.setOverrideCursor(Qt.WaitCursor)   # 마우스 커서 로딩으로 변경
        try:
            result = self.dataset.get_images_extract()   # Folder_Tree에서 연결한 함수 (analyze_folder 호출)
            self.display_result(result)
        except Exception as e:
            # 오류 메시지를 UI에 표시
            self.result_layout.addWidget(QLabel(f"❌ 분석 중 오류: {e}"))
            return
        finally:
            QApplication.restoreOverrideCursor()            # 마우스 커서 원래대로 변경


    def display_result(self, result):
        """
        기존 위젯 초기화
        """
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
        """
        그룹끼리 표시
        """
        section = QGroupBox(title)          # 이 함수에서 표시할 한 섹션을 감쌈
        section_layout = QVBoxLayout()      # 해당 섹션 안에 그룹들을 세로로 배치하기 위해 레이아웃 생성

        if groups:  # 그룹이 있다면 그룹만큼 반복
            for g_idx, group in enumerate(groups, 1):   # 그룹마다 순서대로 표시  (그룹 번호와 그룹)
                group_box = QGroupBox(f" 그룹 {g_idx} ({len(group)}개)")    # 그룹단위 GUI 박스
                group_layout = QVBoxLayout()                               # 그룹안의 파일들을 세로로 정렬한 레이아웃

                for idx, sim in group:  # 각 그룹안의 인덱스와 유사도 추출
                    full_path = self.dataset.idx_to_path[idx]       # idx로 파일의 절대경로 추출
                    fname = os.path.basename(full_path)             # 절대경로에서 파일 이름 추출
                               
                    # 체크박스 생성
                    h_layout = QHBoxLayout()                               # 체크박스 생성
                    checkbox = QCheckBox(f"{fname} (유사도: {sim:.1f}%)")   # 체크박스에 파일이름과 유사도 표시
                    checkbox.setChecked(False)                             # 체크박스 체크 해제상태 (기본)

                    checkbox.setProperty("file_idx", idx)           # 체크박스 객체에 파일 인덱스 저장      (빼내서 사용하기 용이해짐)
                    checkbox.setProperty("full_path", full_path)    # 체크박스 객체에 파일 전체경로 저장    (빼내서 사용하기 용이해짐)

                    h_layout.addWidget(checkbox)            # 가로 한줄에 체크박스 하나 배치
                    group_layout.addLayout(h_layout)        # 그룹 내에서 파일들을 세로로 쌓아올림

                group_box.setLayout(group_layout)       # 그룹 단위로 group_box에 해당 그룹의 체크박스들 추가 
                section_layout.addWidget(group_box)     # group_box를 섹션 레이아웃에 추가
        else:
            section_layout.addWidget(QLabel(" 그룹 없음"))  # groups가 아예 없는 경우

        section.setLayout(section_layout)       # 섹션 전체 레이아웃을 적용
        self.result_layout.addWidget(section)   # 섹션을 최종 결과 화면(self.result_layout)에 추가

        # ✅ 선택 항목 휴지통 버튼
        btn_delete_selected = QPushButton("선택 항목 휴지통으로 보내기")                        # 휴지통 보내기 버튼 객체 생성
        btn_delete_selected.clicked.connect(lambda _, s=section: self.delete_selected(s))   # 클릭 시, 체크박스가 활성화 된 파일만 삭제하는 함수 동작
        section_layout.addWidget(btn_delete_selected)                                       # 섹션에 휴지통 보내기 버튼 위젯 추가


    def delete_selected(self, section):
        """
        체크박스 확인 후 삭제
        """
        # 섹션 레이아웃 < 그룹 박스 < 행 레이아웃(그룹안의 각 파일들) < 체크박스

        # 세션 레이아웃에 포함된 모든 레이아웃(그룹박스/버튼 등)을 순회
        for layout_index in range(section.layout().count()):    
            item = section.layout().itemAt(layout_index)    
            widget = item.widget()                          

            if isinstance(widget, QGroupBox):   # 그룹박스 안의 인스턴스(파일)가 존재하는지 여부 확인
                group_layout = widget.layout()  # 그룹박스 안의 그룹 레이아웃(각 파일) 추출

                # 그룹에 속한 파일들을 하나씩 순회
                for i in range(group_layout.count()):       
                    h_layout_item = group_layout.itemAt(i)  # 그룹 레이아웃에 속한 아이템 추출
                    h_layout = h_layout_item.layout()       # 아이템이 레이아웃이면 해당 레이아웃 추출
                    # 레이아웃 여부 확인
                    if h_layout:

                        # 각 수평 레이아웃에 포함된 위젯들(체크박스)을 순회
                        for j in range(h_layout.count()):
                            w = h_layout.itemAt(j).widget()                                     # "수평 레이아웃"에서 "체크박스" 추출

                            # 체크박스 선택 여부 확인
                            if isinstance(w, QCheckBox) and w.isChecked() and w.isEnabled():    
                                
                                full_path = w.property("full_path")                             # "체크박스"에서 "해당 파일의 절대경로" 추출
                                # 파일 경로가 실제로 존재하는지 확인
                                if os.path.exists(full_path):
                                    send2trash(full_path)                   # 휴지통으로 해당 파일 이동
                                    w.setText(w.text() + " ✅ 삭제됨")      # 체크박스 텍스트에 '삭제됨' 추가
                                    w.setEnabled(False)                     # 체크박스 비활성화

