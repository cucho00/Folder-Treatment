### 진행 상황
데이터셋 전송되는지 확인 못했음. (방법을 모르겠는데 알려주세요.)
다른 결과 보는 버튼이 작동을 안함. (다른 추론이 동작하는지 확인해야함)

### GUI.py 실행 시

폴더 선택 - 서버 시작 - 클라이언트 시작 순으로 실행해야 함.

좌측하단 - 서버 로그
우측하단 - 클라이언트 로그

클라이언트 로그에는 생성될 폴더, 삭제될 폴더, 추천 트리 최상위가 출력됨

### 실행 전 필수 모듈 설치

```bash
pip3 install PyQt5 torch torch_geometric scikit-learn flwr
```

### GUI.py 실행

Folder-Treatment/ 로 터미널의 위치이동 후 다음 명령어 실행

```bash
# 연합학습 서버 실행파일
python3 GUI_S_work.py       
# "폴더 처리 시스템" 실행 파일
python3 GUI_work.py         
```

실행 시 GUI.py 내부에 클라이언트, 서버 py 파일 경로를 하드코딩 해두었으니 반드시 해당 폴더 구조를 유지해야 함

# GCN-Folder-Project
