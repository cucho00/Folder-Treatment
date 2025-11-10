import os
import networkx as nx
from pathlib import Path
from torch_geometric.utils import from_networkx
from app import analyze_folder
import torch


# 1. os.walk() 등을 통해 폴더 구조를 탐색
# 2. networkx를 사용해 트리 그래프 생성 (nx.Graph() 또는 nx.DiGraph())
# 3. networkx → PyG Data 객체로 변환
# PyG Data 객체는 "노드의 특성 행렬", "노드간의 연결 정보", "그래프 전체 라벨" 총 3개의 값 필요

class PyG_Dataset:
    def __init__(self, root_path: str | Path):
        self.root = str(root_path)              # 정리할 폴더 경로
        self.G = nx.DiGraph()                   # Networkx 라이브러리의 "유향 그래프"(Directed Graph) 객체인 "G"를 생성
        self.path_to_idx: dict[str, int] = {}   # 전체 경로(키) : index(값) 구조의 딕셔너리
        self.idx_to_path: list[str] = []        # "index"위치에 "전체 경로"를 저장하는 리스트

    def build_graph_from_folder(self, label: int = 0, with_label: bool = True):
        
        root = self.root
        idx = 0

        for dirpath, dirnames, filenames in os.walk(root):   
            # os.walk(path)는 탐색 시작 경로 path의 하위 폴더를 재귀적으로 탐색하면서, 
            # 각 폴더마다 아래의 3가지 정보를 튜플 형태로 순차적으로 반환함
                # dirpath = 현재 탐색 중인 폴더의 전체 경로
                # dirnames = 현재 폴더 안에 있는 하위 폴더들의 이름 (문자열 리스트)
                # filenames = 현재 폴더 안에 있는 파일들의 이름 (문자열 리스트)

            depth = dirpath[len(root):].count(os.sep)
                # dirpath[len(root_dir):] = 현재 위치의 절대 경로 - root_path의 절대경로
                # .count(os.sep) = 그 상대 경로에서 "/"(linux), "\"(Window)의 갯수를 셈셈

            # 노드 정의 (폴더)
            for dname in dirnames:
                full_path = os.path.join(dirpath, dname)    # 해당 폴더의 전체 경로 추출
                self.path_to_idx[full_path] = idx                   # "폴더의 전체 경로 : index 값" 의 형태로 Key:Value 저장
                self.idx_to_path.append(full_path)               # "폴더의 전체 경로"를 "index 값" 위치에 저장


                is_file, size_kb = 0, 0.0   # 0 = 폴더, 0.0 = 폴더 크기(없음)
                mtime = os.path.getmtime(full_path)             # 마지막 수정 시간
                features = [is_file, size_kb, depth, mtime]     # 노드 특성을 행렬로 저장 
                    # 파일 여부, 파일 크기, root 폴더에서의 깊이, 부모 폴더 이름, 파일 이름, 확장자, 마지막 수정 시간
                self.G.add_node(idx, 
                        x=torch.tensor(features, dtype=torch.float),
                        name = dname, 
                        ext = "",
                        parent = os.path.basename(dirpath)
                )
                    # G 에 노드에 대한 아래의 3가지 정보를 추가
                        # idx = 해당 폴더의 고유한 index 값
                        # features = 노드의 특성을 저장한 행렬 
                        # dtype = 특성 Data 타입
                idx += 1                # 노드를 하나 추가하면 다음 index로 넘어감

            # 노드 정의 (파일)
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)    # 해당 파일의 전체 경로 추출
                self.path_to_idx[full_path] = idx                   # "파일의 전체 경로 : index 값" 의 형태로 Key:Value 저장
                self.idx_to_path.append(full_path)               # "폴더의 전체 경로"를 "index 값" 위치에 저장

                is_file = 1             # 1 = 파일
                try:    # 파일 크기 추출 시도
                    size_kb = os.path.getsize(full_path) / 1024.0   # KB 단위
                    mtime = os.path.getmtime(full_path)             # 마지막 수정 시간
                except Exception: # 파일 크기를 추출할 수 없다면 0.0
                    size_kb, mtime = 0.0, 0.0
                features = [is_file, size_kb, depth, mtime]        # 노드 특성을 행렬로 저장장
                self.G.add_node(idx, 
                        x=torch.tensor(features, dtype=torch.float),
                        name = fname,
                        ext = os.path.splitext(fname)[1].lower(),   # 확장자 (예: .txt)
                        parent = os.path.basename(dirpath)
                )
                    # G 에 노드에 대한 아래의 3가지 정보를 추가
                        # idx = 해당 폴더의 고유한 index 값
                        # features = 노드의 특성을 저장한 행렬 
                        # dtype = 특성 Data 타입
                idx += 1                # 노드를 하나 추가하면 다음 index로 넘어감
            
            # "폴더 -> 하위폴더" 간의 간선 추가
                # 해당 os.walk()를 통해 반환되는 모든 폴더들과 그 안의 폴더를 간선으로 연결
            for dname in dirnames:          
                p = dirpath                                     # 부모 폴더 지정 (전체 경로)
                c = os.path.join(dirpath, dname)                # 자식 폴더 지정 (전체 경로)
                if p in self.path_to_idx and c in self.path_to_idx:        # 부모, 자식에 대한 노드가 존재한다면,
                    self.G.add_edge(self.path_to_idx[p], self.path_to_idx[c])   # 두 노드 사이의 방향 간선을 G에 추가
                        # edge_index 정보 추가

            # "폴더 -> 파일" 간의 간선 추가
                # 해당 os.walk()를 통해 반환되는 모든 폴더들과 그 안의 파일을 간선으로 연결
            for fname in filenames:
                p = dirpath                                     # 부모 폴더 지정
                c = os.path.join(dirpath, fname)                # 자식 파일 지정
                if p in self.path_to_idx and c in self.path_to_idx:        # 부모, 자식에 대한 노드가 존재한다면,
                    self.G.add_edge(self.path_to_idx[p], self.path_to_idx[c])   # 두 노드 사이의 방향 간선을 G에 추가
                        # edge_index 정보 추가

        data = from_networkx(self.G)
            # networkx에 G를 넣으면, G는 유향 그래프이기에 
            # 유향 그래프에 맞는 PyG 형태의 데이터로 변환
                # data.x = 노드의 특성 행렬
                # data.edge_index = 간선 정보
                # data.y = 정답(label)
            # 이제부터 data는 "PyG 데이터" 객체이다.
        
        # 학습용일 때만 y 생성
        if with_label:
            data.y = torch.full((data.x.size(0),), label, dtype=torch.long)
            # PyG 객체에 부족한 라벨 정보(문제의 정답) 추가
                # 그래프의 모든 노드에 label의 답을 지정한 것

        return data         # 3개의 데이터를 모두 포함한 PyG 데이터 셋을 반환


    # 외부에 정의된 함수를 그대로 호출해서 결과를 반환하는 래퍼(wrapper) 메서드
    def get_images_extract(self, selected_exts=None) -> dict:
        return analyze_folder(self.idx_to_path ,self.G, selected_exts=selected_exts)