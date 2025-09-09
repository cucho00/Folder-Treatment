import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import networkx as nx
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ✅ Tesseract 실행 파일 경로 지정 (Windows 기준)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===== CNN 모델 (이미지 특징 추출용) =====
resnet_model = models.resnet18(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


TEXT_THRESHOLD = 30
TEXT_SIMILARITY_THRESHOLD = 0.7
IMAGE_SIMILARITY_THRESHOLD = 84
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')


def extract_image_feature(img_path):        # 이미지 파일 유사도 분석
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feature = resnet_model(img_tensor).squeeze().numpy()
        return feature
    except Exception as e:
        return None

def extract_text(img_path):                 # 이미지 속 글자 유사도 분석
    try:
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img, lang="kor+eng").strip()
        return text
    except Exception as e:
        return ""


def analyze_folder(idx_to_path: list[str], PyG: nx.Graph):            # 루트 폴더 경로속 이미지 파일들의 유사도 분석 후 결과가 저장된 딕셔너리 반환

    results = {
        "total_images": 0,
        "text_images": [],
        "image_images": [],
        "text_groups": [],
        "image_groups": [],
        "text_group_count": 0,
        "image_group_count": 0,
    }

    image_nodes = []        # 이미지 파일인 노드의 인덱스 값만 저장할 리스트
    for idx, attrs in PyG.nodes(data=True):         # PyG 데이터셋의 노드들의 idx(노드 인덱스 값, int)와 attrs(기타 정보 값, dict)을 추출하여 반복문에 사용
        ext = str(attrs.get("ext", "")).lower()     # attrs 에서 ext(확장자)만 추출
        if ext in IMAGE_EXTENSIONS:                 # 이미지 확장자 판별
            image_nodes.append(idx)                 # 이미지 노드의 index번호 만 저장
    results["total_images"] = len(image_nodes)  # 탐지된 파일 이름이 저장된 리스트 항목의 갯수를 저장


    text_images = []
    text_contents = []

    image_images = []
    image_features = {}

    for idx in image_nodes:                           # 이미지 노드의 인덱스를 순차적으로 확인
        full_path = idx_to_path[idx]                        # 노드의 인덱스로 해당 파일의 전체 경로 반환
        fname = os.path.basename(full_path)                 # 전체 경로에서 파일 이름만 추출
        text = extract_text(full_path)                      # 텍스트가 있는 파일은 텍스트 추출

        if len(text) >= TEXT_THRESHOLD:                 # 텍스트 길이 확인 (30글자 이상)
            text_images.append((idx, fname, text))          # 텍스트를 포함한 이미지 파일을 리스트에 저장 (중복 이름을 방지하기 위해 idx 포함)
            text_contents.append(text)                      # 이미지 파일의 텍스트 저장
        else:
            vec = extract_image_feature(full_path)      # 일반 이미지 벡터값 추출
            if vec is not None:                             
                image_images.append((idx, fname))           # 텍스트 없는 이미지 파일을 리스트에 저장 (중복 이름을 방지하기 위해 idx 포함)
                image_features[idx] = vec                   # 이미지 파일의 벡터값을 저장 (파일 이름 중복을 방지하기 위해 idx로 구분)

    results["text_images"] = text_images                # results 딕셔너리 안의 "text_images", "image_images"라는 키 항목에
    results["image_images"] = image_images              # 방금 구성한 text_images, image_images 리스트를 저장

    # ===== 텍스트 이미지 유사도 그룹화 =====
    text_groups = []
    if text_contents:
        vectorizer = TfidfVectorizer().fit_transform(text_contents)     # 문자열을 벡터로 변환 (TF-IDF)
        sim_matrix = cosine_similarity(vectorizer)                      # 벡터 간 코사인 유사도로 비교 (행렬 반환)
        visited = set()                                                 # 그룹화 할때, 중복 그룹화하지 않도록 집합 생성

        for i in range(len(text_images)):           # i 와 j 텍스트 간의 유사도 분석
            if i in visited:
                continue
            base = text_images[i][0]                # [i][0] = (idx, fname, text)에서의 "idx"값
            group = [(base, 100.0)]                 # 그룹 대표 텍스트는 100% 유사도
            visited.add(i)                          # 그룹 대표 텍스트를 집합에 추가

            for j in range(i + 1, len(text_images)):
                if j not in visited and sim_matrix[i][j] >= TEXT_SIMILARITY_THRESHOLD:      # 두 문자열 간 유사도가 기준을 넘어서는 경우
                    score = round(sim_matrix[i][j] * 100, 2)        # 두 문자열 간 유사도 값을 점수로 계산하여
                    group.append((text_images[j][0], score))        # group 리스트에 추가       (# [j][0] = (idx, fname, text)에서의 "idx"값)
                    visited.add(j)                              # 그룹에 속한 문자열은 visited 그룹에 포함되며 중복 그룹이 되지 않도록 제외됨

            if len(group) > 1:                      # 하나 이상 그룹이 존재하는 경우
                text_groups.append(group)           # 존재하는 그룹을 test_groups 리스트에 추가

    results["text_groups"] = text_groups            # results 딕셔너리 안의 "text_groups", "text_group_count" 라는 키 항목에
    results["text_group_count"] = len(text_groups)  # 방금 구성한 text_groups 리스트와 그룹의 갯수를 저장

    # ===== 일반 이미지 유사도 그룹화 =====
    image_groups = []
    visited = set()                             # 그룹화 할때, 중복 그룹화하지 않도록 집합 생성
    image_list = list(image_features.keys())    # image_features 딕셔너리의 키(이미지 파일 노드의 idx)만 따로 뽑은 리스트

    for i in range(len(image_list)):            # i 와 j 텍스트 간의 유사도 분석

        if image_list[i] in visited:            # 이전 작업에서의 그룹화 여부 확인 
            continue

        base = image_list[i]                    # 그룹 대표 이미지 노드의 인덱스
        base_vec = image_features[base]         # 그룹 대표 이미지 노드의 벡터 값
        group = [(base, 100.0)]                 # 그룹 대표 이미지는 100% 유사도
        visited.add(base)                       # 그룹 대표 이미지를 집합에 추가

        for j in range(i + 1, len(image_list)):
            compare = image_list[j]                     # j의 노드 인덱스
            if compare in visited:                      # 집합에 포함 여부 확인
                continue
            sim = cosine_similarity([base_vec], [image_features[compare]])[0][0] * 100      # 두 이미지 벡터 간 유사도 값을 점수로 계산하여
            if sim >= IMAGE_SIMILARITY_THRESHOLD:           # 유사도가 일정 수치 이상이라면
                group.append((compare, round(sim, 2)))          # 그룹에 유사도와 j를 추가
                visited.add(compare)                            # 집합에 j를 추가

        if len(group) > 1:
            image_groups.append(group)

    results["image_groups"] = image_groups
    results["image_group_count"] = len(image_groups)

    return results
