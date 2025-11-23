import os
import sys
import datetime
import sqlite3
import pandas as pd
import uuid

# Add these lines at the top of your script
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10
from PIL import Image, ImageEnhance, ImageOps
from scipy.spatial.distance import cosine
import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms

# 결과 저장을 위한 폴더 생성
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Result 폴더 생성
ensure_dir("Result")

# SQLite DB 초기화
def init_db():
    conn = sqlite3.connect('similarity_results.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        real_image_path TEXT,
        ai_image_path TEXT,
        ssim_score REAL,
        psnr_score REAL,
        vgg_score REAL,
        avg_score REAL
    )
    ''')
    conn.commit()
    conn.close()

# DB 초기화 실행 - 이 부분이 추가됨
init_db()

# DB에 결과 저장
def save_results(real_image_path, ai_image_path, ssim_score, psnr_score, vgg_score, avg_score):
    conn = sqlite3.connect('similarity_results.db')
    c = conn.cursor()
    result_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    c.execute('''
    INSERT INTO results (id, timestamp, real_image_path, ai_image_path, ssim_score, psnr_score, vgg_score, avg_score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (result_id, timestamp, real_image_path, ai_image_path, ssim_score, psnr_score, vgg_score, avg_score))
    conn.commit()
    conn.close()
    return result_id

# DB에서 결과 가져오기
def get_results():
    conn = sqlite3.connect('similarity_results.db')
    query = "SELECT * FROM results ORDER BY timestamp DESC"
    try:
        results = pd.read_sql(query, conn)
    except:
        results = pd.DataFrame(columns=["id", "timestamp", "real_image_path", "ai_image_path", "ssim_score", "psnr_score", "vgg_score", "avg_score"])
    conn.close()
    return results

# 간단한 배경 제거 함수 구현 (임계값 기반)
def remove_background(image, threshold=240):
    """Simple background removal based on color threshold"""
    img_array = np.array(image)
    
    # RGB 이미지를 처리
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # 밝은 배경 픽셀 마스크 생성
        mask = np.all(img_array > threshold, axis=2)
        
        # RGBA 이미지 생성 (알파 채널 포함)
        rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = img_array
        rgba[:, :, 3] = np.where(mask, 0, 255)  # 배경은 투명하게, 객체는 불투명하게
        
        return Image.fromarray(rgba)
    return image

# 이미지 크기 조정 함수
def resize_image(image, size):
    image = image.resize(size)
    return np.array(image)

# 이미지 회전 함수 - 짤림 방지를 위해 expand=True 추가
def rotate_image(image, angle):
    return image.rotate(angle, expand=True, resample=Image.BICUBIC)

# 이미지 좌우 반전 함수
def flip_image_horizontal(image):
    return ImageOps.mirror(image)

# 이미지 밝기 조정 함수
def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# 이미지 대비 조정 함수
def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

# 이미지 색상 조정 함수
def adjust_color(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

# 이미지 선명도 조정 함수
def adjust_sharpness(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

# SSIM 비교 함수
def compare_ssim(img1, img2):
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return float(score * 100)  # float32를 일반 float으로 변환

# PSNR 비교 함수 (MAX_PSNR을 기준으로 백분율 계산)
def compare_psnr(img1, img2):
    mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
    if mse == 0:
        return 100.0
    psnr = 20 * log10(255.0 / np.sqrt(mse))
    # PSNR을 백분율로 변환 (50dB를 100%로 가정)
    MAX_PSNR = 50.0
    percentage = min(psnr / MAX_PSNR * 100, 100)
    return float(percentage)  # numpy 타입을 일반 float으로 변환

# VGG16 기반 Cosine 유사도 비교 함수 (PyTorch 사용)
def compare_vgg_cosine(img1, img2):
    # Load pretrained model
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    # Remove the classifier to get features only
    model = torch.nn.Sequential(*list(model.children())[:-1])
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process images
    img1_tensor = preprocess(img1).unsqueeze(0)
    img2_tensor = preprocess(img2).unsqueeze(0)
    
    # Get features
    with torch.no_grad():
        feat1 = model(img1_tensor).flatten().numpy()
        feat2 = model(img2_tensor).flatten().numpy()
    
    return float((1 - cosine(feat1, feat2)) * 100)  # numpy 타입을 일반 float으로 변환

# 유사도 측정 방식에 대한 설명 함수
def show_metric_explanation():
    st.markdown("## 💡 유사도 측정 방식 설명")
    
    st.markdown("### 1. SSIM (Structural Similarity Index Measure)")
    st.markdown("""
    SSIM은 이미지의 구조적 유사성을 측정하는 지표입니다. 인간의 시각 시스템이 이미지의 구조적 정보에 민감하다는 점에 착안해 개발되었습니다.
    
    - **범위**: 0% (완전히 다름) ~ 100% (동일함)
    - **특징**: 밝기, 대비, 구조의 변화를 고려하여 계산
    - **활용**: 이미지 압축, 화질 평가 등에 주로 사용
    """)
    
    st.markdown("### 2. PSNR (Peak Signal-to-Noise Ratio)")
    st.markdown("""
    PSNR은 원본 이미지와 처리된 이미지 간의 픽셀 차이를 기반으로 한 품질 측정 지표입니다. 두 이미지 간의 '오차'를 측정합니다.
    
    - **원리**: MSE(평균 제곱 오차)를 기반으로 계산
    - **단위**: dB (데시벨) - 높을수록 유사도가 높음
    - **특징**: 픽셀 단위의 차이를 정량적으로 표현
    - **한계**: 인간의 시각적 인식과 항상 일치하지 않음
    """)
    
    st.markdown("### 3. VGG16 기반 코사인 유사도")
    st.markdown("""
    딥러닝 모델(VGG16)을 사용하여 이미지의 고수준 특징을 추출한 후, 그 특징 벡터 간의 코사인 유사도를 계산합니다.
    
    - **원리**: 사전 학습된 CNN 모델이 인식하는 이미지 특징의 유사성 측정
    - **범위**: 0% (완전히 다름) ~ 100% (동일함)
    - **특징**: 색상, 질감, 물체 등 이미지의 의미적 내용 비교 가능
    - **장점**: 인간의 시각적 인식과 더 유사한 결과를 제공하는 경향이 있음
    """)

# 슬라이더와 에디트 박스 조합 컴포넌트
def slider_with_input(label, min_val, max_val, default_val, step, key):
    col1, col2 = st.columns([7, 3])
    with col1:
        slider_value = st.slider(label, min_val, max_val, default_val, step, key=f"slider_{key}")
    with col2:
        input_value = st.number_input("", min_val, max_val, slider_value, step, key=f"input_{key}", label_visibility="collapsed")
        
    # 슬라이더와 입력 값 동기화
    if input_value != slider_value:
        return input_value
    return slider_value

# Streamlit UI
st.title("이미지 유사도 비교 도구")

# 사이드바에 이미지 크기 및 회전 옵션 추가
st.sidebar.header("이미지 설정")


# 이미지 크기 선택을 직접 입력 가능하도록 변경
st.sidebar.subheader("이미지 크기 설정")
size_option = st.sidebar.radio("크기 설정 방식", ["기본 크기", "직접 입력"])

if size_option == "기본 크기":
    image_size = st.sidebar.selectbox(
        "이미지 크기",
        options=[(224, 224), (256, 256), (384, 384), (512, 512)],
        format_func=lambda x: f"{x[0]}x{x[1]}"
    )
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        width = st.number_input("너비", min_value=32, max_value=1024, value=256, step=16)
    with col2:
        height = st.number_input("높이", min_value=32, max_value=1024, value=256, step=16)
    image_size = (width, height)

# 유사도 측정 방식 설명 표시 여부
show_explanation = st.sidebar.checkbox("유사도 측정 방식 설명 보기", value=False)

if show_explanation:
    show_metric_explanation()

# 이미지 업로드 받기
st.markdown("## 📸 이미지 업로드")
img1 = st.file_uploader("실제 사진 업로드", type=["jpg", "png", "jpeg"])
img2 = st.file_uploader("AI 생성 사진 업로드", type=["jpg", "png", "jpeg"])

# 두 이미지가 모두 업로드된 경우 비교 진행
if img1 and img2:
    image1 = Image.open(img1).convert("RGB")
    image2 = Image.open(img2).convert("RGB")
    
    # 이미지 원본 정보 표시
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"실제 사진 크기: {image1.width} x {image1.height}")
    with col2:
        st.write(f"AI 생성 사진 크기: {image2.width} x {image2.height}")
    
    # 이미지 편집 기능 추가
    st.markdown("## 🖌️ 이미지 편집")
    
    # 탭으로 각 이미지 편집 섹션 분리
    tab1, tab2 = st.tabs(["실제 사진 편집", "AI 생성 사진 편집"])
    
    with tab1:
        st.subheader("실제 사진 편집")
        
        # 배경 제거 옵션
        remove_bg1 = st.checkbox("배경 제거", key="remove_bg1")
        if remove_bg1:
            try:
                with st.spinner("배경을 제거하는 중..."):
                    image1 = remove_background(image1)
                st.success("배경이 제거되었습니다!")
            except Exception as e:
                st.error(f"배경 제거 중 오류가 발생했습니다: {str(e)}")
        
        # 좌우반전 옵션
        flip1 = st.checkbox("좌우반전", key="flip1")
        if flip1:
            image1 = flip_image_horizontal(image1)
        
        # 회전 옵션 (슬라이더 + 에디트 박스)
        rotation1 = slider_with_input("회전 각도", -180, 180, 0, 1, "rot1")
        if rotation1 != 0:
            image1 = rotate_image(image1, rotation1)
        
        # 밝기 조정 (슬라이더 + 에디트 박스)
        brightness1 = slider_with_input("밝기 조정", 0.0, 3.0, 1.0, 0.05, "bright1")
        if brightness1 != 1.0:
            image1 = adjust_brightness(image1, brightness1)
        
        # 대비 조정 (슬라이더 + 에디트 박스)
        contrast1 = slider_with_input("대비 조정", 0.0, 3.0, 1.0, 0.05, "contrast1")
        if contrast1 != 1.0:
            image1 = adjust_contrast(image1, contrast1)
        
        # 색상 조정 (슬라이더 + 에디트 박스)
        color1 = slider_with_input("색상 조정", 0.0, 3.0, 1.0, 0.05, "color1")
        if color1 != 1.0:
            image1 = adjust_color(image1, color1)
        
        # 선명도 조정 (슬라이더 + 에디트 박스)
        sharpness1 = slider_with_input("선명도 조정", 0.0, 3.0, 1.0, 0.05, "sharp1")
        if sharpness1 != 1.0:
            image1 = adjust_sharpness(image1, sharpness1)
            
        # 조정값 일괄 적용 섹션
        st.write("---")
        st.subheader("조정값 직접 입력")
        col1, col2, col3 = st.columns(3)
        with col1:
            custom_rotation1 = st.number_input("회전 각도 값", -180, 180, rotation1, 1, key="custom_rot1")
        with col2:
            custom_brightness1 = st.number_input("밝기 값", 0.0, 3.0, brightness1, 0.05, key="custom_bright1")
        with col3:
            custom_contrast1 = st.number_input("대비 값", 0.0, 3.0, contrast1, 0.05, key="custom_contrast1")
        
        col1, col2 = st.columns(2)
        with col1:
            custom_color1 = st.number_input("색상 값", 0.0, 3.0, color1, 0.05, key="custom_color1")
        with col2:
            custom_sharpness1 = st.number_input("선명도 값", 0.0, 3.0, sharpness1, 0.05, key="custom_sharp1")
        
        if st.button("조정값 일괄 적용", key="apply_custom1"):
            # 조정값 적용
            image1_original = Image.open(img1).convert("RGB")
            
            # 배경 제거 (일괄 적용 시에도 배경 제거 옵션이 켜져 있으면 적용)
            if remove_bg1:
                try:
                    with st.spinner("배경을 제거하는 중..."):
                        image1_original = remove_background(image1_original)
                except Exception as e:
                    st.error(f"배경 제거 중 오류가 발생했습니다: {str(e)}")
            
            # 좌우반전
            if flip1:
                image1_original = flip_image_horizontal(image1_original)
                
            # 다른 조정 적용
            image1 = rotate_image(image1_original, custom_rotation1)
            image1 = adjust_brightness(image1, custom_brightness1)
            image1 = adjust_contrast(image1, custom_contrast1)
            image1 = adjust_color(image1, custom_color1)
            image1 = adjust_sharpness(image1, custom_sharpness1)
            
            st.success("조정값이 일괄 적용되었습니다.")
            
        st.image(image1, caption="편집된 실제 사진", use_column_width=True)
    
    with tab2:
        st.subheader("AI 생성 사진 편집")
        
        # 배경 제거 옵션
        remove_bg2 = st.checkbox("배경 제거", key="remove_bg2")
        if remove_bg2:
            try:
                with st.spinner("배경을 제거하는 중..."):
                    image2 = remove_background(image2)
                st.success("배경이 제거되었습니다!")
            except Exception as e:
                st.error(f"배경 제거 중 오류가 발생했습니다: {str(e)}")
        
        # 좌우반전 옵션
        flip2 = st.checkbox("좌우반전", key="flip2")
        if flip2:
            image2 = flip_image_horizontal(image2)
        
        # 회전 옵션 (슬라이더 + 에디트 박스)
        rotation2 = slider_with_input("회전 각도", -180, 180, 0, 1, "rot2")
        if rotation2 != 0:
            image2 = rotate_image(image2, rotation2)
        
        # 밝기 조정 (슬라이더 + 에디트 박스)
        brightness2 = slider_with_input("밝기 조정", 0.0, 3.0, 1.0, 0.05, "bright2")
        if brightness2 != 1.0:
            image2 = adjust_brightness(image2, brightness2)
        
        # 대비 조정 (슬라이더 + 에디트 박스)
        contrast2 = slider_with_input("대비 조정", 0.0, 3.0, 1.0, 0.05, "contrast2")
        if contrast2 != 1.0:
            image2 = adjust_contrast(image2, contrast2)
        
        # 색상 조정 (슬라이더 + 에디트 박스)
        color2 = slider_with_input("색상 조정", 0.0, 3.0, 1.0, 0.05, "color2")
        if color2 != 1.0:
            image2 = adjust_color(image2, color2)
        
        # 선명도 조정 (슬라이더 + 에디트 박스)
        sharpness2 = slider_with_input("선명도 조정", 0.0, 3.0, 1.0, 0.05, "sharp2")
        if sharpness2 != 1.0:
            image2 = adjust_sharpness(image2, sharpness2)
            
        # 조정값 일괄 적용 섹션
        st.write("---")
        st.subheader("조정값 직접 입력")
        col1, col2, col3 = st.columns(3)
        with col1:
            custom_rotation2 = st.number_input("회전 각도 값", -180, 180, rotation2, 1, key="custom_rot2")
        with col2:
            custom_brightness2 = st.number_input("밝기 값", 0.0, 3.0, brightness2, 0.05, key="custom_bright2")
        with col3:
            custom_contrast2 = st.number_input("대비 값", 0.0, 3.0, contrast2, 0.05, key="custom_contrast2")
        
        col1, col2 = st.columns(2)
        with col1:
            custom_color2 = st.number_input("색상 값", 0.0, 3.0, color2, 0.05, key="custom_color2")
        with col2:
            custom_sharpness2 = st.number_input("선명도 값", 0.0, 3.0, sharpness2, 0.05, key="custom_sharp2")
        
        if st.button("조정값 일괄 적용", key="apply_custom2"):
            # 조정값 적용
            image2_original = Image.open(img2).convert("RGB")
            
            # 배경 제거 (일괄 적용 시에도 배경 제거 옵션이 켜져 있으면 적용)
            if remove_bg2:
                try:
                    with st.spinner("배경을 제거하는 중..."):
                        image2_original = remove_background(image2_original)
                except Exception as e:
                    st.error(f"배경 제거 중 오류가 발생했습니다: {str(e)}")
            
            # 좌우반전
            if flip2:
                image2_original = flip_image_horizontal(image2_original)
                
            # 다른 조정 적용
            image2 = rotate_image(image2_original, custom_rotation2)
            image2 = adjust_brightness(image2, custom_brightness2)
            image2 = adjust_contrast(image2, custom_contrast2)
            image2 = adjust_color(image2, custom_color2)
            image2 = adjust_sharpness(image2, custom_sharpness2)
            
            st.success("조정값이 일괄 적용되었습니다.")
            
        st.image(image2, caption="편집된 AI 생성 사진", use_column_width=True)
    
    # 이미지 비교 섹션
    st.markdown("## 🔍 이미지 비교")
    
    # 두 이미지 비교 보기
    st.image([image1, image2], caption=["편집된 실제 사진", "편집된 AI 생성 사진"], width=300)

    # 이미지 리사이즈
    resized1 = resize_image(image1, image_size)
    resized2 = resize_image(image2, image_size)

    # 유사도 계산 버튼
    if st.button("유사도 계산하기"):
        with st.spinner("유사도를 계산 중입니다..."):
            # 고유 ID 생성
            result_id = str(uuid.uuid4())
            
            # 이미지 저장
            real_image_filename = f"Result/real_{result_id}.png"
            ai_image_filename = f"Result/ai_{result_id}.png"
            
            # PIL 이미지 저장
            image1.save(real_image_filename)
            image2.save(ai_image_filename)
            
            # 유사도 계산
            ssim_score = compare_ssim(resized1, resized2)
            psnr_score = compare_psnr(resized1, resized2)
            vgg_score = compare_vgg_cosine(image1, image2)
            avg_score = (ssim_score + psnr_score + vgg_score) / 3
            
            # 결과를 DB에 저장
            saved_id = save_results(
                real_image_filename, 
                ai_image_filename, 
                ssim_score, 
                psnr_score, 
                vgg_score, 
                avg_score
            )

            # 결과 출력
            st.markdown("## 📊 유사도 비교 결과")
            
            # 표 형식으로 결과 표시
            results_df = {
                "비교 방식": ["SSIM", "PSNR", "VGG16 기반 Cosine 유사도"],
                "유사도 점수 (%)": [f"{ssim_score:.2f}%", f"{psnr_score:.2f}%", f"{vgg_score:.2f}%"]
            }
            st.table(results_df)
            
            # 시각적 게이지로 결과 표시
            st.markdown("### 시각적 유사도 표시")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**SSIM**: {ssim_score:.2f}%")
                st.progress(min(ssim_score/100, 1.0))
                if ssim_score > 80:
                    st.success("매우 유사한 구조")
                elif ssim_score > 60:
                    st.info("유사한 구조")
                else:
                    st.warning("구조적 차이가 큼")
                
            with col2:
                st.markdown(f"**PSNR**: {psnr_score:.2f}%")
                st.progress(min(psnr_score/100, 1.0))
                if psnr_score > 80:
                    st.success("매우 낮은 오차")
                elif psnr_score > 60:
                    st.info("적절한 오차 수준")
                else:
                    st.warning("오차가 큼")
                
            with col3:
                st.markdown(f"**VGG16**: {vgg_score:.2f}%")
                st.progress(min(vgg_score/100, 1.0))
                if vgg_score > 80:
                    st.success("매우 유사한 특징")
                elif vgg_score > 60:
                    st.info("유사한 특징")
                else:
                    st.warning("특징 차이가 큼")
                
            # 종합 분석 결과
            st.markdown(f"### 종합 유사도: {avg_score:.2f}%")
            
            if avg_score > 80:
                st.success("두 이미지가 매우 유사합니다.")
            elif avg_score > 60:
                st.info("두 이미지가 어느 정도 유사합니다.")
            else:
                st.warning("두 이미지의 유사도가 낮습니다.")
else:
    st.info("이미지 비교를 시작하려면 두 장의 이미지를 업로드해주세요.")
