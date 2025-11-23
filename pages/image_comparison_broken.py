import streamlit as st
import uuid
from PIL import Image
import sys
import os

# 현재 디렉토리를 상위 폴더로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 상대 경로로 import (수정된 부분)
from image_processing import (
    resize_image, rotate_image, flip_image_horizontal, remove_background,
    adjust_brightness, adjust_contrast, adjust_color, adjust_sharpness, crop_image
)
from similarity_metrics import compare_ssim, compare_psnr, compare_vgg_cosine
from ui_components import slider_with_input, display_similarity_results, integrated_crop_interface, download_cropped_image
from db_utils import save_results
from crop_manager import CropManager, display_crop_gallery, crop_comparison_interface
from color_extractor import display_color_analysis_ui

def app():  # 이 함수가 올바르게 정의되어야 합니다
    st.title("이미지 유사도 비교 도구")
    
    # 이미지 업로드 받기
    st.markdown("## 📸 이미지 업로드")
    img1 = st.file_uploader("실제 사진 업로드", type=["jpg", "png", "jpeg"])
    img2 = st.file_uploader("AI 생성 사진 업로드", type=["jpg", "png", "jpeg"])

    # 크롭 비교만을 위한 별도 섹션 추가
    st.markdown("---")
    st.markdown("## ✂️ 크롭 이미지 비교")
    
    with st.expander("크롭된 이미지 유사도 비교", expanded=False):
        st.markdown("### 📤 크롭된 이미지 업로드하여 비교")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 첫 번째 크롭 이미지")
            crop_img1 = st.file_uploader("첫 번째 크롭 이미지", type=["jpg", "png", "jpeg"], key="crop_compare1")
            if crop_img1:
                crop_image1 = Image.open(crop_img1).convert("RGB")
                st.image(crop_image1, caption=f"크기: {crop_image1.width}×{crop_image1.height}", use_column_width=True)
        
        with col2:
            st.markdown("#### 두 번째 크롭 이미지")
            crop_img2 = st.file_uploader("두 번째 크롭 이미지", type=["jpg", "png", "jpeg"], key="crop_compare2")
            if crop_img2:
                crop_image2 = Image.open(crop_img2).convert("RGB")
                st.image(crop_image2, caption=f"크기: {crop_image2.width}×{crop_image2.height}", use_column_width=True)
        
        # 크롭 이미지 비교
        if crop_img1 and crop_img2:
            if st.button("🔍 크롭 이미지 유사도 계산"):
                with st.spinner("크롭된 이미지의 유사도를 계산 중입니다..."):
                    try:
                        # 이미지 크기 조정
                        crop_size = (256, 256)
                        resized_crop1 = resize_image(crop_image1, crop_size)
                        resized_crop2 = resize_image(crop_image2, crop_size)
                        
                        # 유사도 계산
                        crop_ssim = compare_ssim(resized_crop1, resized_crop2)
                        crop_psnr = compare_psnr(resized_crop1, resized_crop2)
                        crop_vgg = compare_vgg_cosine(crop_image1, crop_image2)
                        crop_avg = (crop_ssim + crop_psnr + crop_vgg) / 3
                        
                        # 결과 표시
                        st.markdown("#### 🎯 크롭 이미지 유사도 결과")
                        display_similarity_results(crop_ssim, crop_psnr, crop_vgg, crop_avg)
                        
                        # 결과 저장
                        crop_id = str(uuid.uuid4())
                        crop1_filename = f"Result/crop1_{crop_id}.png"
                        crop2_filename = f"Result/crop2_{crop_id}.png"
                        
                        crop_image1.save(crop1_filename)
                        crop_image2.save(crop2_filename)
                        
                        saved_crop_id = save_results(
                            crop1_filename, crop2_filename,
                            crop_ssim, crop_psnr, crop_vgg, crop_avg
                        )
                        
                        st.success(f"크롭 이미지 비교 결과가 저장되었습니다. ID: {saved_crop_id}")
                        
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {str(e)}")

    # 색상 분석 섹션 추가
    st.markdown("---")
    st.markdown("## 🎨 RGB 색상 분석")
    
    with st.expander("이미지 색상 분석", expanded=False):
        st.markdown("### 📤 색상 분석할 이미지 업로드")
        
        color_img = st.file_uploader("색상 분석할 이미지", type=["jpg", "png", "jpeg"], key="color_analysis_img")
        if color_img:
            color_image = Image.open(color_img).convert("RGB")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(color_image, caption=f"분석 대상 이미지 (크기: {color_image.width}×{color_image.height})", use_column_width=True)
            with col2:
                st.write("**이미지 정보**")
                st.write(f"• 크기: {color_image.width} × {color_image.height}")
                st.write(f"• 총 픽셀: {color_image.width * color_image.height:,}")
                st.write(f"• 포맷: RGB")
            
            # 색상 분석 UI 표시
            display_color_analysis_ui(color_image, "main")

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
            
            # 편집 설정 섹션
            st.markdown("### 🎛️ 편집 설정")
            
            # 배경 제거 옵션 (라디오 버튼으로 변경)
            bg_option1 = st.radio(
                "배경 설정", 
                ["원본 유지", "배경 제거"], 
                key="bg_option1",
                horizontal=True
            )
            
            # 배경 제거 적용
            if bg_option1 == "배경 제거":
                try:
                    with st.spinner("배경을 제거하는 중..."):
                        image1 = remove_background(image1)
                    st.success("배경이 제거되었습니다!")
                except Exception as e:
                    st.error(f"배경 제거 중 오류가 발생했습니다: {str(e)}")
            
            # 미리보기 섹션
            st.markdown("### 👀 현재 이미지 미리보기")
            st.image(image1, caption=f"{'배경 제거된' if bg_option1 == '배경 제거' else '원본'} 실제 사진", use_column_width=True)
            
            # 추가 편집 옵션들
            st.markdown("### ✨ 추가 편집")
            
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
            
            # 최종 편집된 이미지 표시
            st.markdown("### 🎯 최종 편집 결과")
            st.image(image1, caption="최종 편집된 실제 사진", use_column_width=True)
            
            # 이미지 크롭 기능 추가
            with st.expander("✂️ 이미지 자르기", expanded=False):
                left1, top1, right1, bottom1 = integrated_crop_interface(image1, "real")
                
                # 실시간 크롭 미리보기
                if left1 < right1 and top1 < bottom1:
                    preview_crop1 = crop_image(image1, left1, top1, right1, bottom1)
                    st.markdown("#### 🔍 실시간 크롭 미리보기")
                    st.image(preview_crop1, caption=f"크롭 영역: {right1-left1}x{bottom1-top1} 픽셀", width=300)
                    
                    # 크롭된 이미지 다운로드
                    download_cropped_image(preview_crop1, "cropped_real_image.png")
            
            # 색상 분석 기능 추가
            with st.expander("🎨 색상 분석", expanded=False):
                st.markdown("#### 실제 사진 색상 분석")
                display_color_analysis_ui(image1, "real")
        
        with tab2:
            st.subheader("AI 생성 사진 편집")
            
            # 편집 설정 섹션
            st.markdown("### 🎛️ 편집 설정")
            
            # 배경 제거 옵션 (라디오 버튼으로 변경)
            bg_option2 = st.radio(
                "배경 설정", 
                ["원본 유지", "배경 제거"], 
                key="bg_option2",
                horizontal=True
            )
            
            # 배경 제거 적용
            if bg_option2 == "배경 제거":
                try:
                    with st.spinner("배경을 제거하는 중..."):
                        image2 = remove_background(image2)
                    st.success("배경이 제거되었습니다!")
                except Exception as e:
                    st.error(f"배경 제거 중 오류가 발생했습니다: {str(e)}")
            
            # 미리보기 섹션
            st.markdown("### 👀 현재 이미지 미리보기")
            st.image(image2, caption=f"{'배경 제거된' if bg_option2 == '배경 제거' else '원본'} AI 생성 사진", use_column_width=True)
            
            # 추가 편집 옵션들
            st.markdown("### ✨ 추가 편집")
            
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
            
            # 최종 편집된 이미지 표시
            st.markdown("### 🎯 최종 편집 결과")
            st.image(image2, caption="최종 편집된 AI 생성 사진", use_column_width=True)
            
            # 이미지 크롭 기능 추가
            with st.expander("✂️ 이미지 자르기", expanded=False):
                left2, top2, right2, bottom2 = integrated_crop_interface(image2, "ai")
                
                # 실시간 크롭 미리보기
                if left2 < right2 and top2 < bottom2:
                    preview_crop2 = crop_image(image2, left2, top2, right2, bottom2)
                    st.markdown("#### 🔍 실시간 크롭 미리보기")
                    st.image(preview_crop2, caption=f"크롭 영역: {right2-left2}x{bottom2-top2} 픽셀", width=300)
                    
                    # 크롭된 이미지 다운로드
                    download_cropped_image(preview_crop2, "cropped_ai_image.png")
            
            # 색상 분석 기능 추가
            with st.expander("🎨 색상 분석", expanded=False):
                st.markdown("#### AI 생성 사진 색상 분석")
                display_color_analysis_ui(image2, "ai")
        
        # 이미지 비교 섹션
        st.markdown("## 🔍 이미지 비교")
        
        # 두 이미지 비교 보기
        st.image([image1, image2], caption=["편집된 실제 사진", "편집된 AI 생성 사진"], width=300)

        # 이미지 크기 설정에서 이미지 리사이즈
        image_size = st.session_state.get('image_size', (256, 256))
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

                # 결과 표시
                display_similarity_results(ssim_score, psnr_score, vgg_score, avg_score)
                st.success(f"결과가 저장되었습니다. 결과 ID: {saved_id}")
    else:
        st.info("이미지 비교를 시작하려면 두 장의 이미지를 업로드해주세요.")