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
    adjust_brightness, adjust_contrast, adjust_color, adjust_sharpness, crop_image,
    remove_background_with_mask, safe_image_open
)
from similarity_metrics import compare_ssim, compare_psnr, compare_vgg_cosine, compare_lpips, compare_lpips_ensemble
from ui_components import slider_with_input, display_similarity_results, integrated_crop_interface, download_cropped_image, download_background_removed_image, regional_background_removal_interface, display_enhanced_similarity_results
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
                try:
                    crop_image1 = safe_image_open(crop_img1).convert("RGB")
                    st.image(crop_image1, caption=f"크기: {crop_image1.width}×{crop_image1.height}", use_column_width=True)
                except Exception as e:
                    st.error(f"이미지 로드 실패: {str(e)}")
                    crop_img1 = None
        
        with col2:
            st.markdown("#### 두 번째 크롭 이미지")
            crop_img2 = st.file_uploader("두 번째 크롭 이미지", type=["jpg", "png", "jpeg"], key="crop_compare2")
            if crop_img2:
                try:
                    crop_image2 = safe_image_open(crop_img2).convert("RGB")
                    st.image(crop_image2, caption=f"크기: {crop_image2.width}×{crop_image2.height}", use_column_width=True)
                except Exception as e:
                    st.error(f"이미지 로드 실패: {str(e)}")
                    crop_img2 = None
        
        # 크롭 이미지 비교
        if crop_img1 and crop_img2:
            if st.button("🔍 크롭 이미지 유사도 계산"):
                with st.spinner("크롭된 이미지의 유사도를 계산 중입니다..."):
                    try:
                        # 이미지 크기 조정
                        crop_size = (256, 256)
                        resized_crop1 = resize_image(crop_image1, crop_size)
                        resized_crop2 = resize_image(crop_image2, crop_size)
                        
                        # 유사도 계산 (모든 메트릭 포함)
                        crop_scores = {}
                        crop_scores['SSIM'] = compare_ssim(resized_crop1, resized_crop2)
                        crop_scores['PSNR'] = compare_psnr(resized_crop1, resized_crop2)
                        crop_scores['VGG_Cosine'] = compare_vgg_cosine(crop_image1, crop_image2)
                        crop_scores['LPIPS'] = compare_lpips(crop_image1, crop_image2, net='alex')
                        
                        # 평균 점수 계산 (LPIPS 정규화 포함)
                        normalized_crop_scores = []
                        for metric, score in crop_scores.items():
                            if metric == 'LPIPS':
                                # LPIPS 거리를 유사도로 변환 (0-1 -> 100-0)
                                normalized_crop_scores.append((1 - score) * 100)
                            else:
                                normalized_crop_scores.append(score)
                        
                        crop_avg = sum(normalized_crop_scores) / len(normalized_crop_scores) if normalized_crop_scores else 0.0
                        
                        # 결과 표시 - 향상된 버전 사용
                        st.markdown("#### 🎯 크롭 이미지 유사도 결과")
                        display_enhanced_similarity_results(crop_scores, crop_avg)
                        
                        # 결과 저장
                        crop_id = str(uuid.uuid4())
                        crop1_filename = f"Result/crop1_{crop_id}.png"
                        crop2_filename = f"Result/crop2_{crop_id}.png"
                        
                        crop_image1.save(crop1_filename)
                        crop_image2.save(crop2_filename)
                        
                        # 기존 형식의 값들도 설정 (DB 호환성)
                        crop_ssim = crop_scores['SSIM']
                        crop_psnr = crop_scores['PSNR'] 
                        crop_vgg = crop_scores['VGG_Cosine']
                        
                        # 기존 형식의 값들도 설정 (DB 호환성)
                        crop_ssim = crop_scores['SSIM']
                        crop_psnr = crop_scores['PSNR'] 
                        crop_vgg = crop_scores['VGG_Cosine']
                        
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
            try:
                color_image = safe_image_open(color_img).convert("RGB")
                
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
                
            except Exception as e:
                st.error(f"이미지 로드 실패: {str(e)}")

    # 두 이미지가 모두 업로드된 경우 비교 진행
    if img1 and img2:
        try:
            image1 = safe_image_open(img1).convert("RGB")
            image2 = safe_image_open(img2).convert("RGB")
        except Exception as e:
            st.error(f"이미지 로드 실패: {str(e)}")
            st.stop()
        
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
                ["원본 유지", "전체 배경 제거", "영역 지정 배경 제거"], 
                key="bg_option1",
                horizontal=True
            )
            
            # 배경 제거 적용
            if bg_option1 == "전체 배경 제거":
                try:
                    with st.spinner("배경을 제거하는 중..."):
                        image1 = remove_background(image1)
                    st.success("배경이 제거되었습니다!")
                    
                    # 배경 제거된 이미지 다운로드 버튼 추가
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        download_background_removed_image(image1, "real_photo_bg_removed.png")
                        
                except Exception as e:
                    st.error(f"배경 제거 중 오류가 발생했습니다: {str(e)}")
            
            elif bg_option1 == "영역 지정 배경 제거":
                # 영역 지정 배경 제거 인터페이스
                region_config = regional_background_removal_interface(image1, "real_region_bg")
                
                # 배경 제거 실행 버튼
                if st.button("🎯 영역 지정 배경 제거 실행", key="exec_region_bg1", type="primary"):
                    try:
                        with st.spinner("선택된 영역의 배경을 제거하는 중..."):
                            if region_config['mask_type'] == "전체 이미지":
                                image1 = remove_background(image1, region_config['threshold'])
                            elif region_config['mask_type'] == "사각형 영역":
                                image1 = remove_background_with_mask(
                                    image1, 
                                    mask_coords=region_config['mask_coords'], 
                                    mask_type="rectangle",
                                    threshold=region_config['threshold'],
                                    invert_mask=region_config['invert_mask']
                                )
                            elif region_config['mask_type'] == "다각형 영역":
                                image1 = remove_background_with_mask(
                                    image1, 
                                    mask_coords=region_config['mask_coords'], 
                                    mask_type="polygon",
                                    threshold=region_config['threshold'],
                                    invert_mask=region_config['invert_mask']
                                )
                        
                        st.success("✅ 영역 지정 배경 제거가 완료되었습니다!")
                        
                        # 배경 제거된 이미지 다운로드 버튼
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            download_background_removed_image(image1, "real_photo_regional_bg_removed.png")
                            
                    except Exception as e:
                        st.error(f"영역 지정 배경 제거 중 오류가 발생했습니다: {str(e)}")
                        st.error(f"상세 오류: {type(e).__name__}: {e}")
            
            # 미리보기 섹션
            st.markdown("### 👀 현재 이미지 미리보기")
            bg_status1 = "원본" if bg_option1 == "원본 유지" else ("전체 배경 제거됨" if bg_option1 == "전체 배경 제거" else "영역 배경 제거됨")
            st.image(image1, caption=f"{bg_status1} 실제 사진", use_column_width=True)
            
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
                ["원본 유지", "전체 배경 제거", "영역 지정 배경 제거"], 
                key="bg_option2",
                horizontal=True
            )
            
            # 배경 제거 적용
            if bg_option2 == "전체 배경 제거":
                try:
                    with st.spinner("배경을 제거하는 중..."):
                        image2 = remove_background(image2)
                    st.success("배경이 제거되었습니다!")
                    
                    # 배경 제거된 이미지 다운로드 버튼 추가
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        download_background_removed_image(image2, "ai_photo_bg_removed.png")
                        
                except Exception as e:
                    st.error(f"배경 제거 중 오류가 발생했습니다: {str(e)}")
            
            elif bg_option2 == "영역 지정 배경 제거":
                # 영역 지정 배경 제거 인터페이스
                region_config2 = regional_background_removal_interface(image2, "ai_region_bg")
                
                # 배경 제거 실행 버튼
                if st.button("🎯 영역 지정 배경 제거 실행", key="exec_region_bg2", type="primary"):
                    try:
                        with st.spinner("선택된 영역의 배경을 제거하는 중..."):
                            if region_config2['mask_type'] == "전체 이미지":
                                image2 = remove_background(image2, region_config2['threshold'])
                            elif region_config2['mask_type'] == "사각형 영역":
                                image2 = remove_background_with_mask(
                                    image2, 
                                    mask_coords=region_config2['mask_coords'], 
                                    mask_type="rectangle",
                                    threshold=region_config2['threshold'],
                                    invert_mask=region_config2['invert_mask']
                                )
                            elif region_config2['mask_type'] == "다각형 영역":
                                image2 = remove_background_with_mask(
                                    image2, 
                                    mask_coords=region_config2['mask_coords'], 
                                    mask_type="polygon",
                                    threshold=region_config2['threshold'],
                                    invert_mask=region_config2['invert_mask']
                                )
                        
                        st.success("✅ 영역 지정 배경 제거가 완료되었습니다!")
                        
                        # 배경 제거된 이미지 다운로드 버튼
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            download_background_removed_image(image2, "ai_photo_regional_bg_removed.png")
                            
                    except Exception as e:
                        st.error(f"영역 지정 배경 제거 중 오류가 발생했습니다: {str(e)}")
                        st.error(f"상세 오류: {type(e).__name__}: {e}")
            
            # 미리보기 섹션
            st.markdown("### 👀 현재 이미지 미리보기")
            bg_status2 = "원본" if bg_option2 == "원본 유지" else ("전체 배경 제거됨" if bg_option2 == "전체 배경 제거" else "영역 배경 제거됨")
            st.image(image2, caption=f"{bg_status2} AI 생성 사진", use_column_width=True)
            
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
        
        # 배경 제거 상태 확인
        bg_removed_real = bg_option1 != "원본 유지"
        bg_removed_ai = bg_option2 != "원본 유지"
        
        # 비교 이미지 초기화 (기본값은 편집된 이미지)
        comparison_img1 = image1
        comparison_img2 = image2
        
        # 이미지 비교 섹션
        st.markdown("## 🔍 이미지 비교")
        
        # 두 이미지 비교 보기
        st.image([comparison_img1, comparison_img2], 
                caption=["비교용 실제 사진", "비교용 AI 생성 사진"], width=300)

        # 이미지 크기 설정에서 이미지 리사이즈
        image_size = st.session_state.get('image_size', (256, 256))
        resized1 = resize_image(comparison_img1, image_size)
        resized2 = resize_image(comparison_img2, image_size)

        # 유사도 계산 설정
        st.markdown("### ⚙️ 유사도 계산 설정")
        
        # 배경 제거 이미지 사용 옵션
        if bg_removed_real or bg_removed_ai:
            st.markdown("#### 🎭 배경 처리 옵션")
            st.info("💡 배경이 제거된 이미지가 감지되었습니다. 유사도 비교에 사용할 이미지를 선택하세요.")
            
            compare_bg_removed = st.radio(
                "유사도 비교 대상 선택",
                ["편집된 이미지 (배경 처리 포함)", "원본 업로드 이미지"],
                help="편집된 이미지: 배경 제거 및 기타 편집이 적용된 이미지\n원본 업로드 이미지: 최초 업로드한 원본 이미지"
            )
            
            if compare_bg_removed == "편집된 이미지 (배경 처리 포함)":
                comparison_img1 = image1  # 편집된 이미지
                comparison_img2 = image2
                st.success("✅ 배경 처리된 편집 이미지로 유사도 비교를 진행합니다.")
            else:
                # 원본 이미지 사용
                try:
                    comparison_img1 = safe_image_open(img1).convert("RGB")
                    comparison_img2 = safe_image_open(img2).convert("RGB")
                    st.info("ℹ️ 원본 업로드 이미지로 유사도 비교를 진행합니다.")
                except:
                    comparison_img1 = image1
                    comparison_img2 = image2
                    st.warning("⚠️ 원본 이미지 로드 실패, 편집된 이미지를 사용합니다.")
        else:
            comparison_img1 = image1
            comparison_img2 = image2
            st.info("ℹ️ 편집된 이미지로 유사도 비교를 진행합니다.")
        
        # 유사도 계산 옵션 선택
        st.markdown("### 📊 유사도 메트릭 선택")
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            use_lpips = st.checkbox("🧠 LPIPS (학습된 지각적 유사도)", value=True, 
                                   help="인간의 시각적 인지와 가장 유사한 고급 유사도 측정")
            use_ssim = st.checkbox("📏 SSIM (구조적 유사도)", value=True)
            use_psnr = st.checkbox("📊 PSNR (신호 대 잡음비)", value=True)
        
        with col_metric2:
            use_vgg = st.checkbox("🧑‍💻 VGG Cosine (딥러닝 기반)", value=True)
            lpips_mode = st.selectbox("🔬 LPIPS 모드",
                                    ["기본 (AlexNet)", "앙상블 (Alex+VGG)"],
                                    help="기본: 빠르고 정확한 결과, 앙상블: 더 정확하지만 속도가 느림")
        
        # 유사도 계산 버튼
        if st.button("유사도 계산하기"):
            # 선택된 메트릭 검증
            if not any([use_lpips, use_ssim, use_psnr, use_vgg]):
                st.error("적어도 하나의 유사도 메트릭을 선택해주세요.")
                return
            
            with st.spinner("유사도를 계산 중입니다..."):
                # 고유 ID 생성
                result_id = str(uuid.uuid4())
                
                # 이미지 저장
                real_image_filename = f"Result/real_{result_id}.png"
                ai_image_filename = f"Result/ai_{result_id}.png"
                
                # 선택된 이미지 저장
                comparison_img1.save(real_image_filename)
                comparison_img2.save(ai_image_filename)
                
                # 선택된 메트릭에 따른 유사도 계산
                scores = {}
                
                if use_ssim:
                    with st.status("📏 SSIM 계산 중..."):
                        scores['SSIM'] = compare_ssim(resized1, resized2)
                
                if use_psnr:
                    with st.status("📊 PSNR 계산 중..."):
                        scores['PSNR'] = compare_psnr(resized1, resized2)
                
                if use_vgg:
                    with st.status("🧑‍💻 VGG Cosine 계산 중..."):
                        scores['VGG_Cosine'] = compare_vgg_cosine(comparison_img1, comparison_img2)
                
                if use_lpips:
                    with st.status("🧠 LPIPS 계산 중..."):
                        if lpips_mode == "앙상블 (Alex+VGG)":
                            scores['LPIPS'] = compare_lpips_ensemble(comparison_img1, comparison_img2)
                        else:
                            scores['LPIPS'] = compare_lpips(comparison_img1, comparison_img2, net='alex')
                
                # 평균 점수 계산 (LPIPS 정규화 포함)
                normalized_scores = []
                for metric, score in scores.items():
                    if metric == 'LPIPS':
                        # LPIPS 거리를 유사도로 변환 (0-1 -> 100-0)
                        normalized_scores.append((1 - score) * 100)
                    else:
                        normalized_scores.append(score)
                
                avg_score = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
                
                # 기존 형식의 값들도 설정 (DB 호환성)
                ssim_score = scores.get('SSIM', 0.0)
                psnr_score = scores.get('PSNR', 0.0) 
                vgg_score = scores.get('VGG_Cosine', 0.0)
                
                # 결과를 DB에 저장
                saved_id = save_results(
                    real_image_filename, 
                    ai_image_filename, 
                    ssim_score, 
                    psnr_score, 
                    vgg_score, 
                    avg_score
                )

                # 결과 표시 - 향상된 버전
                display_enhanced_similarity_results(scores, avg_score)
                st.success(f"결과가 저장되었습니다. 결과 ID: {saved_id}")
    else:
        st.info("이미지 비교를 시작하려면 두 장의 이미지를 업로드해주세요.")