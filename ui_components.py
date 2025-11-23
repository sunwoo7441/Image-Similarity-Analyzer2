import streamlit as st
from PIL import Image
import io
import numpy as np
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

try:
    from streamlit_cropper import st_cropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False

import streamlit as st
from PIL import Image
import io
import numpy as np
try:
    from streamlit_cropper import st_cropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False

import streamlit as st
from PIL import Image
import io
import numpy as np

# 간단한 이미지 크롭 가이드 인터페이스
def visual_crop_guide_interface(image, key_prefix=""):
    """시각적 가이드를 제공하는 크롭 인터페이스"""
    st.markdown("### 🖱️ 시각적 크롭 가이드")
    st.write("📐 아래 정보를 참고하여 수동으로 크롭 영역을 설정하세요.")
    
    # 이미지 정보 표시
    width, height = image.size
    
    # 이미지를 작은 크기로 표시 (참고용)
    display_size = 400
    if width > display_size or height > display_size:
        ratio = min(display_size / width, display_size / height)
        display_width = int(width * ratio)
        display_height = int(height * ratio)
        display_image = image.resize((display_width, display_height))
    else:
        display_image = image
        display_width, display_height = width, height
    
    st.image(display_image, caption=f"참고용 이미지 (실제 크기: {width}×{height}px)", width=display_size)
    
    # 가이드 정보
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        � **이미지 정보**
        - 원본 크기: {width} × {height} 픽셀
        - 가로 중앙: {width//2} 픽셀
        - 세로 중앙: {height//2} 픽셀
        """)
    
    with col2:
        st.info(f"""
        💡 **크롭 팁**
        - 왼쪽 위 모서리가 (0, 0)
        - 오른쪽 아래 모서리가 ({width}, {height})
        - 중앙에서 정사각형: 약 ({width//4}, {height//4}) → ({3*width//4}, {3*height//4})
        """)
    
    return None  # 수동 입력으로 넘어감

# 통합 크롭 인터페이스 (수정됨)
def integrated_crop_interface(image, key_prefix=""):
    """수동 입력과 시각적 가이드를 통합한 크롭 인터페이스"""
    st.markdown("### ✂️ 이미지 자르기")
    
    # 크롭 방식 선택
    crop_method = st.radio(
        "크롭 방식 선택",
        ["📐 정밀 설정", "🖼️ 시각적 가이드"],
        key=f"crop_method_{key_prefix}",
        horizontal=True
    )
    
    if crop_method == "🖼️ 시각적 가이드":
        visual_crop_guide_interface(image, key_prefix)
    
    # 항상 수동 입력 인터페이스 표시
    left, top, right, bottom = image_crop_interface(image, key_prefix)
    
    # 실시간 크롭 영역 표시
    if left < right and top < bottom:
        st.markdown("#### 🔍 크롭 영역 미리보기")
        
        # 크롭 영역 정보
        crop_width = right - left
        crop_height = bottom - top
        total_pixels = crop_width * crop_height
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("크롭 크기", f"{crop_width}×{crop_height}")
        with col2:
            st.metric("총 픽셀", f"{total_pixels:,}")
        with col3:
            orig_total = image.size[0] * image.size[1]
            percentage = (total_pixels / orig_total) * 100
            st.metric("원본 대비", f"{percentage:.1f}%")
        
        # 크롭 영역이 유효한 경우에만 미리보기 표시
        try:
            from image_processing import crop_image
            preview_crop = crop_image(image, left, top, right, bottom)
            
            # 미리보기 이미지 크기 조정
            max_preview_width = 300
            if crop_width > max_preview_width:
                ratio = max_preview_width / crop_width
                preview_width = max_preview_width
                preview_height = int(crop_height * ratio)
                preview_crop = preview_crop.resize((preview_width, preview_height))
            
            st.image(preview_crop, caption=f"크롭 미리보기 ({crop_width}×{crop_height}px)", width=max_preview_width)
            
        except Exception as e:
            st.warning(f"미리보기를 생성할 수 없습니다: {e}")
    
    return left, top, right, bottom

# 이미지 크롭 인터페이스 (개선됨)
def image_crop_interface(image, key_prefix=""):
    """이미지 크롭을 위한 개선된 UI 인터페이스 - StreamlitAPIException 수정됨"""
    st.markdown("### 📐 정밀 크롭 설정")
    
    # 이미지 크기 정보 표시
    width, height = image.size
    st.write(f"**📏 원본 이미지 크기**: {width} × {height} 픽셀")
    
    # 빠른 설정 초기화 처리 (위젯 생성 전)
    preset_applied = st.session_state.get(f"{key_prefix}_preset_applied", False)
    preset_values = st.session_state.get(f"{key_prefix}_preset_values", {})
    
    # 기본값 설정
    default_left = 0
    default_top = 0
    default_right = min(width, 200)
    default_bottom = min(height, 200)
    
    if preset_applied and preset_values:
        # Preset이 적용된 경우 해당 값들로 업데이트
        default_left = preset_values.get('left', 0)
        default_top = preset_values.get('top', 0)
        default_right = preset_values.get('right', min(width, 200))
        default_bottom = preset_values.get('bottom', min(height, 200))
        
        # preset 상태 초기화
        st.session_state[f"{key_prefix}_preset_applied"] = False
        st.session_state[f"{key_prefix}_preset_values"] = {}
        
        st.success(f"✨ 빠른 설정이 적용되었습니다!")
        st.info(f"크롭 영역: {default_right-default_left} × {default_bottom-default_top} 픽셀")
    
    # 크롭 설정 방법 선택
    setting_method = st.radio(
        "설정 방법 선택",
        ["📍 좌표로 설정", "📏 크기와 위치로 설정"],
        key=f"{key_prefix}_setting_method",
        horizontal=True
    )
    
    if setting_method == "📍 좌표로 설정":
        # 기존 좌표 방식
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 시작 좌표 (왼쪽 위)**")
            left = st.number_input("왼쪽 (X)", min_value=0, max_value=width-1, value=default_left, key=f"{key_prefix}_left")
            top = st.number_input("위쪽 (Y)", min_value=0, max_value=height-1, value=default_top, key=f"{key_prefix}_top")
        
        with col2:
            st.write("**🎯 끝 좌표 (오른쪽 아래)**")
            right = st.number_input("오른쪽 (X)", min_value=left+1, max_value=width, value=max(left+1, default_right), key=f"{key_prefix}_right")
            bottom = st.number_input("아래쪽 (Y)", min_value=top+1, max_value=height, value=max(top+1, default_bottom), key=f"{key_prefix}_bottom")
    
    else:
        # 크기와 위치 방식
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📐 크롭 크기**")
            crop_width = st.number_input("너비 (픽셀)", min_value=1, max_value=width, value=min(200, width), key=f"{key_prefix}_crop_width")
            crop_height = st.number_input("높이 (픽셀)", min_value=1, max_value=height, value=min(200, height), key=f"{key_prefix}_crop_height")
        
        with col2:
            st.write("**📍 크롭 위치**")
            start_x = st.number_input("시작 X", min_value=0, max_value=width-crop_width, value=0, key=f"{key_prefix}_start_x")
            start_y = st.number_input("시작 Y", min_value=0, max_value=height-crop_height, value=0, key=f"{key_prefix}_start_y")
        
        # 좌표로 변환
        left = start_x
        top = start_y
        right = start_x + crop_width
        bottom = start_y + crop_height
    
    # 크롭 영역 크기 표시
    crop_width = right - left
    crop_height = bottom - top
    st.write(f"**✂️ 크롭 영역**: {crop_width} × {crop_height} 픽셀 ({crop_width * crop_height:,} 총 픽셀)")
    
    # 빠른 설정 버튼들 (session_state 직접 수정 방지)
    st.write("**⚡ 빠른 설정**")
    ratio_col1, ratio_col2, ratio_col3, ratio_col4, ratio_col5 = st.columns(5)
    
    with ratio_col1:
        if st.button("🔲 정사각형", key=f"{key_prefix}_square", help="중앙에서 정사각형으로 크롭"):
            size = min(width, height) // 2
            st.session_state[f"{key_prefix}_preset_values"] = {
                'left': (width - size) // 2,
                'top': (height - size) // 2,
                'right': (width + size) // 2,
                'bottom': (height + size) // 2
            }
            st.session_state[f"{key_prefix}_preset_applied"] = True
            st.rerun()
    
    with ratio_col2:
        if st.button("📺 16:9", key=f"{key_prefix}_16_9", help="16:9 와이드스크린 비율"):
            if width >= height:
                new_height = width * 9 // 16
                if new_height <= height:
                    preset_vals = {
                        'left': 0,
                        'top': (height - new_height) // 2,
                        'right': width,
                        'bottom': (height + new_height) // 2
                    }
                else:
                    new_width = height * 16 // 9
                    preset_vals = {
                        'left': (width - new_width) // 2,
                        'top': 0,
                        'right': (width + new_width) // 2,
                        'bottom': height
                    }
            else:
                new_width = height * 16 // 9
                if new_width <= width:
                    preset_vals = {
                        'left': (width - new_width) // 2,
                        'top': 0,
                        'right': (width + new_width) // 2,
                        'bottom': height
                    }
                else:
                    new_height = width * 9 // 16
                    preset_vals = {
                        'left': 0,
                        'top': (height - new_height) // 2,
                        'right': width,
                        'bottom': (height + new_height) // 2
                    }
            
            st.session_state[f"{key_prefix}_preset_values"] = preset_vals
            st.session_state[f"{key_prefix}_preset_applied"] = True
            st.rerun()
    
    with ratio_col3:
        if st.button("📷 4:3", key=f"{key_prefix}_4_3", help="4:3 표준 비율"):
            if width >= height:
                new_height = width * 3 // 4
                if new_height <= height:
                    preset_vals = {
                        'left': 0,
                        'top': (height - new_height) // 2,
                        'right': width,
                        'bottom': (height + new_height) // 2
                    }
                else:
                    new_width = height * 4 // 3
                    preset_vals = {
                        'left': (width - new_width) // 2,
                        'top': 0,
                        'right': (width + new_width) // 2,
                        'bottom': height
                    }
            else:
                new_width = height * 4 // 3
                if new_width <= width:
                    preset_vals = {
                        'left': (width - new_width) // 2,
                        'top': 0,
                        'right': (width + new_width) // 2,
                        'bottom': height
                    }
                else:
                    new_height = width * 3 // 4
                    preset_vals = {
                        'left': 0,
                        'top': (height - new_height) // 2,
                        'right': width,
                        'bottom': (height + new_height) // 2
                    }
            
            st.session_state[f"{key_prefix}_preset_values"] = preset_vals
            st.session_state[f"{key_prefix}_preset_applied"] = True
            st.rerun()
    
    with ratio_col4:
        if st.button("🎯 중앙 50%", key=f"{key_prefix}_center", help="중앙에서 50% 크기로 크롭"):
            margin_x = width // 4
            margin_y = height // 4
            st.session_state[f"{key_prefix}_preset_values"] = {
                'left': margin_x,
                'top': margin_y,
                'right': width - margin_x,
                'bottom': height - margin_y
            }
            st.session_state[f"{key_prefix}_preset_applied"] = True
            st.rerun()
    
    with ratio_col5:
        if st.button("🖼️ 전체", key=f"{key_prefix}_full", help="전체 이미지 선택"):
            st.session_state[f"{key_prefix}_preset_values"] = {
                'left': 0,
                'top': 0,
                'right': width,
                'bottom': height
            }
            st.session_state[f"{key_prefix}_preset_applied"] = True
            st.rerun()
    
    return left, top, right, bottom

# 크롭된 이미지를 파일로 다운로드하는 함수
def download_cropped_image(cropped_image, filename="cropped_image.png"):
    """크롭된 이미지를 다운로드할 수 있는 버튼 생성"""
    buffer = io.BytesIO()
    cropped_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    st.download_button(
        label="🔽 잘린 이미지 다운로드",
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
        use_container_width=True
    )

def download_background_removed_image(bg_removed_image, filename="background_removed.png"):
    """배경 제거된 이미지를 다운로드할 수 있는 버튼 생성"""
    try:
        buffer = io.BytesIO()
        # PNG 형식으로 저장 (투명도 지원)
        bg_removed_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        st.download_button(
            label="📥 배경 제거 이미지 다운로드",
            data=buffer.getvalue(),
            file_name=filename,
            mime="image/png",
            use_container_width=True,
            help="배경이 제거된 이미지를 PNG 형식으로 다운로드합니다 (투명배경 포함)"
        )
        return True
    except Exception as e:
        st.error(f"다운로드 준비 중 오류가 발생했습니다: {str(e)}")
        return False

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

# 결과 시각화 함수
def display_similarity_results(ssim_score, psnr_score, vgg_score, avg_score):
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
        st.progress(float(min(ssim_score/100, 1.0)))
        if ssim_score > 80:
            st.success("매우 유사한 구조")
        elif ssim_score > 60:
            st.info("유사한 구조")
        else:
            st.warning("구조적 차이가 큼")
        
    with col2:
        st.markdown(f"**PSNR**: {psnr_score:.2f}%")
        st.progress(float(min(psnr_score/100, 1.0)))
        if psnr_score > 80:
            st.success("매우 유사한 품질")
        elif psnr_score > 60:
            st.info("양호한 품질")
        else:
            st.warning("품질 차이가 큼")
        
    with col3:
        st.markdown(f"**VGG16 기반 Cosine 유사도**: {vgg_score:.2f}%")
        st.progress(float(min(vgg_score/100, 1.0)))
        if vgg_score > 80:
            st.success("매우 유사한 의미적 내용")
        elif vgg_score > 60:
            st.info("유사한 의미적 내용")
        else:
            st.warning("의미적 차이가 큼")
    
    # 평균 점수 표시
    st.markdown(f"**평균 유사도 점수**: {avg_score:.2f}%")
    st.progress(float(min(avg_score/100, 1.0)))
    if avg_score > 80:
        st.success("전반적으로 매우 유사")
    elif avg_score > 60:
        st.info("전반적으로 유사")
    else:
        st.warning("전반적으로 차이가 큼")

def regional_background_removal_interface(image, key_prefix="region_bg"):
    """영역 지정 배경 제거를 위한 인터페이스"""
    import image_processing
    import math
    
    st.markdown("### 🎯 영역 지정 배경 제거")
    st.write("특정 영역에만 배경 제거를 적용할 수 있습니다.")
    
    # 이미지 크기 정보
    width, height = image.size
    st.write(f"**📏 이미지 크기**: {width} × {height} 픽셀")
    
    # 마스크 타입 선택
    mask_type = st.radio(
        "🔲 마스크 타입 선택",
        ["전체 이미지", "사각형 영역", "다각형 영역"],
        key=f"{key_prefix}_mask_type",
        horizontal=True,
        help="전체 이미지: 모든 영역, 사각형: 직사각형 영역, 다각형: 자유로운 모양"
    )
    
    mask_coords = None
    
    if mask_type == "사각형 영역":
        st.write("**📐 사각형 영역 설정**")
        
        # 빠른 설정 버튼들
        st.write("**⚡ 빠른 설정**")
        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        
        # 기본값 설정
        default_left = st.session_state.get(f"{key_prefix}_rect_left", 0)
        default_top = st.session_state.get(f"{key_prefix}_rect_top", 0) 
        default_right = st.session_state.get(f"{key_prefix}_rect_right", min(width, 200))
        default_bottom = st.session_state.get(f"{key_prefix}_rect_bottom", min(height, 200))
        
        with quick_col1:
            if st.button("🎯 중앙", key=f"{key_prefix}_center_rect"):
                margin_x = width // 4
                margin_y = height // 4
                st.session_state[f"{key_prefix}_rect_left"] = margin_x
                st.session_state[f"{key_prefix}_rect_top"] = margin_y
                st.session_state[f"{key_prefix}_rect_right"] = width - margin_x
                st.session_state[f"{key_prefix}_rect_bottom"] = height - margin_y
                st.rerun()
        
        with quick_col2:
            if st.button("📺 상단", key=f"{key_prefix}_top_rect"):
                st.session_state[f"{key_prefix}_rect_left"] = 0
                st.session_state[f"{key_prefix}_rect_top"] = 0
                st.session_state[f"{key_prefix}_rect_right"] = width
                st.session_state[f"{key_prefix}_rect_bottom"] = height // 2
                st.rerun()
                
        with quick_col3:
            if st.button("📱 좌측", key=f"{key_prefix}_left_rect"):
                st.session_state[f"{key_prefix}_rect_left"] = 0
                st.session_state[f"{key_prefix}_rect_top"] = 0
                st.session_state[f"{key_prefix}_rect_right"] = width // 2
                st.session_state[f"{key_prefix}_rect_bottom"] = height
                st.rerun()
                
        with quick_col4:
            if st.button("🖼️ 전체", key=f"{key_prefix}_full_rect"):
                st.session_state[f"{key_prefix}_rect_left"] = 0
                st.session_state[f"{key_prefix}_rect_top"] = 0
                st.session_state[f"{key_prefix}_rect_right"] = width
                st.session_state[f"{key_prefix}_rect_bottom"] = height
                st.rerun()
        
        # 좌표 입력
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 시작 좌표 (왼쪽 위)**")
            left = st.number_input("왼쪽 (X)", min_value=0, max_value=width-1, value=default_left, key=f"{key_prefix}_rect_left")
            top = st.number_input("위쪽 (Y)", min_value=0, max_value=height-1, value=default_top, key=f"{key_prefix}_rect_top")
        
        with col2:
            st.write("**🎯 끝 좌표 (오른쪽 아래)**")
            right = st.number_input("오른쪽 (X)", min_value=left+1, max_value=width, value=max(left+1, default_right), key=f"{key_prefix}_rect_right")
            bottom = st.number_input("아래쪽 (Y)", min_value=top+1, max_value=height, value=max(top+1, default_bottom), key=f"{key_prefix}_rect_bottom")
        
        mask_coords = [left, top, right, bottom]
        
        # 영역 크기 표시
        region_width = right - left
        region_height = bottom - top
        st.write(f"**🎯 선택된 영역**: {region_width} × {region_height} 픽셀")
        
    elif mask_type == "다각형 영역":
        st.write("**🔺 다각형 영역 설정**")
        st.info("💡 다각형의 각 꼭지점 좌표를 입력하세요 (최소 3개 필요)")
        
        # 다각형 포인트 개수 선택
        num_points = st.slider("꼭지점 개수", min_value=3, max_value=10, value=4, key=f"{key_prefix}_num_points")
        
        polygon_points = []
        
        # 빠른 설정 버튼
        st.write("**⚡ 다각형 템플릿**")
        template_col1, template_col2, template_col3 = st.columns(3)
        
        with template_col1:
            if st.button("🔺 삼각형", key=f"{key_prefix}_triangle"):
                # 상단 중앙, 하단 좌/우 삼각형
                points = [
                    (width//2, height//4),          # 상단 중앙
                    (width//4, height*3//4),        # 하단 좌측
                    (width*3//4, height*3//4)       # 하단 우측
                ]
                for i, (x, y) in enumerate(points[:num_points]):
                    st.session_state[f"{key_prefix}_poly_x_{i}"] = x
                    st.session_state[f"{key_prefix}_poly_y_{i}"] = y
                st.rerun()
        
        with template_col2:
            if st.button("💎 다이아몬드", key=f"{key_prefix}_diamond"):
                # 다이아몬드 모양
                points = [
                    (width//2, height//4),          # 상단
                    (width*3//4, height//2),        # 우측
                    (width//2, height*3//4),        # 하단
                    (width//4, height//2)           # 좌측
                ]
                for i, (x, y) in enumerate(points[:num_points]):
                    st.session_state[f"{key_prefix}_poly_x_{i}"] = x
                    st.session_state[f"{key_prefix}_poly_y_{i}"] = y
                st.rerun()
        
        with template_col3:
            if st.button("⭐ 별모양", key=f"{key_prefix}_star"):
                # 간단한 별 모양
                center_x, center_y = width//2, height//2
                radius = min(width, height) // 4
                points = []
                for i in range(num_points):
                    angle = (2 * math.pi * i / num_points) - (math.pi/2)
                    r = radius if i % 2 == 0 else radius // 2
                    x = center_x + int(r * math.cos(angle))
                    y = center_y + int(r * math.sin(angle))
                    points.append((x, y))
                
                for i, (x, y) in enumerate(points):
                    st.session_state[f"{key_prefix}_poly_x_{i}"] = max(0, min(width-1, x))
                    st.session_state[f"{key_prefix}_poly_y_{i}"] = max(0, min(height-1, y))
                st.rerun()
        
        # 각 꼭지점 좌표 입력
        for i in range(num_points):
            col1, col2 = st.columns(2)
            with col1:
                x = st.number_input(
                    f"점 {i+1} - X 좌표", 
                    min_value=0, max_value=width-1, 
                    value=st.session_state.get(f"{key_prefix}_poly_x_{i}", width//4), 
                    key=f"{key_prefix}_poly_x_{i}"
                )
            with col2:
                y = st.number_input(
                    f"점 {i+1} - Y 좌표", 
                    min_value=0, max_value=height-1, 
                    value=st.session_state.get(f"{key_prefix}_poly_y_{i}", height//4), 
                    key=f"{key_prefix}_poly_y_{i}"
                )
            polygon_points.append((x, y))
        
        mask_coords = polygon_points
        st.write(f"**🔺 다각형 꼭지점**: {len(polygon_points)}개")
        
    # 마스크 옵션
    if mask_type != "전체 이미지":
        st.write("**🎛️ 마스크 옵션**")
        invert_mask = st.checkbox(
            "마스크 영역 반전 (선택 영역 외부에 배경제거 적용)", 
            key=f"{key_prefix}_invert",
            help="체크하면 선택된 영역은 유지하고 나머지 영역의 배경을 제거합니다"
        )
    else:
        invert_mask = False
    
    # 배경 제거 임계값 설정 (rembg 없을 때 사용)
    if not image_processing.REMBG_AVAILABLE:
        st.write("**⚙️ 배경 제거 설정**")
        threshold = st.slider(
            "임계값 (높을수록 더 밝은 배경 제거)", 
            min_value=100, max_value=255, 
            value=240, 
            key=f"{key_prefix}_threshold",
            help="밝은 배경을 제거하는 기준값입니다"
        )
    else:
        threshold = 240
    
    return {
        'mask_type': mask_type,
        'mask_coords': mask_coords,
        'invert_mask': invert_mask,
        'threshold': threshold
    }

# LPIPS를 포함한 향상된 유사도 결과 표시 함수
def display_enhanced_similarity_results(scores, avg_score):
    """다양한 유사도 메트릭 결과를 표시하는 향상된 함수"""
    st.markdown("## 📊 고급 유사도 비교 결과")
    
    if not scores:
        st.error("계산된 유사도 점수가 없습니다.")
        return
    
    # 메트릭별 설명
    metric_descriptions = {
        'SSIM': '🔍 구조적 유사도 (인간의 시각적 품질 인식)',
        'PSNR': '📶 신호 대 잡음비 (픽셀 레벨 정확도)',
        'VGG_Cosine': '🧠 딥러닝 기반 특징 유사도',
        'LPIPS': '👁️ 학습된 지각적 거리 (낮을수록 유사함)'
    }
    
    # 점수별 색상 지정
    def get_score_color(score, metric_name=''):
        if metric_name == 'LPIPS':
            # LPIPS는 거리값이므로 낮을수록 좋음
            if score <= 0.2:
                return "🟢"
            elif score <= 0.4:
                return "🟡"
            elif score <= 0.6:
                return "🟠"
            else:
                return "🔴"
        else:
            # 기존 메트릭들 (높을수록 좋음)
            if score >= 80:
                return "🟢"
            elif score >= 60:
                return "🟡"
            elif score >= 40:
                return "🟠"
            else:
                return "🔴"
    
    # 평균 점수 계산 (LPIPS 포함 시 정규화)
    normalized_scores = []
    for metric, score in scores.items():
        if metric == 'LPIPS':
            # LPIPS 거리를 유사도로 변환 (0-1 -> 100-0)
            normalized_scores.append((1 - score) * 100)
        else:
            normalized_scores.append(score)
    
    normalized_avg = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0
    
    # 메인 결과 표시
    st.markdown("### 📈 종합 유사도 점수")
    st.markdown(f"## {get_score_color(normalized_avg)} **{normalized_avg:.1f}%** (평균)")
    
    # 상세 결과 테이블
    st.markdown("### 📋 상세 메트릭 결과")
    
    # 테이블 데이터 준비
    table_data = []
    for metric, score in scores.items():
        description = metric_descriptions.get(metric, f"{metric} 유사도")
        color = get_score_color(score, metric)
        
        if metric == 'LPIPS':
            # LPIPS는 거리값으로 표시
            score_text = f"{score:.3f} (거리)"
        else:
            # 다른 메트릭은 백분율로 표시
            score_text = f"{score:.2f}%"
            
        table_data.append([
            f"{color} {metric}",
            description,
            score_text
        ])
    
    # 테이블 표시
    import pandas as pd
    df = pd.DataFrame(table_data, columns=["메트릭", "설명", "점수"])
    st.table(df)
    
    # 시각적 프로그레스 바
    st.markdown("### 📊 시각적 유사도 표시")
    
    for metric, score in scores.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{metric}**: {metric_descriptions.get(metric, metric)}")
            if metric == 'LPIPS':
                # LPIPS: 거리이므로 역방향 프로그레스 (낮을수록 좋음)
                progress_value = max(0, 1 - score)
                st.progress(progress_value)
            else:
                # 다른 메트릭: 높을수록 좋음
                st.progress(min(score / 100, 1.0))
        with col2:
            if metric == 'LPIPS':
                st.metric(metric, f"{score:.3f}")
            else:
                st.metric(metric, f"{score:.1f}%")
    
    # 결과 해석 가이드
    st.markdown("### 🎯 결과 해석 가이드")
    
    if avg_score >= 85:
        st.success("🌟 **매우 높은 유사도**: 두 이미지는 거의 동일합니다!")
        interpretation = "이미지들이 시각적으로 매우 유사하며, 대부분의 관점에서 일치합니다."
    elif avg_score >= 70:
        st.info("✨ **높은 유사도**: 두 이미지는 상당히 유사합니다.")
        interpretation = "이미지들이 전반적으로 유사하지만 일부 세부사항에서 차이가 있을 수 있습니다."
    elif avg_score >= 50:
        st.warning("⚖️ **중간 유사도**: 유사한 부분과 다른 부분이 혼재합니다.")
        interpretation = "이미지들이 부분적으로 유사하지만 상당한 차이점들이 존재합니다."
    else:
        st.error("❌ **낮은 유사도**: 두 이미지는 상당히 다릅니다.")
        interpretation = "이미지들이 대부분의 관점에서 다르며, 유사성보다는 차이점이 두드러집니다."
    
    st.info(f"💡 **해석**: {interpretation}")
    
    # LPIPS 특별 안내
    if 'LPIPS' in scores:
        st.markdown("### 🧠 LPIPS에 대하여")
        st.info(
            "🔬 **LPIPS(Learned Perceptual Image Patch Similarity)**는 "
            "인간의 시각적 인지 과정을 모방한 최신 유사도 측정 방법입니다.\n\n"
            "• **특징**: 딥러닝 모델이 학습한 시각적 특징을 기반으로 측정\n"
            "• **장점**: 전통적인 메트릭보다 인간의 판단과 높은 상관관계\n"
            "• **용도**: 이미지 품질 평가, 생성 모델 성능 측정에 주로 사용"
        )