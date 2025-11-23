import streamlit as st
import uuid
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import sys
import os

# 현재 디렉토리를 상위 폴더로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui_components import download_cropped_image
from db_utils import save_work_history

def unsharp_mask_filter(image, radius=2, strength=1.5):
    """언샵 마스크 필터를 적용하여 이미지를 선명하게 만듭니다."""
    # PIL 이미지를 numpy 배열로 변환
    img_array = np.array(image)
    
    # OpenCV를 사용하여 가우시안 블러 적용
    blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
    
    # 언샵 마스크 적용
    sharpened = cv2.addWeighted(img_array, 1.0 + strength, blurred, -strength, 0)
    
    # numpy 배열을 다시 PIL 이미지로 변환
    return Image.fromarray(np.uint8(np.clip(sharpened, 0, 255)))

def laplacian_sharpen(image, alpha=1.5):
    """라플라시안 필터를 사용하여 이미지를 선명하게 만듭니다."""
    img_array = np.array(image)
    
    # 라플라시안 커널
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    
    # 각 채널에 대해 필터 적용
    if len(img_array.shape) == 3:
        sharpened = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            sharpened[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, kernel)
    else:
        sharpened = cv2.filter2D(img_array, -1, kernel)
    
    # 알파 블렌딩
    result = cv2.addWeighted(img_array, 1 - alpha, sharpened, alpha, 0)
    
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))

def high_pass_sharpen(image, radius=3, strength=2.0):
    """고주파 필터를 사용하여 이미지를 선명하게 만듭니다."""
    img_array = np.array(image).astype(np.float32)
    
    # 가우시안 블러로 저주파 성분 추출
    low_freq = cv2.GaussianBlur(img_array, (0, 0), radius)
    
    # 고주파 성분 = 원본 - 저주파
    high_freq = img_array - low_freq
    
    # 선명화 = 원본 + (고주파 * 강도)
    sharpened = img_array + (high_freq * strength)
    
    return Image.fromarray(np.uint8(np.clip(sharpened, 0, 255)))

def region_sharpen(image, x, y, width, height, method='unsharp', **params):
    """이미지의 특정 영역만 선명하게 만듭니다."""
    # 원본 이미지 복사
    result_image = image.copy()
    
    # 선택된 영역 추출
    region = image.crop((x, y, x + width, y + height))
    
    # 선택된 방법으로 선명화 적용
    if method == 'unsharp':
        sharpened_region = unsharp_mask_filter(region, 
                                             radius=params.get('radius', 2), 
                                             strength=params.get('strength', 1.5))
    elif method == 'laplacian':
        sharpened_region = laplacian_sharpen(region, 
                                           alpha=params.get('alpha', 1.5))
    elif method == 'highpass':
        sharpened_region = high_pass_sharpen(region, 
                                           radius=params.get('radius', 3), 
                                           strength=params.get('strength', 2.0))
    else:  # PIL 기본 필터
        enhancer = ImageEnhance.Sharpness(region)
        sharpened_region = enhancer.enhance(params.get('factor', 2.0))
    
    # 선명화된 영역을 원본에 붙여넣기
    result_image.paste(sharpened_region, (x, y))
    
    return result_image

def draw_selection_box(image, x, y, width, height):
    """선택 영역을 시각적으로 표시합니다."""
    display_image = image.copy()
    
    # OpenCV를 사용하여 선택 박스 그리기
    img_array = np.array(display_image)
    
    # 빨간색 테두리 그리기
    cv2.rectangle(img_array, (x, y), (x + width, y + height), (255, 0, 0), 3)
    
    # 반투명 오버레이 추가
    overlay = img_array.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 255, 0), -1)
    img_array = cv2.addWeighted(img_array, 0.9, overlay, 0.1, 0)
    
    return Image.fromarray(img_array)

def app():
    st.title("✨ 이미지 선명화 도구")
    
    st.markdown("""
    ### 🎯 특정 영역 선명화
    이미지의 흐릿한 부분을 선택적으로 선명하게 만들 수 있는 도구입니다.
    """)
    
    # 이미지 업로드
    st.markdown("## 📸 이미지 업로드")
    uploaded_file = st.file_uploader("선명화할 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # 이미지 로드
        original_image = Image.open(uploaded_file).convert("RGB")
        
        # 이미지 정보 표시
        st.markdown("### 📋 이미지 정보")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(original_image, caption=f"원본 이미지 (크기: {original_image.width}×{original_image.height})", use_column_width=True)
        
        with col2:
            st.info(f"""
            **이미지 정보**
            - 크기: {original_image.width} × {original_image.height}
            - 총 픽셀: {original_image.width * original_image.height:,}
            - 포맷: RGB
            """)
        
        # 선명화 설정
        st.markdown("## ⚙️ 선명화 설정")
        
        # 선명화 방법 선택
        col1, col2 = st.columns([1, 1])
        
        with col1:
            sharpen_method = st.selectbox(
                "선명화 방법 선택",
                options=['unsharp', 'laplacian', 'highpass', 'pil_enhance'],
                format_func=lambda x: {
                    'unsharp': '🎯 언샵 마스크 (추천)',
                    'laplacian': '⚡ 라플라시안 필터',
                    'highpass': '🔍 고주파 필터',
                    'pil_enhance': '🛠️ 기본 선명화'
                }[x]
            )
        
        with col2:
            # 전체 이미지 선명화 옵션
            apply_to_all = st.checkbox("전체 이미지에 적용", value=False)
        
        # 방법별 파라미터 설정
        st.markdown("### 🎛️ 세부 설정")
        
        if sharpen_method == 'unsharp':
            col1, col2 = st.columns(2)
            with col1:
                radius = int(st.slider("블러 반경", 1, 10, 2))
            with col2:
                strength = st.slider("선명화 강도", 0.5, 3.0, 1.5, 0.1)
            params = {'radius': radius, 'strength': strength}
            
        elif sharpen_method == 'laplacian':
            alpha = st.slider("선명화 강도", 0.5, 3.0, 1.5, 0.1)
            params = {'alpha': alpha}
            
        elif sharpen_method == 'highpass':
            col1, col2 = st.columns(2)
            with col1:
                radius = int(st.slider("필터 반경", 1, 10, 3))
            with col2:
                strength = st.slider("선명화 강도", 0.5, 5.0, 2.0, 0.1)
            params = {'radius': radius, 'strength': strength}
            
        else:  # pil_enhance
            factor = st.slider("선명화 정도", 0.5, 5.0, 2.0, 0.1)
            params = {'factor': factor}
        
        # 영역 선택 또는 전체 적용
        if not apply_to_all:
            st.markdown("### 📐 영역 선택")
            
            col1, col2 = st.columns(2)
        with col1:
            x = int(st.number_input("시작 X 좌표", 0, original_image.width-1, 0))
            width = int(st.number_input("너비", 1, original_image.width-x, min(200, original_image.width-x)))
        
        with col2:
            y = int(st.number_input("시작 Y 좌표", 0, original_image.height-1, 0))
            height = int(st.number_input("높이", 1, original_image.height-y, min(200, original_image.height-y)))            # 선택 영역 미리보기
            if st.checkbox("선택 영역 표시", value=True):
                selection_preview = draw_selection_box(original_image, x, y, width, height)
                st.image(selection_preview, caption="선택된 영역 (빨간 테두리)", use_column_width=True)
        
        # 선명화 실행
        if st.button("🚀 선명화 실행", type="primary"):
            with st.spinner("이미지를 선명하게 만드는 중..."):
                try:
                    if apply_to_all:
                        # 전체 이미지에 선명화 적용
                        if sharpen_method == 'unsharp':
                            result_image = unsharp_mask_filter(original_image, **params)
                        elif sharpen_method == 'laplacian':
                            result_image = laplacian_sharpen(original_image, **params)
                        elif sharpen_method == 'highpass':
                            result_image = high_pass_sharpen(original_image, **params)
                        else:  # pil_enhance
                            enhancer = ImageEnhance.Sharpness(original_image)
                            result_image = enhancer.enhance(params['factor'])
                    else:
                        # 특정 영역만 선명화 적용
                        result_image = region_sharpen(original_image, x, y, width, height, sharpen_method, **params)
                    
                    # 결과 표시
                    st.markdown("## 🎯 선명화 결과")
                    
                    # Before & After 비교
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(original_image, caption="원본 이미지", use_column_width=True)
                    
                    with col2:
                        st.image(result_image, caption="선명화된 이미지", use_column_width=True)
                    
                    # 확대 비교 (선택 영역만)
                    if not apply_to_all:
                        st.markdown("### 🔍 선택 영역 확대 비교")
                        
                        original_crop = original_image.crop((x, y, x + width, y + height))
                        result_crop = result_image.crop((x, y, x + width, y + height))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(original_crop, caption="원본 영역", use_column_width=True)
                        with col2:
                            st.image(result_crop, caption="선명화된 영역", use_column_width=True)
                    
                    # 다운로드 옵션
                    st.markdown("### 💾 결과 다운로드")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        download_cropped_image(result_image, "sharpened_image.png")
                    
                    if not apply_to_all:
                        with col2:
                            result_crop = result_image.crop((x, y, x + width, y + height))
                            download_cropped_image(result_crop, "sharpened_region.png")
                    
                    with col3:
                        # 비교 이미지 생성
                        comparison_width = max(original_image.width, result_image.width)
                        comparison_height = original_image.height + result_image.height + 50
                        
                        comparison_image = Image.new('RGB', (comparison_width, comparison_height), (255, 255, 255))
                        comparison_image.paste(original_image, (0, 0))
                        comparison_image.paste(result_image, (0, original_image.height + 50))
                        
                        download_cropped_image(comparison_image, "before_after_comparison.png")
                    
                    # 작업 이력 저장
                    try:
                        work_id = str(uuid.uuid4())
                        save_work_history(
                            work_type="image_sharpening",
                            title="이미지 선명화",
                            description=f"방법: {sharpen_method}, 전체적용: {apply_to_all}"
                        )
                        st.success("✅ 선명화가 완료되었고 작업 이력이 저장되었습니다!")
                    except Exception as e:
                        st.warning(f"작업 이력 저장 중 오류: {str(e)}")
                
                except Exception as e:
                    st.error(f"선명화 처리 중 오류가 발생했습니다: {str(e)}")
        
        # 사용법 가이드
        with st.expander("📖 사용법 가이드", expanded=False):
            st.markdown("""
            ### 🎯 선명화 방법별 특징:
            
            #### 🎯 언샵 마스크 (추천)
            - **용도**: 일반적인 이미지 선명화
            - **특징**: 자연스러운 결과, 노이즈 적음
            - **설정**: 반경(작을수록 세밀), 강도(클수록 강함)
            
            #### ⚡ 라플라시안 필터
            - **용도**: 빠른 선명화, 엣지 강조
            - **특징**: 강한 엣지 강조 효과
            - **설정**: 강도만 조절 가능
            
            #### 🔍 고주파 필터
            - **용도**: 세밀한 디테일 강조
            - **특징**: 텍스처와 디테일에 효과적
            - **설정**: 반경과 강도 모두 조절
            
            #### 🛠️ 기본 선명화
            - **용도**: 간단한 선명화
            - **특징**: PIL 라이브러리 기본 기능
            - **설정**: 선명화 정도만 조절
            
            ### 💡 사용 팁:
            - **흐릿한 텍스트**: 언샵 마스크, 강도 1.5-2.0
            - **사진 디테일**: 고주파 필터, 반경 2-3
            - **빠른 처리**: 라플라시안 필터 사용
            - **자연스러운 결과**: 언샵 마스크 권장
            
            ### ⚠️ 주의사항:
            - 너무 강한 설정은 노이즈를 증가시킬 수 있습니다
            - 이미 선명한 이미지는 과도한 처리를 피하세요
            - 큰 이미지는 처리 시간이 오래 걸릴 수 있습니다
            """)
    
    else:
        st.info("📸 이미지를 업로드하여 선명화 작업을 시작하세요!")

if __name__ == "__main__":
    app()