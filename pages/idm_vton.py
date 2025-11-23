import streamlit as st
import webbrowser
import sys
import os

# 현재 디렉토리를 상위 폴더로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def app():
    st.title("🔄 IDM-VTON - Virtual Try-On")
    
    # 헤더 섹션
    st.markdown("---")
    st.markdown("## 👕 Virtual Try-On with IDM-VTON")
    
    # 소개 섹션
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🌟 IDM-VTON이란?
        
        **IDM-VTON (Improving Diffusion Models for Authentic Virtual Try-on)**은 
        최신 AI 기술을 활용한 가상 의류 착용 시스템입니다.
        
        #### ✨ 주요 기능:
        - **🎯 정확한 피팅**: 실제와 같은 자연스러운 의류 착용 효과
        - **🖼️ 고품질 결과**: 고해상도의 사실적인 이미지 생성
        - **⚡ 빠른 처리**: 효율적인 AI 모델을 통한 신속한 결과
        - **🎨 다양한 의류**: 다양한 종류의 의류와 스타일 지원
        
        #### 💡 사용 방법:
        1. 인물 사진 업로드
        2. 착용하고 싶은 의류 이미지 업로드
        3. AI가 자연스럽게 합성된 결과 이미지 생성
        """)
    
    with col2:
        st.info("""
        **🚀 Hugging Face Spaces에서 제공**
        
        이 도구는 Hugging Face의 
        고성능 AI 모델을 기반으로 
        작동합니다.
        
        **🔗 직접 접속하여 사용하세요!**
        """)
    
    # 메인 액션 섹션
    st.markdown("---")
    st.markdown("## 🌐 IDM-VTON 접속")
    
    # URL 정보
    idm_vton_url = "https://huggingface.co/spaces/yisol/IDM-VTON?utm_source=chatgpt.com"
    
    # 버튼과 링크 섹션
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 큰 버튼으로 링크 열기
        if st.button("🚀 IDM-VTON 웹사이트 열기", type="primary", use_container_width=True):
            st.markdown(f"""
            <script>
                window.open('{idm_vton_url}', '_blank');
            </script>
            """, unsafe_allow_html=True)
            st.success("새 탭에서 IDM-VTON이 열립니다!")
    
    # 직접 링크 제공
    st.markdown("---")
    st.markdown("### 🔗 직접 링크")
    st.markdown(f"**링크를 복사하여 브라우저에서 열어보세요:**")
    st.code(idm_vton_url, language=None)
    
    # iframe으로 페이지 임베드 (선택사항)
    st.markdown("---")
    st.markdown("### 📺 미리보기")
    
    show_preview = st.checkbox("웹페이지 미리보기 보기", value=False)
    
    if show_preview:
        try:
            st.markdown("**IDM-VTON 웹페이지 미리보기:**")
            st.markdown(f"""
            <iframe src="{idm_vton_url}" 
                    width="100%" 
                    height="800" 
                    frameborder="0">
            </iframe>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning("미리보기를 로드할 수 없습니다. 직접 링크를 사용해주세요.")
            st.error(f"오류: {str(e)}")
    
    # 사용 팁 섹션
    st.markdown("---")
    st.markdown("## 💡 사용 팁")
    
    with st.expander("📋 IDM-VTON 사용 가이드", expanded=False):
        st.markdown("""
        ### 🎯 최적의 결과를 위한 팁:
        
        #### 📸 인물 사진 준비:
        - **정면 포즈**: 정면을 향한 자연스러운 포즈
        - **깔끔한 배경**: 단순한 배경 또는 배경 제거된 이미지
        - **고화질**: 가능한 고해상도 이미지 사용
        - **전신 또는 상반신**: 착용할 의류에 따라 적절한 범위
        
        #### 👕 의류 이미지 준비:
        - **평평한 상태**: 의류가 평평하게 펼쳐진 상태
        - **깔끔한 배경**: 흰색 또는 투명 배경 권장
        - **고품질**: 의류의 디테일이 잘 보이는 이미지
        - **적절한 크기**: 너무 작거나 크지 않은 적당한 크기
        
        #### ⚙️ 사용 과정:
        1. IDM-VTON 웹사이트 접속
        2. 인물 이미지 업로드
        3. 의류 이미지 업로드
        4. 설정 조정 (필요시)
        5. 생성 버튼 클릭 후 결과 확인
        
        #### 🔧 문제 해결:
        - **느린 처리**: 서버 부하로 인한 지연 가능, 잠시 후 재시도
        - **오류 발생**: 이미지 형식이나 크기 확인
        - **품질 불만족**: 다른 각도나 포즈의 이미지로 재시도
        """)
    
    # 관련 도구 섹션
    st.markdown("---")
    st.markdown("## 🛠️ 관련 도구")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎨 이미지 전처리
        - **이미지 편집**: 밝기, 대비, 색상 조정
        - **배경 제거**: 깔끔한 배경 처리
        - **크롭 기능**: 필요한 부분만 자르기
        """)
        
        if st.button("🖌️ 이미지 편집 페이지로", key="edit_page"):
            st.switch_page("pages/image_comparison.py")
    
    with col2:
        st.markdown("""
        ### 🚀 이미지 업스케일링
        - **AI 업스케일링**: 이미지 해상도 향상
        - **품질 개선**: Real-ESRGAN, SwinIR 등
        - **고해상도 변환**: 더 선명한 결과
        """)
        
        if st.button("📈 업스케일링 페이지로", key="upscale_page"):
            st.switch_page("pages/image_upscaler.py")
    
    # footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        💡 IDM-VTON은 Hugging Face에서 제공하는 오픈소스 AI 모델입니다.<br>
        🔗 더 많은 AI 도구는 <a href="https://huggingface.co/spaces" target="_blank">Hugging Face Spaces</a>에서 확인하세요.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()