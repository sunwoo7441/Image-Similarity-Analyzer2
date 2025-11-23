import os
import sys

# Add these lines at the top of your script
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
# 모듈 import 방식 변경
import pages.image_comparison as image_comparison
import pages.results_viewer as results_viewer
import pages.results_summary as results_summary  # 새로 추가
import pages.color_analysis as color_analysis  # 색상 분석 페이지 추가
import pages.image_upscaler as image_upscaler  # 이미지 업스케일러 페이지 추가
import pages.idm_vton as idm_vton  # IDM-VTON 페이지 추가
import pages.image_sharpener as image_sharpener  # 이미지 선명화 페이지 추가
from db_utils import init_db
from ui_components import show_metric_explanation

def main():
    # 데이터베이스 초기화
    init_db()
    
    # 사이드바 메뉴
    st.sidebar.title("메뉴")
    page = st.sidebar.radio("페이지 선택", [
        "🖼️ 이미지 비교", 
        "🎨 색상 분석", 
        "🚀 이미지 업스케일러",
        "✨ 이미지 선명화",
        "🔄 IDM-VTON",
        "📋 결과 조회", 
        "📊 결과 요약"
    ])  # 이미지 선명화 메뉴 추가
    
    
    # 이미지 크기 선택을 직접 입력 가능하도록 변경
    st.sidebar.header("이미지 설정")
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
    
    # 세션 상태에 이미지 크기 저장
    st.session_state['image_size'] = image_size
    
    # 유사도 측정 방식 설명 표시 여부
    show_explanation = st.sidebar.checkbox("유사도 측정 방식 설명 보기", value=False)

    if show_explanation:
        show_metric_explanation()
    
    # 선택된 페이지에 따라 내용 표시
    if page == "🖼️ 이미지 비교":
        image_comparison.app()
    elif page == "🎨 색상 분석":
        color_analysis.app()
    elif page == "🚀 이미지 업스케일러":
        image_upscaler.main()
    elif page == "✨ 이미지 선명화":
        image_sharpener.app()
    elif page == "🔄 IDM-VTON":
        idm_vton.app()
    elif page == "📋 결과 조회":
        results_viewer.app()
    else:  # 📊 결과 요약
        results_summary.app()

if __name__ == "__main__":
    main()