# pages/color_analysis.py - 전용 색상 분석 페이지

import streamlit as st
import numpy as np
from PIL import Image
import sys
import os
import io
import base64
import pandas as pd
import json

# 현재 디렉토리를 상위 폴더로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from color_extractor import ColorExtractor, display_color_analysis_ui
from image_processing import resize_image, safe_image_open
from db_utils import save_work_history, get_work_history, save_work_comment  # 작업 히스토리 저장 함수 추가

def app():
    st.title("🎨 RGB 색상 분석기")
    st.markdown("이미지에서 RGB 색상을 추출하고 상세한 색상 분석을 수행합니다.")
    
    # 사이드바에 기능 옵션
    st.sidebar.header("🔧 분석 옵션")
    
    # 메인 업로드 섹션
    st.markdown("## 📤 이미지 업로드")
    
    # 탭으로 구분된 업로드 방식
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ 단일 이미지 분석", "🔄 이미지 비교 분석", "📁 다중 이미지 분석", "📊 결과 조회"])
    
    with tab1:
        st.markdown("### 단일 이미지 색상 분석")
        
        uploaded_file = st.file_uploader(
            "분석할 이미지를 업로드하세요", 
            type=["jpg", "png", "jpeg", "bmp", "tiff"],
            key="single_image"
        )
        
        if uploaded_file:
            # 이미지 로드 및 표시
            try:
                image = safe_image_open(uploaded_file).convert("RGB")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(image, caption=f"업로드된 이미지", use_container_width=True)
                
            except Exception as e:
                st.error(f"이미지 로드 실패: {str(e)}")
                return
            
            with col2:
                st.markdown("#### 📊 이미지 정보")
                st.write(f"**파일명:** {uploaded_file.name}")
                st.write(f"**크기:** {image.width} × {image.height}")
                st.write(f"**총 픽셀:** {image.width * image.height:,}")
                st.write(f"**종횡비:** {image.width/image.height:.2f}")
                
                # 파일 크기 정보
                file_size = len(uploaded_file.getvalue())
                if file_size > 1024*1024:
                    st.write(f"**파일 크기:** {file_size/(1024*1024):.1f} MB")
                else:
                    st.write(f"**파일 크기:** {file_size/1024:.1f} KB")
            
            # 색상 분석 실행
            st.markdown("---")
            display_color_analysis_ui(image, "single")
            
            # 추가 분석 도구
            st.markdown("---")
            st.markdown("### 🔍 고급 색상 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 색상 팔레트 저장", key="save_palette"):
                    save_color_palette(image, uploaded_file.name)
            
            with col2:
                if st.button("📊 상세 리포트 생성", key="detailed_report"):
                    generate_detailed_report(image, uploaded_file.name)
    
    with tab2:
        st.markdown("### 이미지 간 색상 비교 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 첫 번째 이미지")
            img1 = st.file_uploader(
                "첫 번째 이미지", 
                type=["jpg", "png", "jpeg", "bmp", "tiff"],
                key="compare_img1"
            )
            if img1:
                try:
                    image1 = safe_image_open(img1).convert("RGB")
                    st.image(image1, caption="이미지 1", use_container_width=True)
                except Exception as e:
                    st.error(f"이미지 1 로드 실패: {str(e)}")
                    img1 = None
        
        with col2:
            st.markdown("#### 두 번째 이미지")
            img2 = st.file_uploader(
                "두 번째 이미지", 
                type=["jpg", "png", "jpeg", "bmp", "tiff"],
                key="compare_img2"
            )
            if img2:
                try:
                    image2 = safe_image_open(img2).convert("RGB")
                    st.image(image2, caption="이미지 2", use_container_width=True)
                except Exception as e:
                    st.error(f"이미지 2 로드 실패: {str(e)}")
                    img2 = None
        
        # 이미지 비교 분석
        if img1 and img2:
            st.markdown("---")
            st.markdown("### 🔍 색상 비교 분석")
            
            if st.button("🎨 색상 비교 분석 시작", key="compare_colors"):
                compare_image_colors(image1, image2, img1.name, img2.name)
    
    with tab3:
        st.markdown("### 다중 이미지 색상 분석")
        
        uploaded_files = st.file_uploader(
            "여러 이미지를 업로드하세요 (최대 5개)",
            type=["jpg", "png", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key="multi_images"
        )
        
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("최대 5개의 이미지만 분석할 수 있습니다.")
                uploaded_files = uploaded_files[:5]
            
            st.markdown(f"### 📊 {len(uploaded_files)}개 이미지 분석")
            
            # 이미지 미리보기
            cols = st.columns(min(len(uploaded_files), 3))
            images = []
            
            for i, file in enumerate(uploaded_files):
                try:
                    image = safe_image_open(file).convert("RGB")
                    images.append((image, file.name))
                    
                    with cols[i % 3]:
                        st.image(image, caption=file.name, use_container_width=True)
                except Exception as e:
                    st.error(f"이미지 '{file.name}' 로드 실패: {str(e)}")
                    continue
            
            if st.button("🎨 다중 이미지 색상 분석", key="multi_analysis"):
                analyze_multiple_images(images)
    
    with tab4:
        st.markdown("### 📊 색상 분석 결과 조회")
        display_color_analysis_history()

def save_color_palette(image, filename):
    """색상 팔레트를 파일로 저장"""
    try:
        color_extractor = ColorExtractor()
        dominant_colors = color_extractor.extract_dominant_colors_simple(image, 8)
        stats = color_extractor.get_color_statistics(image)
        
        # 이미지를 임시 저장
        temp_dir = "Result/color_analysis"
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, f"input_{filename}")
        image.save(temp_image_path)
        
        # 작업 히스토리 저장
        parameters = {
            "colors_extracted": len(dominant_colors),
            "color_harmony": color_extractor.get_color_harmony_type(dominant_colors)
        }
        
        results = {
            "dominant_colors": [
                {
                    "hex": color['hex'],
                    "rgb": color['rgb'],
                    "percentage": color['percentage']
                } for color in dominant_colors
            ],
            "color_temperature": stats['overall']['temperature'],
            "avg_brightness": stats['overall']['brightness']
        }
        
        work_id = save_work_history(
            work_type="color_analysis",
            title=f"색상 팔레트 분석 - {filename}",
            description=f"{len(dominant_colors)}개 주요 색상 추출",
            input_images=json.dumps([temp_image_path]),
            parameters=json.dumps(parameters),
            results=json.dumps(results)
        )
        
        # 색상 정보를 텍스트로 저장
        palette_text = f"색상 팔레트 - {filename}\n"
        palette_text += "=" * 50 + "\n\n"
        
        for i, color_info in enumerate(dominant_colors):
            palette_text += f"{i+1}. RGB: {color_info['rgb']}\n"
            palette_text += f"   HEX: {color_info['hex']}\n"
            palette_text += f"   비율: {color_info['percentage']:.2f}%\n"
            palette_text += f"   픽셀 수: {color_info['count']:,}\n\n"
        
        # 다운로드 버튼
        st.download_button(
            label="📥 색상 팔레트 다운로드",
            data=palette_text,
            file_name=f"color_palette_{filename.split('.')[0]}.txt",
            mime="text/plain"
        )
        
        st.success(f"색상 팔레트가 준비되고 작업 히스토리에 저장되었습니다! (ID: {work_id[:8]})")
        
    except Exception as e:
        st.error(f"색상 팔레트 저장 중 오류가 발생했습니다: {str(e)}")

def generate_detailed_report(image, filename):
    """상세한 색상 분석 리포트 생성"""
    try:
        color_extractor = ColorExtractor()
        
        # 다양한 분석 수행
        dominant_colors = color_extractor.extract_dominant_colors_simple(image, 8)
        stats = color_extractor.get_color_statistics(image)
        palette, positions = color_extractor.extract_color_palette(image, 8)
        
        # 리포트 생성
        report = f"RGB 색상 분석 리포트\n"
        report += f"파일명: {filename}\n"
        report += f"분석 일시: {st.session_state.get('current_time', '2025-11-05')}\n"
        report += "=" * 60 + "\n\n"
        
        # 기본 정보
        report += "1. 이미지 기본 정보\n"
        report += f"   크기: {image.width} × {image.height}\n"
        report += f"   총 픽셀: {image.width * image.height:,}\n"
        report += f"   종횡비: {image.width/image.height:.2f}\n\n"
        
        # 주요 색상
        report += "2. 주요 색상 분석\n"
        for i, color_info in enumerate(dominant_colors):
            report += f"   {i+1}위: RGB{color_info['rgb']} ({color_info['hex']}) - {color_info['percentage']:.2f}%\n"
        report += "\n"
        
        # RGB 통계
        report += "3. RGB 채널별 통계\n"
        for channel in ['red', 'green', 'blue']:
            data = stats[channel]
            report += f"   {channel.upper()} 채널:\n"
            report += f"     평균: {data['mean']:.1f}\n"
            report += f"     표준편차: {data['std']:.1f}\n"
            report += f"     범위: {data['min']} ~ {data['max']}\n"
            report += f"     중간값: {data['median']:.1f}\n"
        
        # 전체 통계
        report += f"\n4. 전체 이미지 통계\n"
        report += f"   평균 밝기: {stats['overall']['brightness']:.1f}\n"
        report += f"   대비 (표준편차): {stats['overall']['contrast']:.1f}\n"
        report += f"   색온도: {stats['overall']['temperature']:.0f}K\n"
        
        # 색상 조화 분석
        color_info = color_extractor.extract_dominant_colors_simple(image, 5)
        harmony_type = color_extractor.get_color_harmony_type(color_info)
        report += f"\n5. 색상 조화 분석\n"
        report += f"   조화 유형: {harmony_type}\n"
        
        # 다운로드 버튼
        st.download_button(
            label="📋 상세 리포트 다운로드",
            data=report,
            file_name=f"color_analysis_report_{filename.split('.')[0]}.txt",
            mime="text/plain"
        )
        
        st.success("상세 리포트가 준비되었습니다!")
        
    except Exception as e:
        st.error(f"리포트 생성 중 오류가 발생했습니다: {str(e)}")

def compare_image_colors(image1, image2, name1, name2):
    """두 이미지의 색상을 비교 분석"""
    color_extractor = ColorExtractor()
    
    try:
        with st.spinner("이미지 색상을 비교 분석하는 중..."):
            # 각 이미지의 주요 색상 추출
            colors1 = color_extractor.extract_dominant_colors_simple(image1, 5)
            colors2 = color_extractor.extract_dominant_colors_simple(image2, 5)
            
            # 색상 통계 계산
            stats1 = color_extractor.get_color_statistics(image1)
            stats2 = color_extractor.get_color_statistics(image2)
        
        st.markdown("#### 🎨 주요 색상 비교")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{name1}의 주요 색상**")
            for i, color in enumerate(colors1):
                color_box = f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 30px; height: 30px; background-color: {color['hex']}; 
                                border: 1px solid #333; margin-right: 10px;"></div>
                    <span>{color['hex']} ({color['percentage']:.1f}%)</span>
                </div>
                """
                st.markdown(color_box, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{name2}의 주요 색상**")
            for i, color in enumerate(colors2):
                color_box = f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 30px; height: 30px; background-color: {color['hex']}; 
                                border: 1px solid #333; margin-right: 10px;"></div>
                    <span>{color['percentage']:.1f}%)</span>
                </div>
                """
                st.markdown(color_box, unsafe_allow_html=True)
        
        # 통계 비교
        st.markdown("#### 📊 RGB 채널 통계 비교")
        
        import pandas as pd
        
        comparison_data = []
        for channel in ['red', 'green', 'blue']:
            comparison_data.append({
                '채널': channel.upper(),
                f'{name1} 평균': f"{stats1[channel]['mean']:.1f}",
                f'{name2} 평균': f"{stats2[channel]['mean']:.1f}",
                '차이': f"{abs(stats1[channel]['mean'] - stats2[channel]['mean']):.1f}",
                f'{name1} 표준편차': f"{stats1[channel]['std']:.1f}",
                f'{name2} 표준편차': f"{stats2[channel]['std']:.1f}"
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # 전체 비교 메트릭
        st.markdown("#### 🔍 종합 비교")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brightness_diff = abs(stats1['overall']['brightness'] - stats2['overall']['brightness'])
            st.metric("밝기 차이", f"{brightness_diff:.1f}")
        
        with col2:
            contrast_diff = abs(stats1['overall']['contrast'] - stats2['overall']['contrast'])
            st.metric("대비 차이", f"{contrast_diff:.1f}")
        
        with col3:
            temp_diff = abs(stats1['overall']['temperature'] - stats2['overall']['temperature'])
            st.metric("색온도 차이", f"{temp_diff:.0f}K")
        
        # 색상 조화 비교
        harmony1 = color_extractor.get_color_harmony_type(colors1)
        harmony2 = color_extractor.get_color_harmony_type(colors2)
        
        st.markdown("#### 🎨 색상 조화 비교")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**{name1}**: {harmony1}")
        with col2:
            st.info(f"**{name2}**: {harmony2}")
        
        # 유사도 계산
        similarity_score = calculate_color_similarity(colors1, colors2)
        st.markdown(f"#### 🔍 색상 유사도: {similarity_score:.1f}%")
        
        # 유사도에 따른 메시지
        if similarity_score > 80:
            st.success("매우 유사한 색상 구성을 가지고 있습니다.")
        elif similarity_score > 60:
            st.info("어느 정도 유사한 색상 구성을 가지고 있습니다.")
        elif similarity_score > 40:
            st.warning("약간의 색상 차이가 있습니다.")
        else:
            st.error("상당히 다른 색상 구성을 가지고 있습니다.")
        
        # 구분선 추가
        st.markdown("---")
        
        # 결과 저장 - 항상 표시
        st.markdown("#### 💾 결과 저장")
        if st.button("💾 비교 결과 저장", key="save_comparison", help="분석 결과를 데이터베이스에 저장합니다"):
            try:
                save_color_comparison_result(image1, image2, name1, name2, similarity_score, colors1, colors2, stats1, stats2)
            except Exception as save_error:
                st.error(f"저장 중 오류가 발생했습니다: {str(save_error)}")
                st.error("자세한 오류 정보:")
                import traceback
                st.code(traceback.format_exc())
        
    except Exception as e:
        st.error(f"색상 비교 분석 중 오류가 발생했습니다: {str(e)}")
        st.error("자세한 오류 정보:")
        import traceback
        st.code(traceback.format_exc())

def calculate_color_similarity(colors1, colors2):
    """두 색상 팔레트 간의 유사도 계산"""
    try:
        total_similarity = 0
        count = 0
        
        for color1 in colors1:
            rgb1 = np.array(color1['rgb'])
            max_similarity = 0
            
            for color2 in colors2:
                rgb2 = np.array(color2['rgb'])
                # 유클리드 거리 기반 유사도
                distance = np.linalg.norm(rgb1 - rgb2)
                similarity = max(0, 100 - (distance / 4.41))  # 정규화
                max_similarity = max(max_similarity, similarity)
            
            total_similarity += max_similarity * color1['percentage'] / 100
            count += color1['percentage'] / 100
        
        return total_similarity / count if count > 0 else 0
    except:
        return 0

def analyze_multiple_images(images):
    """다중 이미지 색상 분석"""
    try:
        color_extractor = ColorExtractor()
        
        with st.spinner("다중 이미지를 분석하는 중..."):
            all_colors = []
            all_stats = []
            
            for image, name in images:
                colors = color_extractor.extract_dominant_colors_simple(image, 3)
                stats = color_extractor.get_color_statistics(image)
                all_colors.append((name, colors))
                all_stats.append((name, stats))
        
        st.markdown("#### 🎨 각 이미지별 주요 색상")
        
        # 각 이미지의 주요 색상 표시
        for name, colors in all_colors:
            st.markdown(f"**{name}**")
            
            color_row = ""
            for color in colors[:3]:
                color_row += f"""
                <div style="display: inline-block; margin: 5px;">
                    <div style="width: 50px; height: 50px; background-color: {color['hex']}; 
                                border: 1px solid #333; text-align: center; line-height: 50px;
                                color: {'white' if sum(color['rgb']) < 382 else 'black'}; font-size: 10px;">
                        {color['percentage']:.0f}%
                    </div>
                    <div style="text-align: center; font-size: 12px;">{color['hex']}</div>
                </div>
                """
            
            st.markdown(color_row, unsafe_allow_html=True)
            st.markdown("---")
        
        # 통계 요약
        st.markdown("#### 📊 이미지별 통계 요약")
        
        import pandas as pd
        
        summary_data = []
        for name, stats in all_stats:
            summary_data.append({
                '이미지': name,
                '평균 밝기': f"{stats['overall']['brightness']:.1f}",
                '대비': f"{stats['overall']['contrast']:.1f}",
                'R 평균': f"{stats['red']['mean']:.1f}",
                'G 평균': f"{stats['green']['mean']:.1f}",
                'B 평균': f"{stats['blue']['mean']:.1f}"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"다중 이미지 분석 중 오류가 발생했습니다: {str(e)}")

def get_similarity_level(score):
    """유사도 점수에 따른 레벨 반환"""
    if score >= 90:
        return "매우 높음"
    elif score >= 80:
        return "높음"
    elif score >= 60:
        return "보통"
    elif score >= 40:
        return "낮음"
    else:
        return "매우 낮음"

def generate_text_summary_report(parameters, results, similarity_score):
    """텍스트 요약 보고서 생성"""
    from datetime import datetime
    
    report = f"""
{'='*80}
                    색상 비교 분석 보고서
{'='*80}

분석 정보:
- 분석 일시: {parameters['timestamp']}
- 비교 ID: {parameters['comparison_id']}
- 분석 유형: 색상 유사도 비교

{'='*80}
이미지 정보
{'='*80}

[이미지 1]
- 파일명: {parameters['image1_info']['name']}
- 크기: {parameters['image1_info']['size'][0]} × {parameters['image1_info']['size'][1]} 픽셀
- 총 픽셀: {results['image1_analysis']['image_properties']['total_pixels']:,}개
- 종횡비: {results['image1_analysis']['image_properties']['aspect_ratio']}
- 색상 조화: {results['image1_analysis']['color_harmony']}

[이미지 2]  
- 파일명: {parameters['image2_info']['name']}
- 크기: {parameters['image2_info']['size'][0]} × {parameters['image2_info']['size'][1]} 픽셀
- 총 픽셀: {results['image2_analysis']['image_properties']['total_pixels']:,}개
- 종횡비: {results['image2_analysis']['image_properties']['aspect_ratio']}
- 색상 조화: {results['image2_analysis']['color_harmony']}

{'='*80}
색상 유사도 분석 결과
{'='*80}

전체 유사도: {similarity_score:.2f}% ({get_similarity_level(similarity_score)})

{'='*80}
주요 색상 분석
{'='*80}

[이미지 1 - 주요 색상]
"""
    
    for color in results['image1_analysis']['dominant_colors'][:5]:
        report += f"  {color['rank']}위: {color['hex']} (RGB: {color['rgb']}) - {color['percentage']:.2f}%\n"
    
    report += f"\n[이미지 2 - 주요 색상]\n"
    for color in results['image2_analysis']['dominant_colors'][:5]:
        report += f"  {color['rank']}위: {color['hex']} (RGB: {color['rgb']}) - {color['percentage']:.2f}%\n"
    
    report += f"""
{'='*80}
RGB 채널별 통계 비교
{'='*80}

Red 채널:
- 이미지1 평균: {results['image1_analysis']['color_statistics']['red_channel']['mean']:.1f}
- 이미지2 평균: {results['image2_analysis']['color_statistics']['red_channel']['mean']:.1f}
- 차이: {results['comparison_metrics']['red_mean_difference']:.1f}

Green 채널:
- 이미지1 평균: {results['image1_analysis']['color_statistics']['green_channel']['mean']:.1f}
- 이미지2 평균: {results['image2_analysis']['color_statistics']['green_channel']['mean']:.1f}
- 차이: {results['comparison_metrics']['green_mean_difference']:.1f}

Blue 채널:
- 이미지1 평균: {results['image1_analysis']['color_statistics']['blue_channel']['mean']:.1f}
- 이미지2 평균: {results['image2_analysis']['color_statistics']['blue_channel']['mean']:.1f}
- 차이: {results['comparison_metrics']['blue_mean_difference']:.1f}

{'='*80}
전체 이미지 특성 비교
{'='*80}

밝기:
- 이미지1: {results['image1_analysis']['color_statistics']['overall']['brightness']:.1f}
- 이미지2: {results['image2_analysis']['color_statistics']['overall']['brightness']:.1f}
- 차이: {results['comparison_metrics']['brightness_difference']:.1f}

대비:
- 이미지1: {results['image1_analysis']['color_statistics']['overall']['contrast']:.1f}
- 이미지2: {results['image2_analysis']['color_statistics']['overall']['contrast']:.1f}
- 차이: {results['comparison_metrics']['contrast_difference']:.1f}

색온도:
- 이미지1: {results['image1_analysis']['color_statistics']['overall']['temperature']:.0f}K
- 이미지2: {results['image2_analysis']['color_statistics']['overall']['temperature']:.0f}K
- 차이: {results['comparison_metrics']['temperature_difference']:.0f}K

{'='*80}
저장된 파일 정보
{'='*80}

원본 이미지:
- {results['file_paths']['original_image1']}
- {results['file_paths']['original_image2']}

썸네일 이미지:
- {results['file_paths']['thumbnail_image1']}
- {results['file_paths']['thumbnail_image2']}

저장 위치: {results['file_paths']['storage_directory']}

{'='*80}
보고서 생성 완료 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
    
    return report

def save_color_comparison_result(image1, image2, name1, name2, similarity_score, colors1, colors2, stats1, stats2):
    """색상 비교 분석 결과를 포괄적으로 저장"""
    try:
        st.info("🔄 결과를 저장하는 중...")
        
        # 현재 시간 정보
        from datetime import datetime
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        
        # 저장 디렉토리 생성 (날짜/시간별 구조)
        temp_dir = f"Result/color_comparison/{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        st.success(f"📁 저장 디렉토리 생성: {temp_dir}")
        
        import uuid
        comparison_id = str(uuid.uuid4())[:8]
        
        # 안전한 파일명 생성
        def clean_filename(name):
            name_without_ext = os.path.splitext(name)[0]
            ext = os.path.splitext(name)[1] if os.path.splitext(name)[1] else '.png'
            clean_name = "".join(c for c in name_without_ext if c.isalnum() or c in (' ', '_', '-')).strip()
            return f"{clean_name}{ext}"
        
        clean_name1 = clean_filename(name1)
        clean_name2 = clean_filename(name2)
        
        # 파일 경로 설정
        original_image1_path = os.path.join(temp_dir, f"original_1_{comparison_id}_{clean_name1}")
        original_image2_path = os.path.join(temp_dir, f"original_2_{comparison_id}_{clean_name2}")
        thumbnail_image1_path = os.path.join(temp_dir, f"thumb_1_{comparison_id}_{clean_name1}")
        thumbnail_image2_path = os.path.join(temp_dir, f"thumb_2_{comparison_id}_{clean_name2}")
        
        # 원본 이미지 저장 (고품질)
        image1.save(original_image1_path, quality=95, optimize=True)
        image2.save(original_image2_path, quality=95, optimize=True)
        
        # 썸네일 생성 및 저장 (빠른 로딩용)
        thumbnail1 = image1.copy()
        thumbnail2 = image2.copy()
        thumbnail1.thumbnail((300, 300), Image.Resampling.LANCZOS)
        thumbnail2.thumbnail((300, 300), Image.Resampling.LANCZOS)
        thumbnail1.save(thumbnail_image1_path, quality=85)
        thumbnail2.save(thumbnail_image2_path, quality=85)
        
        st.success("📸 이미지 저장 완료 (원본 + 썸네일)")
        
        # 색상 조화 정보 추출
        color_extractor = ColorExtractor()
        harmony1 = color_extractor.get_color_harmony_type(colors1)
        harmony2 = color_extractor.get_color_harmony_type(colors2)
        
        # 포괄적인 매개변수 저장
        parameters = {
            "comparison_id": comparison_id,
            "timestamp": current_time.isoformat(),
            "image1_info": {
                "original_name": name1,
                "clean_name": clean_name1,
                "size": list(image1.size),
                "mode": image1.mode,
                "format": getattr(image1, 'format', 'PNG'),
                "file_size_bytes": len(image1.tobytes())
            },
            "image2_info": {
                "original_name": name2,
                "clean_name": clean_name2,
                "size": list(image2.size),
                "mode": image2.mode,
                "format": getattr(image2, 'format', 'PNG'),
                "file_size_bytes": len(image2.tobytes())
            },
            "analysis_settings": {
                "colors_extracted": len(colors1),
                "comparison_method": "dominant_color_similarity",
                "analysis_date": current_time.strftime("%Y-%m-%d"),
                "analysis_time": current_time.strftime("%H:%M:%S")
            }
        }
        
        # 완전한 분석 결과 데이터
        results = {
            "summary": {
                "similarity_score": similarity_score,
                "similarity_level": get_similarity_level(similarity_score),
                "total_colors_analyzed": len(colors1) + len(colors2),
                "harmony_compatibility": harmony1 == harmony2
            },
            "image1_analysis": {
                "filename": name1,
                "dominant_colors": [
                    {
                        "rank": i+1,
                        "hex": color['hex'],
                        "rgb": list(color['rgb']),
                        "percentage": color['percentage'],
                        "pixel_count": color.get('count', 0)
                    } for i, color in enumerate(colors1)
                ],
                "color_statistics": {
                    "red_channel": dict(stats1['red']),
                    "green_channel": dict(stats1['green']),
                    "blue_channel": dict(stats1['blue']),
                    "overall": dict(stats1['overall'])
                },
                "color_harmony": harmony1,
                "image_properties": {
                    "width": image1.size[0],
                    "height": image1.size[1],
                    "total_pixels": image1.size[0] * image1.size[1],
                    "aspect_ratio": round(image1.size[0] / image1.size[1], 3)
                }
            },
            "image2_analysis": {
                "filename": name2,
                "dominant_colors": [
                    {
                        "rank": i+1,
                        "hex": color['hex'],
                        "rgb": list(color['rgb']),
                        "percentage": color['percentage'],
                        "pixel_count": color.get('count', 0)
                    } for i, color in enumerate(colors2)
                ],
                "color_statistics": {
                    "red_channel": dict(stats2['red']),
                    "green_channel": dict(stats2['green']),
                    "blue_channel": dict(stats2['blue']),
                    "overall": dict(stats2['overall'])
                },
                "color_harmony": harmony2,
                "image_properties": {
                    "width": image2.size[0],
                    "height": image2.size[1],
                    "total_pixels": image2.size[0] * image2.size[1],
                    "aspect_ratio": round(image2.size[0] / image2.size[1], 3)
                }
            },
            "comparison_metrics": {
                "brightness_difference": abs(stats1['overall']['brightness'] - stats2['overall']['brightness']),
                "contrast_difference": abs(stats1['overall']['contrast'] - stats2['overall']['contrast']),
                "temperature_difference": abs(stats1['overall']['temperature'] - stats2['overall']['temperature']),
                "red_mean_difference": abs(stats1['red']['mean'] - stats2['red']['mean']),
                "green_mean_difference": abs(stats1['green']['mean'] - stats2['green']['mean']),
                "blue_mean_difference": abs(stats1['blue']['mean'] - stats2['blue']['mean']),
                "harmony_match": harmony1 == harmony2,
                "size_ratio": min(image1.size[0] * image1.size[1], image2.size[0] * image2.size[1]) / max(image1.size[0] * image1.size[1], image2.size[0] * image2.size[1])
            },
            "file_paths": {
                "original_image1": original_image1_path,
                "original_image2": original_image2_path,
                "thumbnail_image1": thumbnail_image1_path,
                "thumbnail_image2": thumbnail_image2_path,
                "storage_directory": temp_dir
            }
        }
        
        st.success("📊 상세 분석 데이터 준비 완료")
        
        # JSON 상세 보고서 생성
        report_path = os.path.join(temp_dir, f"analysis_report_{comparison_id}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "version": "1.0",
                    "generated_by": "Image Similarity Analyzer",
                    "generated_at": current_time.isoformat()
                },
                "parameters": parameters,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        # 텍스트 요약 보고서 생성
        summary_report = generate_text_summary_report(parameters, results, similarity_score)
        summary_path = os.path.join(temp_dir, f"summary_report_{comparison_id}.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        # CSV 데이터 생성 (스프레드시트 분석용)
        csv_data = f"""분석항목,이미지1,이미지2,차이\n"""
        csv_data += f"""파일명,{name1},{name2},-\n"""
        csv_data += f"""크기,{image1.size[0]}×{image1.size[1]},{image2.size[0]}×{image2.size[1]},-\n"""
        csv_data += f"""밝기,{stats1['overall']['brightness']:.1f},{stats2['overall']['brightness']:.1f},{abs(stats1['overall']['brightness'] - stats2['overall']['brightness']):.1f}\n"""
        csv_data += f"""대비,{stats1['overall']['contrast']:.1f},{stats2['overall']['contrast']:.1f},{abs(stats1['overall']['contrast'] - stats2['overall']['contrast']):.1f}\n"""
        csv_data += f"""색온도,{stats1['overall']['temperature']:.0f}K,{stats2['overall']['temperature']:.0f}K,{abs(stats1['overall']['temperature'] - stats2['overall']['temperature']):.0f}K\n"""
        csv_data += f"""색상조화,{harmony1},{harmony2},{harmony1 == harmony2}\n"""
        csv_data += f"""유사도,-,-,{similarity_score:.1f}%\n"""
        
        csv_path = os.path.join(temp_dir, f"comparison_data_{comparison_id}.csv")
        with open(csv_path, 'w', encoding='utf-8-sig') as f:  # BOM 추가로 Excel 호환성 향상
            f.write(csv_data)
        
        st.success("📄 보고서 파일 생성 완료")
        
        # 데이터베이스에 저장
        work_id = save_work_history(
            work_type="color_comparison",
            title=f"색상 비교 분석 - {clean_name1} vs {clean_name2}",
            description=f"색상 유사도 {similarity_score:.1f}% ({get_similarity_level(similarity_score)})",
            input_images=json.dumps([original_image1_path, original_image2_path]),
            output_images=json.dumps([thumbnail_image1_path, thumbnail_image2_path, report_path, summary_path, csv_path]),
            parameters=json.dumps(parameters),
            results=json.dumps(results)
        )
        
        # 성공 메시지와 요약 정보
        st.success("🎉 색상 비교 결과가 성공적으로 저장되었습니다!")
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🆔 작업 ID", work_id[:8])
        with col2:
            st.metric("📊 유사도", f"{similarity_score:.1f}%")
        with col3:
            st.metric("📁 총 파일", "7개")
        with col4:
            st.metric("💾 저장 용량", f"{sum(os.path.getsize(f) for f in [original_image1_path, original_image2_path, report_path, summary_path, csv_path] if os.path.exists(f)) // 1024}KB")
        
        # 저장된 파일들 상세 정보
        with st.expander("📂 저장된 파일 상세 정보", expanded=True):
            st.markdown("**🖼️ 이미지 파일:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(thumbnail_image1_path):
                    st.image(thumbnail_image1_path, caption=f"이미지 1: {clean_name1}", width=150)
            with col2:
                if os.path.exists(thumbnail_image2_path):
                    st.image(thumbnail_image2_path, caption=f"이미지 2: {clean_name2}", width=150)
            
            st.markdown("**📄 생성된 파일 목록:**")
            files_info = [
                ("원본 이미지 1", os.path.basename(original_image1_path), f"{os.path.getsize(original_image1_path) // 1024}KB"),
                ("원본 이미지 2", os.path.basename(original_image2_path), f"{os.path.getsize(original_image2_path) // 1024}KB"),
                ("썸네일 1", os.path.basename(thumbnail_image1_path), f"{os.path.getsize(thumbnail_image1_path) // 1024}KB"),
                ("썸네일 2", os.path.basename(thumbnail_image2_path), f"{os.path.getsize(thumbnail_image2_path) // 1024}KB"),
                ("JSON 보고서", os.path.basename(report_path), f"{os.path.getsize(report_path) // 1024}KB"),
                ("텍스트 요약", os.path.basename(summary_path), f"{os.path.getsize(summary_path) // 1024}KB"),
                ("CSV 데이터", os.path.basename(csv_path), f"{os.path.getsize(csv_path)}B")
            ]
            
            for file_type, filename, size in files_info:
                st.write(f"- **{file_type}**: `{filename}` ({size})")
            
            st.info(f"**💾 저장 위치**: `{temp_dir}`")
        
        # 다운로드 섹션
        st.markdown("---")
        st.markdown("#### 📥 보고서 다운로드")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📋 텍스트 요약 보고서",
                data=summary_report,
                file_name=f"color_comparison_summary_{comparison_id}.txt",
                mime="text/plain",
                help="사람이 읽기 쉬운 텍스트 형식의 요약 보고서"
            )
        
        with col2:
            with open(report_path, 'r', encoding='utf-8') as f:
                json_report = f.read()
            st.download_button(
                label="📊 JSON 상세 데이터",
                data=json_report,
                file_name=f"color_comparison_detail_{comparison_id}.json",
                mime="application/json",
                help="모든 분석 데이터가 포함된 구조화된 JSON 파일"
            )
        
        with col3:
            st.download_button(
                label="📈 CSV 비교 데이터",
                data=csv_data,
                file_name=f"color_comparison_data_{comparison_id}.csv",
                mime="text/csv",
                help="Excel에서 열어볼 수 있는 비교 데이터"
            )
        
        # 코멘트 입력 섹션
        st.markdown("---")
        st.markdown("#### 💬 분석 코멘트 추가")
        comment = st.text_area(
            "이 색상 비교 분석에 대한 코멘트나 관찰 내용을 입력하세요:",
            key=f"comment_{work_id}",
            placeholder="예: 두 이미지는 전반적으로 따뜻한 색조를 보이지만, 첫 번째 이미지가 더 밝고 대비가 강합니다. 색상 조화 패턴은 유사하나 채도에서 차이를 보입니다.",
            height=100
        )
        
        if st.button("💬 코멘트 저장", key=f"save_comment_{work_id}"):
            if comment.strip():
                save_work_comment(work_id, comment)
                st.success("💬 코멘트가 저장되었습니다!")
            else:
                st.warning("코멘트를 입력해주세요.")
        
    except Exception as e:
        st.error(f"❌ 결과 저장 중 오류가 발생했습니다!")
        st.error(f"**오류 메시지**: {str(e)}")
        
        # 상세한 오류 정보
        import traceback
        with st.expander("🔍 상세 오류 정보"):
            st.code(traceback.format_exc())
        
        # 디버깅 정보
        with st.expander("🛠️ 디버깅 정보"):
            debug_info = {
                "이미지1 타입": str(type(image1)),
                "이미지2 타입": str(type(image2)),
                "이미지1 크기": str(getattr(image1, 'size', 'N/A')),
                "이미지2 크기": str(getattr(image2, 'size', 'N/A')),
                "파일명1": name1,
                "파일명2": name2,
                "유사도 점수": similarity_score,
                "색상1 개수": len(colors1) if colors1 else 'N/A',
                "색상2 개수": len(colors2) if colors2 else 'N/A'
            }
            
            for key, value in debug_info.items():
                st.write(f"- **{key}**: {value}")

def display_color_analysis_history():
    """색상 분석 결과 히스토리를 표시"""
    try:
        # 색상 분석 관련 작업 히스토리 가져오기
        color_history = get_work_history("color_analysis")
        comparison_history = get_work_history("color_comparison")
        
        # 두 히스토리 합치기
        import pandas as pd
        all_history = pd.concat([color_history, comparison_history], ignore_index=True)
        
        if len(all_history) == 0:
            st.info("저장된 색상 분석 결과가 없습니다.")
            return
        
        # 최신순으로 정렬
        all_history = all_history.sort_values('timestamp', ascending=False)
        
        st.markdown(f"### 📊 총 {len(all_history)}개의 색상 분석 결과")
        
        # 필터 옵션
        col1, col2 = st.columns(2)
        with col1:
            work_type_filter = st.selectbox(
                "작업 유형 필터",
                ["전체", "색상 분석", "색상 비교"],
                key="work_type_filter"
            )
        with col2:
            show_count = st.selectbox(
                "표시할 결과 수",
                [10, 20, 50, "전체"],
                key="show_count"
            )
        
        # 필터 적용
        if work_type_filter == "색상 분석":
            filtered_history = all_history[all_history['work_type'] == 'color_analysis']
        elif work_type_filter == "색상 비교":
            filtered_history = all_history[all_history['work_type'] == 'color_comparison']
        else:
            filtered_history = all_history
        
        # 표시할 개수 제한
        if show_count != "전체":
            filtered_history = filtered_history.head(int(show_count))
        
        # 결과 표시
        for idx, row in filtered_history.iterrows():
            with st.expander(f"🎨 {row['title']} - {row['timestamp'][:16]}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**설명:** {row['description']}")
                    st.write(f"**작업 ID:** {row['id'][:8]}")
                    st.write(f"**작업 유형:** {row['work_type']}")
                    st.write(f"**생성 시간:** {row['timestamp']}")
                    
                    if row['comment']:
                        st.write(f"**코멘트:** {row['comment']}")
                
                with col2:
                    # 입력 이미지가 있으면 표시
                    if row['input_images']:
                        try:
                            import json
                            input_paths = json.loads(row['input_images'])
                            if input_paths and len(input_paths) >= 2:
                                # 썸네일 이미지 경로 찾기
                                output_paths = []
                                if row['output_images']:
                                    try:
                                        output_paths = json.loads(row['output_images'])
                                    except:
                                        pass
                                
                                # 썸네일 이미지 우선 표시, 없으면 원본 표시
                                thumbnail_paths = [p for p in output_paths if 'thumb_' in p]
                                display_paths = thumbnail_paths[:2] if len(thumbnail_paths) >= 2 else input_paths[:2]
                                
                                for i, path in enumerate(display_paths):
                                    if os.path.exists(path):
                                        try:
                                            img = safe_image_open(path)
                                            st.image(img, caption=f"이미지 {i+1}", width=120)
                                        except:
                                            st.write(f"이미지 {i+1}: {os.path.basename(path)}")
                                    else:
                                        st.write(f"이미지 {i+1}: 파일 없음")
                            else:
                                st.write("이미지 정보 없음")
                        except:
                            st.write("이미지 정보를 불러올 수 없습니다.")
                
                # 결과 데이터 표시
                if row['results']:
                    with st.expander(f"📊 상세 결과 보기", expanded=False):
                        try:
                            import json
                            results_data = json.loads(row['results'])
                            
                            if row['work_type'] == 'color_comparison':
                                # 색상 비교 결과 표시
                                st.write(f"**색상 유사도:** {results_data.get('similarity_score', 0):.1f}%")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**첫 번째 이미지 주요 색상:**")
                                    if 'image1_colors' in results_data:
                                        for color in results_data['image1_colors'][:3]:
                                            color_box = f"""
                                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                                <div style="width: 20px; height: 20px; background-color: {color['hex']}; 
                                                            border: 1px solid #333; margin-right: 10px;"></div>
                                                <span>{color['hex']} ({color['percentage']:.1f}%)</span>
                                            </div>
                                            """
                                            st.markdown(color_box, unsafe_allow_html=True)
                                
                                with col2:
                                    st.write("**두 번째 이미지 주요 색상:**")
                                    if 'image2_colors' in results_data:
                                        for color in results_data['image2_colors'][:3]:
                                            color_box = f"""
                                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                                <div style="width: 20px; height: 20px; background-color: {color['hex']}; 
                                                            border: 1px solid #333; margin-right: 10px;"></div>
                                                <span>{color['hex']} ({color['percentage']:.1f}%)</span>
                                            </div>
                                            """
                                            st.markdown(color_box, unsafe_allow_html=True)
                                
                                # 통계 차이 표시
                                if 'brightness_diff' in results_data:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("밝기 차이", f"{results_data['brightness_diff']:.1f}")
                                    with col2:
                                        st.metric("대비 차이", f"{results_data['contrast_diff']:.1f}")
                                    with col3:
                                        st.metric("색온도 차이", f"{results_data['temperature_diff']:.0f}K")
                            
                            elif row['work_type'] == 'color_analysis':
                                # 단일 이미지 색상 분석 결과 표시
                                if 'dominant_colors' in results_data:
                                    st.write("**주요 색상:**")
                                    for color in results_data['dominant_colors'][:5]:
                                        color_box = f"""
                                        <div style="display: flex; align-items: center; margin: 5px 0;">
                                            <div style="width: 30px; height: 30px; background-color: {color['hex']}; 
                                                        border: 1px solid #333; margin-right: 10px;"></div>
                                            <span>{color['hex']} ({color['percentage']:.1f}%)</span>
                                        </div>
                                        """
                                        st.markdown(color_box, unsafe_allow_html=True)
                                
                                if 'color_temperature' in results_data:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("색온도", f"{results_data['color_temperature']:.0f}K")
                                    with col2:
                                        st.metric("평균 밝기", f"{results_data['avg_brightness']:.1f}")
                        
                        except Exception as e:
                            st.error(f"결과 데이터를 불러올 수 없습니다: {str(e)}")
                
                # 코멘트 추가/수정
                st.markdown("---")
                current_comment = row['comment'] if row['comment'] else ""
                new_comment = st.text_area(
                    "코멘트:", 
                    value=current_comment, 
                    key=f"edit_comment_{row['id']}"
                )
                
                if st.button(f"💬 코멘트 {'수정' if current_comment else '추가'}", key=f"update_comment_{row['id']}"):
                    if new_comment != current_comment:
                        save_work_comment(row['id'], new_comment)
                        st.success("코멘트가 저장되었습니다!")
                        st.rerun()
                
                # 결과 삭제 옵션
                if st.button(f"🗑️ 결과 삭제", key=f"delete_{row['id']}", type="secondary"):
                    if st.button(f"⚠️ 정말 삭제하시겠습니까?", key=f"confirm_delete_{row['id']}"):
                        delete_work_history(row['id'])
                        st.success("결과가 삭제되었습니다!")
                        st.rerun()
        
    except Exception as e:
        st.error(f"히스토리를 불러오는 중 오류가 발생했습니다: {str(e)}")

def delete_work_history(work_id):
    """작업 히스토리 삭제"""
    try:
        import sqlite3
        conn = sqlite3.connect('similarity_results.db')
        c = conn.cursor()
        c.execute("DELETE FROM work_history WHERE id = ?", (work_id,))
        conn.commit()
        conn.close()
        return True
    except:
        return False

if __name__ == "__main__":
    app()