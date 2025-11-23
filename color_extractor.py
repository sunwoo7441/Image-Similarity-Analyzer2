# color_extractor.py - RGB 색상 추출 모듈

import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import pandas as pd

class ColorExtractor:
    """이미지에서 RGB 색상을 추출하는 클래스"""
    
    def __init__(self):
        pass
    
    def extract_dominant_colors_simple(self, image, n_colors=5):
        """간단한 방법으로 주요 색상 추출 (K-means 없이)"""
        # PIL 이미지를 numpy 배열로 변환
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # 이미지를 2D 배열로 재구성 (각 픽셀을 RGB 값으로)
        pixels = img_array.reshape(-1, 3)
        
        # 색상 범위를 줄여서 유사한 색상들을 그룹화
        # RGB 값을 32로 나누어 8개 구간으로 나눔 (0-7 범위)
        reduced_pixels = (pixels // 32) * 32
        
        # 고유한 색상과 빈도 계산
        unique_colors = {}
        for pixel in reduced_pixels:
            color_key = tuple(pixel)
            unique_colors[color_key] = unique_colors.get(color_key, 0) + 1
        
        # 빈도순으로 정렬하여 상위 n_colors개 선택
        sorted_colors = sorted(unique_colors.items(), key=lambda x: x[1], reverse=True)
        top_colors = sorted_colors[:n_colors]
        
        # 결과 형식 맞추기
        total_pixels = len(pixels)
        color_info = []
        for color, count in top_colors:
            percentage = (count / total_pixels) * 100
            color_info.append({
                'color': np.array(color),
                'rgb': color,
                'hex': '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]),
                'count': count,
                'percentage': percentage
            })
        
        return color_info
    
    def get_color_temperature(self, image):
        """이미지의 색온도 추정"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # RGB 채널별 평균값
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1]) 
        b_mean = np.mean(img_array[:, :, 2])
        
        # 간단한 색온도 추정 (K)
        # 파란색이 강하면 차갑고, 빨간색이 강하면 따뜻함
        if b_mean > r_mean:
            # 차가운 색조
            temp = 6500 + (b_mean - r_mean) * 20
        else:
            # 따뜻한 색조  
            temp = 6500 - (r_mean - b_mean) * 20
            
        return max(2000, min(10000, temp))  # 2000K~10000K 범위로 제한
    
    def get_color_harmony_type(self, colors):
        """색상 조화 유형 분석"""
        if len(colors) < 2:
            return "단색"
        
        # HSV 변환을 위한 함수
        def rgb_to_hsv(r, g, b):
            r, g, b = r/255.0, g/255.0, b/255.0
            mx = max(r, g, b)
            mn = min(r, g, b)
            df = mx - mn
            
            if mx == mn:
                h = 0
            elif mx == r:
                h = (60 * ((g-b)/df) + 360) % 360
            elif mx == g:
                h = (60 * ((b-r)/df) + 120) % 360
            elif mx == b:
                h = (60 * ((r-g)/df) + 240) % 360
                
            s = 0 if mx == 0 else df/mx
            v = mx
            
            return h, s, v
        
        # 주요 색상들의 색상환 각도 계산
        hues = []
        for color in colors[:3]:  # 상위 3개 색상만 고려
            h, s, v = rgb_to_hsv(*color['rgb'])
            if s > 0.1:  # 채도가 너무 낮은 색상 제외
                hues.append(h)
        
        if len(hues) < 2:
            return "무채색"
        
        # 색상 각도 차이 분석
        hue_diffs = []
        for i in range(len(hues)-1):
            diff = abs(hues[i] - hues[i+1])
            if diff > 180:
                diff = 360 - diff
            hue_diffs.append(diff)
        
        avg_diff = sum(hue_diffs) / len(hue_diffs)
        
        if avg_diff < 30:
            return "유사색 조화"
        elif 150 < avg_diff < 210:
            return "보색 조화"
        elif 90 < avg_diff < 150:
            return "삼각 조화"
        else:
            return "복합 조화"
    
    def extract_color_palette(self, image, grid_size=10):
        """이미지를 격자로 나누어 각 영역의 평균 색상 추출"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        h, w, c = img_array.shape
        
        # 격자 크기 계산
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        palette = []
        positions = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 격자 영역 계산
                start_h = i * cell_h
                end_h = min((i + 1) * cell_h, h)
                start_w = j * cell_w
                end_w = min((j + 1) * cell_w, w)
                
                # 해당 영역의 평균 색상 계산
                cell = img_array[start_h:end_h, start_w:end_w]
                avg_color = np.mean(cell, axis=(0, 1)).astype(int)
                
                palette.append({
                    'position': (i, j),
                    'rgb': tuple(avg_color),
                    'hex': '#{:02x}{:02x}{:02x}'.format(avg_color[0], avg_color[1], avg_color[2])
                })
                positions.append((start_w, start_h, end_w, end_h))
        
        return palette, positions
    
    def get_color_statistics(self, image):
        """이미지의 색상 통계 정보 추출"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # RGB 채널별 통계
        r_channel = img_array[:, :, 0].flatten()
        g_channel = img_array[:, :, 1].flatten()
        b_channel = img_array[:, :, 2].flatten()
        
        # 색온도 계산
        temperature = self.get_color_temperature(image)
        
        stats = {
            'red': {
                'mean': float(np.mean(r_channel)),
                'std': float(np.std(r_channel)),
                'min': int(np.min(r_channel)),
                'max': int(np.max(r_channel)),
                'median': float(np.median(r_channel))
            },
            'green': {
                'mean': float(np.mean(g_channel)),
                'std': float(np.std(g_channel)),
                'min': int(np.min(g_channel)),
                'max': int(np.max(g_channel)),
                'median': float(np.median(g_channel))
            },
            'blue': {
                'mean': float(np.mean(b_channel)),
                'std': float(np.std(b_channel)),
                'min': int(np.min(b_channel)),
                'max': int(np.max(b_channel)),
                'median': float(np.median(b_channel))
            },
            'overall': {
                'brightness': float(np.mean(img_array)),
                'contrast': float(np.std(img_array)),
                'total_pixels': int(img_array.shape[0] * img_array.shape[1]),
                'temperature': temperature
            }
        }
        
        return stats
    
    def create_color_palette_visualization(self, color_info, title="Color Palette"):
        """색상 팔레트 시각화"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
        
        # 색상 팔레트 그리기
        for i, info in enumerate(color_info):
            color_rgb = [c/255.0 for c in info['rgb']]  # matplotlib은 0-1 범위
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, 
                                   edgecolor='black', facecolor=color_rgb)
            ax.add_patch(rect)
            
            # 색상 정보 텍스트 추가
            ax.text(i + 0.5, 0.5, f"{info['percentage']:.1f}%", 
                   ha='center', va='center', fontsize=8, 
                   color='white' if sum(info['rgb']) < 382 else 'black')
        
        ax.set_xlim(0, len(color_info))
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xticks(range(len(color_info)))
        ax.set_xticklabels([info['hex'] for info in color_info], rotation=45)
        ax.set_yticks([])
        
        plt.tight_layout()
        return fig

def display_color_analysis_ui(image, prefix=""):
    """색상 분석 UI 표시"""
    st.markdown("### 🎨 RGB 색상 분석")
    
    color_extractor = ColorExtractor()
    
    # 분석 옵션
    col1, col2 = st.columns(2)
    with col1:
        n_colors = st.slider(
            "추출할 주요 색상 개수", 
            min_value=3, max_value=10, value=5, 
            key=f"n_colors_{prefix}"
        )
    with col2:
        analysis_type = st.radio(
            "분석 유형",
            ["주요 색상", "색상 통계", "모두"],
            key=f"analysis_type_{prefix}",
            horizontal=True
        )
    
    if st.button(f"🎨 색상 분석 시작", key=f"color_analysis_{prefix}"):
        with st.spinner("색상을 분석하는 중..."):
            try:
                if analysis_type in ["주요 색상", "모두"]:
                    # 주요 색상 추출
                    dominant_colors = color_extractor.extract_dominant_colors_simple(image, n_colors)
                    
                    st.markdown("#### 🌈 주요 색상 정보")
                    
                    # 색상 팔레트 시각화
                    fig = color_extractor.create_color_palette_visualization(
                        dominant_colors, f"주요 색상 팔레트 (상위 {n_colors}개)"
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                    # 색상 정보 테이블
                    color_data = []
                    for i, info in enumerate(dominant_colors):
                        color_data.append({
                            '순위': i + 1,
                            'RGB': f"({info['rgb'][0]}, {info['rgb'][1]}, {info['rgb'][2]})",
                            'HEX': info['hex'],
                            '비율(%)': f"{info['percentage']:.2f}%",
                            '픽셀 수': f"{info['count']:,}"
                        })
                    
                    df = pd.DataFrame(color_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # 개별 색상 표시
                    st.markdown("#### 🎯 개별 색상 상세 정보")
                    cols = st.columns(min(len(dominant_colors), 5))
                    for i, info in enumerate(dominant_colors[:5]):
                        with cols[i]:
                            # 색상 박스 HTML
                            color_box = f"""
                            <div style="
                                background-color: {info['hex']};
                                width: 100%;
                                height: 80px;
                                border: 2px solid #333;
                                border-radius: 8px;
                                margin-bottom: 10px;
                            "></div>
                            """
                            st.markdown(color_box, unsafe_allow_html=True)
                            st.write(f"**#{i+1} 색상**")
                            st.write(f"RGB: {info['rgb']}")
                            st.write(f"HEX: {info['hex']}")
                            st.write(f"비율: {info['percentage']:.1f}%")
                
                if analysis_type in ["색상 통계", "모두"]:
                    # 색상 통계 정보
                    stats = color_extractor.get_color_statistics(image)
                    
                    st.markdown("#### 📊 RGB 채널별 통계")
                    
                    # 통계 정보 테이블
                    stats_data = []
                    for channel, data in stats.items():
                        if channel != 'overall':
                            stats_data.append({
                                '채널': channel.upper(),
                                '평균': f"{data['mean']:.1f}",
                                '표준편차': f"{data['std']:.1f}",
                                '최솟값': data['min'],
                                '최댓값': data['max'],
                                '중간값': f"{data['median']:.1f}"
                            })
                    
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats, use_container_width=True)
                    
                    # 전체 이미지 통계
                    st.markdown("#### 📈 전체 이미지 통계")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("평균 밝기", f"{stats['overall']['brightness']:.1f}")
                    with col2:
                        st.metric("대비 (표준편차)", f"{stats['overall']['contrast']:.1f}")
                    with col3:
                        st.metric("총 픽셀 수", f"{stats['overall']['total_pixels']:,}")
                    
                    # RGB 히스토그램
                    st.markdown("#### 📊 RGB 히스토그램")
                    img_array = np.array(image)
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    colors = ['red', 'green', 'blue']
                    channels = [img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]]
                    
                    for i, (channel, color) in enumerate(zip(channels, colors)):
                        axes[i].hist(channel.flatten(), bins=50, color=color, alpha=0.7)
                        axes[i].set_title(f'{color.upper()} 채널 히스토그램')
                        axes[i].set_xlabel('픽셀 값 (0-255)')
                        axes[i].set_ylabel('빈도')
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
            except Exception as e:
                st.error(f"색상 분석 중 오류가 발생했습니다: {str(e)}")
    
    return None