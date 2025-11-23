import streamlit as st
from PIL import Image
import pandas as pd
import os
import sys
import datetime
import numpy as np
import json

# 현재 디렉토리를 상위 폴더로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db_utils import get_results, save_comment, get_work_history, save_work_comment

# 이미지를 정사각형으로 리사이징하는 함수
def resize_and_pad(img, size=120, bg_color=(255, 255, 255)):
    """이미지를 정확한 정사각형 크기로 패딩하여 리사이징"""
    # 원본 비율 유지하면서 thumbnail 크기로 축소
    img_copy = img.copy()
    img_copy.thumbnail((size, size), Image.LANCZOS)
    
    # 빈 정사각형 이미지 생성 (흰색 배경)
    new_img = Image.new('RGB', (size, size), bg_color)
    
    # 중앙에 배치
    new_img.paste(img_copy, ((size - img_copy.width) // 2, (size - img_copy.height) // 2))
    
    return new_img

def app():
    st.title("📊 작업 결과 요약")
    
    # 탭으로 구분
    tab1, tab2, tab3 = st.tabs(["🖼️ 이미지 비교", "🎨 색상 분석 & 업스케일링", "📈 전체 통계"])
    
    with tab1:
        st.markdown("### 이미지 유사도 비교 결과")
        display_similarity_results()
    
    with tab2:
        st.markdown("### 색상 분석 및 이미지 업스케일링 작업")
        display_work_history()
    
    with tab3:
        st.markdown("### 전체 작업 통계")
        display_overall_statistics()

def display_similarity_results():
    """이미지 유사도 비교 결과 표시"""
    # 데이터베이스에서 결과 가져오기
    try:
        results = get_results()
        if len(results) > 0:
            # comment 컬럼이 없으면 빈 컬럼 추가
            if 'comment' not in results.columns:
                results['comment'] = None
                
            # 탭으로 결과 요약 내용 구분
            sub_tab1, sub_tab2 = st.tabs(["결과 테이블", "썸네일 보기"])
            
            with sub_tab1:
                display_similarity_table(results)
            
            with sub_tab2:
                display_similarity_thumbnails(results)
        else:
            st.info("저장된 비교 결과가 없습니다.")
    except Exception as e:
        st.error(f"결과를 불러오는 중 오류가 발생했습니다: {str(e)}")

def display_work_history():
    """작업 히스토리 표시"""
    try:
        # 작업 유형 필터
        work_type_filter = st.selectbox(
            "작업 유형 필터",
            ["전체", "색상 분석", "이미지 업스케일링"],
            key="work_type_filter"
        )
        
        work_type_map = {
            "전체": None,
            "색상 분석": "color_analysis",
            "이미지 업스케일링": "image_upscaling"
        }
        
        # 작업 히스토리 가져오기
        work_history = get_work_history(work_type_map[work_type_filter])
        
        if len(work_history) > 0:
            st.markdown(f"### 📋 {work_type_filter} 작업 내역 ({len(work_history)}건)")
            
            # 작업 카드 형태로 표시
            for _, work in work_history.iterrows():
                with st.expander(f"🔧 {work['title']} - {pd.to_datetime(work['timestamp']).strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**설명:** {work['description']}")
                        
                        # 매개변수 표시
                        if work['parameters']:
                            try:
                                params = json.loads(work['parameters'])
                                st.write("**매개변수:**")
                                for key, value in params.items():
                                    if value is not None:
                                        st.write(f"- {key}: {value}")
                            except:
                                pass
                        
                        # 결과 표시
                        if work['results']:
                            try:
                                results = json.loads(work['results'])
                                st.write("**결과:**")
                                for key, value in results.items():
                                    st.write(f"- {key}: {value}")
                            except:
                                pass
                        
                        # 코멘트 표시 및 편집
                        current_comment = work['comment'] if pd.notna(work['comment']) else ""
                        new_comment = st.text_area(
                            "코멘트", 
                            value=current_comment, 
                            key=f"work_comment_{work['id']}"
                        )
                        
                        if st.button("코멘트 저장", key=f"save_work_comment_{work['id']}"):
                            save_work_comment(work['id'], new_comment)
                            st.success("코멘트가 저장되었습니다!")
                            st.rerun()
                    
                    with col2:
                        # 입력/출력 이미지 표시
                        if work['input_images']:
                            try:
                                input_paths = json.loads(work['input_images'])
                                for i, path in enumerate(input_paths[:2]):  # 최대 2개만 표시
                                    if os.path.exists(path):
                                        img = Image.open(path)
                                        st.image(img, caption=f"입력 이미지 {i+1}", width=150)
                            except Exception as e:
                                st.warning(f"입력 이미지를 불러올 수 없습니다: {str(e)}")
                        
                        if work['output_images']:
                            try:
                                output_paths = json.loads(work['output_images'])
                                for i, path in enumerate(output_paths[:2]):  # 최대 2개만 표시
                                    if os.path.exists(path):
                                        img = Image.open(path)
                                        st.image(img, caption=f"출력 이미지 {i+1}", width=150)
                            except Exception as e:
                                st.warning(f"출력 이미지를 불러올 수 없습니다: {str(e)}")
        else:
            st.info(f"{work_type_filter} 작업 내역이 없습니다.")
            
    except Exception as e:
        st.error(f"작업 히스토리를 불러오는 중 오류가 발생했습니다: {str(e)}")

def display_overall_statistics():
    """전체 작업 통계 표시"""
    try:
        # 이미지 비교 결과 통계
        similarity_results = get_results()
        work_history = get_work_history()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("이미지 비교", f"{len(similarity_results)}건")
        
        with col2:
            color_analysis_count = len(work_history[work_history['work_type'] == 'color_analysis']) if len(work_history) > 0 else 0
            st.metric("색상 분석", f"{color_analysis_count}건")
        
        with col3:
            upscaling_count = len(work_history[work_history['work_type'] == 'image_upscaling']) if len(work_history) > 0 else 0
            st.metric("이미지 업스케일링", f"{upscaling_count}건")
        
        # 시간별 작업 통계
        if len(work_history) > 0 or len(similarity_results) > 0:
            st.markdown("### 📈 일별 작업 통계")
            
            # 모든 작업을 하나의 데이터프레임으로 합치기
            all_works = []
            
            # 이미지 비교 결과 추가
            for _, row in similarity_results.iterrows():
                all_works.append({
                    'date': pd.to_datetime(row['timestamp']).date(),
                    'type': 'image_comparison'
                })
            
            # 작업 히스토리 추가
            for _, row in work_history.iterrows():
                all_works.append({
                    'date': pd.to_datetime(row['timestamp']).date(),
                    'type': row['work_type']
                })
            
            if all_works:
                df = pd.DataFrame(all_works)
                
                # 일별 작업 수 계산
                daily_stats = df.groupby(['date', 'type']).size().unstack(fill_value=0)
                
                # 한글 컬럼명으로 변경
                column_mapping = {
                    'image_comparison': '이미지 비교',
                    'color_analysis': '색상 분석',
                    'image_upscaling': '이미지 업스케일링'
                }
                
                daily_stats = daily_stats.rename(columns=column_mapping)
                
                # 차트 표시
                st.bar_chart(daily_stats)
                
                # 상세 테이블
                st.markdown("### 📊 일별 상세 통계")
                daily_stats['총합'] = daily_stats.sum(axis=1)
                st.dataframe(daily_stats.sort_index(ascending=False))
        
    except Exception as e:
        st.error(f"통계를 불러오는 중 오류가 발생했습니다: {str(e)}")

def display_similarity_table(results):
    """유사도 결과 테이블 표시"""
    
    # 데이터베이스에서 결과 가져오기
    try:
        results = get_results()
        if len(results) > 0:
            # comment 컬럼이 없으면 빈 컬럼 추가
            if 'comment' not in results.columns:
                results['comment'] = None
                
            # 탭으로 결과 요약 내용 구분
            tab1, tab2 = st.tabs(["결과 테이블", "썸네일 보기"])
            
            with tab1:
                st.markdown("## 유사도 결과 통계 및 상세 표")
                
                # 통계용 데이터 처리
                avg_ssim = results['ssim_score'].mean()
                avg_psnr = results['psnr_score'].mean()
                avg_vgg = results['vgg_score'].mean()
                avg_score = results['avg_score'].mean()
                
                max_ssim_idx = results['ssim_score'].idxmax()
                max_psnr_idx = results['psnr_score'].idxmax()
                max_vgg_idx = results['vgg_score'].idxmax()
                max_score_idx = results['avg_score'].idxmax()
                
                min_ssim_idx = results['ssim_score'].idxmin()
                min_psnr_idx = results['psnr_score'].idxmin()
                min_vgg_idx = results['vgg_score'].idxmin()
                min_score_idx = results['avg_score'].idxmin()
                
                # 통계 데이터프레임 생성
                stats_df = pd.DataFrame({
                    "분석": ["평균 유사도", "최대 유사도", "최소 유사도"],
                    "SSIM 점수": [
                        f"{avg_ssim:.2f}%",
                        f"{results.loc[max_ssim_idx, 'ssim_score']:.2f}%",
                        f"{results.loc[min_ssim_idx, 'ssim_score']:.2f}%"
                    ],
                    "PSNR 점수": [
                        f"{avg_psnr:.2f}%",
                        f"{results.loc[max_psnr_idx, 'psnr_score']:.2f}%",
                        f"{results.loc[min_psnr_idx, 'psnr_score']:.2f}%"
                    ],
                    "VGG 점수": [
                        f"{avg_vgg:.2f}%",
                        f"{results.loc[max_vgg_idx, 'vgg_score']:.2f}%",
                        f"{results.loc[min_vgg_idx, 'vgg_score']:.2f}%"
                    ],
                    "평균 점수": [
                        f"{avg_score:.2f}%",
                        f"{results.loc[max_score_idx, 'avg_score']:.2f}%",
                        f"{results.loc[min_score_idx, 'avg_score']:.2f}%"
                    ]
                })
                
                # 통계 테이블 표시
                st.write("### 유사도 통계 요약")
                st.dataframe(
                    stats_df,
                    column_config={
                        "분석": st.column_config.TextColumn("분석"),
                        "SSIM 점수": st.column_config.TextColumn("SSIM"),
                        "PSNR 점수": st.column_config.TextColumn("PSNR"),
                        "VGG 점수": st.column_config.TextColumn("VGG"),
                        "평균 점수": st.column_config.TextColumn("평균")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # 전체 데이터 정렬 기준 선택
                st.write("### 전체 결과 목록")
                sort_by = st.selectbox(
                    "정렬 기준",
                    options=["평균 점수", "SSIM 점수", "PSNR 점수", "VGG 점수", "날짜"],
                    key="table_sort_by"
                )
                
                sort_col_map = {
                    "평균 점수": "avg_score",
                    "SSIM 점수": "ssim_score",
                    "PSNR 점수": "psnr_score",
                    "VGG 점수": "vgg_score",
                    "날짜": "timestamp"
                }
                
                # 내림차순으로 정렬
                sorted_results = results.sort_values(by=sort_col_map[sort_by], ascending=False)
                
                # 표시할 컬럼 설정
                display_df = pd.DataFrame({
                    "날짜": pd.to_datetime(sorted_results['timestamp']).dt.strftime("%Y-%m-%d %H:%M"),
                    "SSIM 점수": sorted_results['ssim_score'],
                    "PSNR 점수": sorted_results['psnr_score'],
                    "VGG 점수": sorted_results['vgg_score'],
                    "평균 점수": sorted_results['avg_score'],
                    "댓글": sorted_results['comment'].fillna('')
                })
                
                # 댓글 있는 행에 표시 스타일 적용
                def highlight_commented(row):
                    has_comment = pd.notna(row['댓글']) and row['댓글'] != ''
                    return ['background-color: #fff0f0; font-weight: bold' if has_comment else '' for _ in row]
                
                # 결과 테이블 표시
                st.dataframe(
                    display_df.style.apply(highlight_commented, axis=1),
                    column_config={
                        "날짜": st.column_config.TextColumn("날짜"),
                        "SSIM 점수": st.column_config.NumberColumn("SSIM", format="%.2f%%"),
                        "PSNR 점수": st.column_config.NumberColumn("PSNR", format="%.2f%%"),
                        "VGG 점수": st.column_config.NumberColumn("VGG", format="%.2f%%"),
                        "평균 점수": st.column_config.NumberColumn("평균", format="%.2f%%"),
                        "댓글": st.column_config.TextColumn("댓글"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # CSV 다운로드 버튼
                csv = sorted_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="결과를 CSV로 다운로드",
                    data=csv,
                    file_name=f"similarity_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
                
                # 통계 그래프 - 유사도 분포와 추이
                st.write("### 유사도 점수 추이 그래프")

                # 그래프에 표시할 항목 선택
                st.write("표시할 유사도 지표 선택:")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    show_ssim = st.checkbox("SSIM", value=True)
                with col2:
                    show_psnr = st.checkbox("PSNR", value=True)
                with col3:
                    show_vgg = st.checkbox("VGG", value=True)
                with col4:
                    show_avg = st.checkbox("평균 점수", value=True)

                # 결과가 있는 경우만 그래프 표시
                if len(results) > 1:
                    # 데이터 준비 - 날짜순으로 정렬
                    chart_data = results.sort_values(by='timestamp').copy().reset_index(drop=True)
                    
                    # 그래프용 데이터프레임 준비 (x축을 실험 번호로 변경)
                    graph_data = pd.DataFrame()
                    graph_data['실험 번호'] = chart_data.index + 1  # 1부터 시작하는 실험 번호
                    
                    # 선택된 지표만 추가
                    if show_ssim:
                        graph_data['SSIM'] = chart_data['ssim_score']
                    if show_psnr:
                        graph_data['PSNR'] = chart_data['psnr_score']
                    if show_vgg:
                        graph_data['VGG'] = chart_data['vgg_score']
                    if show_avg:
                        graph_data['평균'] = chart_data['avg_score']
                    
                    # 선 그래프 표시 - 추이를 보여줌 (x축: 실험 번호)
                    st.write("#### 실험 번호에 따른 유사도 점수 추이")
                    st.line_chart(graph_data.set_index('실험 번호'))

                    # 인터랙티브 실험 결과 보기 기능 추가
                    st.write("#### 실험 결과 상세 보기")
                    st.write("실험 번호를 선택하면 해당 결과의 이미지와 유사도를 확인할 수 있습니다.")

                    # 실험 번호 선택 슬라이더
                    selected_experiment = st.slider("실험 번호 선택", 
                                                   min_value=1, 
                                                   max_value=len(chart_data), 
                                                   value=1)

                    # 선택한 실험 번호에 해당하는 행 찾기 (인덱스는 0부터 시작하므로 1을 빼줌)
                    selected_row = chart_data.iloc[selected_experiment-1]

                    # 선택한 실험 결과 표시
                    col1, col2 = st.columns([3, 2])

                    with col1:
                        try:
                            # 이미지 로드
                            real_img = Image.open(selected_row['real_image_path'])
                            ai_img = Image.open(selected_row['ai_image_path'])
                                
                            # 이미지 크기 조정
                            PREVIEW_SIZE = 200
                            real_thumb = resize_and_pad(real_img, size=PREVIEW_SIZE)
                            ai_thumb = resize_and_pad(ai_img, size=PREVIEW_SIZE)
                                
                            # 이미지 표시
                            st.write("##### 선택한 실험의 이미지")
                            img_cols = st.columns(2)
                            with img_cols[0]:
                                st.image(real_thumb, caption="실제 사진", width=PREVIEW_SIZE)
                            with img_cols[1]:
                                st.image(ai_thumb, caption="AI 사진", width=PREVIEW_SIZE)
                        except Exception as e:
                            st.error(f"이미지를 불러올 수 없습니다: {str(e)}")

                    with col2:
                        # 선택한 실험의 유사도 점수 표시
                        st.write("##### 유사도 점수")
                        selected_scores = pd.DataFrame({
                            "지표": ["SSIM", "PSNR", "VGG", "평균"],
                            "점수": [
                                f"{selected_row['ssim_score']:.2f}%",
                                f"{selected_row['psnr_score']:.2f}%",
                                f"{selected_row['vgg_score']:.2f}%",
                                f"{selected_row['avg_score']:.2f}%"
                            ]
                        })
                        st.dataframe(selected_scores, hide_index=True, use_container_width=True)
                        
                        # 날짜 및 추가 정보
                        st.write(f"**날짜:** {pd.to_datetime(selected_row['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                        
                        # 코멘트가 있으면 표시
                        if pd.notna(selected_row['comment']) and selected_row['comment'] != '':
                            st.info(f"💬 **코멘트:** {selected_row['comment']}")
                            
                        # 상세보기 버튼
                        if st.button("상세보기", key=f"popup_view_{selected_row['id']}"):
                            st.session_state['selected_result_id'] = selected_row['id']
                            st.experimental_rerun()
                    
                    # 추가: 실험 번호와 날짜 매핑 표시
                    with st.expander("실험 번호와 날짜 매핑 확인"):
                        mapping_df = pd.DataFrame({
                            "실험 번호": graph_data['실험 번호'],
                            "날짜": pd.to_datetime(chart_data['timestamp']).dt.strftime('%Y-%m-%d %H:%M'),
                            "평균 유사도": chart_data['avg_score'].round(2)
                        })
                        st.dataframe(mapping_df, use_container_width=True)
                    
                    # 요약 통계 표시 - 막대 그래프
                    st.write("#### 유사도 평가 방법별 평균")
                    summary_data = {}
                    
                    if show_ssim:
                        summary_data['SSIM'] = avg_ssim
                    if show_psnr:
                        summary_data['PSNR'] = avg_psnr
                    if show_vgg:
                        summary_data['VGG'] = avg_vgg
                    if show_avg:
                        summary_data['평균'] = avg_score
                        
                    if summary_data:  # 선택된 항목이 하나라도 있는 경우
                        st.bar_chart(pd.DataFrame([summary_data]))
                    
                    # 분포도 표시 - X축을 유사도 점수 구간으로 하는 히스토그램
                    st.write("#### 유사도 점수 분포")
                    if any([show_ssim, show_psnr, show_vgg, show_avg]):
                        # 표시할 열의 수 결정
                        num_cols = sum([show_ssim, show_psnr, show_vgg, show_avg])
                        dist_cols = st.columns(num_cols)
                        
                        idx = 0
                        if show_ssim:
                            with dist_cols[idx]:
                                st.write("SSIM 분포")
                                # 히스토그램 데이터 생성 (x축: 점수 구간, y축: 빈도)
                                hist_values, bin_edges = np.histogram(chart_data['ssim_score'], bins=10, range=(0, 100))
                                hist_df = pd.DataFrame({
                                    "구간": [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                                    "빈도": hist_values
                                })
                                st.bar_chart(hist_df.set_index("구간"))
                            idx += 1
                        
                        if show_psnr:
                            with dist_cols[idx]:
                                st.write("PSNR 분포")
                                hist_values, bin_edges = np.histogram(chart_data['psnr_score'], bins=10, range=(0, 100))
                                hist_df = pd.DataFrame({
                                    "구간": [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                                    "빈도": hist_values
                                })
                                st.bar_chart(hist_df.set_index("구간"))
                            idx += 1
                        
                        if show_vgg:
                            with dist_cols[idx]:
                                st.write("VGG 분포")
                                hist_values, bin_edges = np.histogram(chart_data['vgg_score'], bins=10, range=(0, 100))
                                hist_df = pd.DataFrame({
                                    "구간": [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                                    "빈도": hist_values
                                })
                                st.bar_chart(hist_df.set_index("구간"))
                            idx += 1
                        
                        if show_avg:
                            with dist_cols[idx]:
                                st.write("평균 점수 분포")
                                hist_values, bin_edges = np.histogram(chart_data['avg_score'], bins=10, range=(0, 100))
                                hist_df = pd.DataFrame({
                                    "구간": [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                                    "빈도": hist_values
                                })
                                st.bar_chart(hist_df.set_index("구간"))
                else:
                    st.info("추이 그래프를 표시하려면 최소 2개 이상의 결과가 필요합니다.")
            
            with tab2:
                st.markdown("### 모든 비교 결과의 썸네일과 유사도 점수")
                
                # 썸네일 사이즈 설정 (5열 배열을 위해 크기 줄임)
                THUMBNAIL_SIZE = 120
                
                # 필터링 옵션
                col1, col2, col3 = st.columns(3)
                with col1:
                    sort_by = st.selectbox(
                        "정렬 기준",
                        options=["최신순", "유사도 높은순", "유사도 낮은순"]
                    )
                with col2:
                    min_similarity = st.slider("최소 유사도(%)", 0, 100, 0, 5)
                with col3:
                    view_mode = st.radio("보기 모드", ["썸네일", "상세"])
                
                # 결과 필터링 및 정렬
                filtered_results = results[results['avg_score'] >= min_similarity]
                
                if sort_by == "최신순":
                    filtered_results = filtered_results.sort_values('timestamp', ascending=False)
                elif sort_by == "유사도 높은순":
                    filtered_results = filtered_results.sort_values('avg_score', ascending=False)
                else:  # 유사도 낮은순
                    filtered_results = filtered_results.sort_values('avg_score', ascending=True)
                
                # 결과 표시
                if len(filtered_results) > 0:
                    if view_mode == "썸네일":
                        # 썸네일 그리드 표시 (5열 변경)
                        st.write("### 비교 결과 썸네일")
                        
                        # CSS 스타일 추가 - 썸네일 카드 스타일링
                        st.markdown("""
                        <style>
                        .thumbnail-card {
                            border: 1px solid #e6e6e6;
                            border-radius: 5px;
                            padding: 5px;
                            margin-bottom: 15px;
                            text-align: center;
                            background-color: #ffffff;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }
                        .thumb-container {
                            height: 120px;
                            width: 120px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin: 0 auto;
                        }
                        .similarity-badge {
                            background-color: #f0f2f6;
                            border-radius: 4px;
                            padding: 2px 6px;
                            margin: 5px 0;
                            display: inline-block;
                            font-weight: bold;
                        }
                        .date-text {
                            color: #666;
                            font-size: 0.8em;
                            margin-bottom: 5px;
                        }
                        .comment-icon {
                            color: red;
                            font-weight: bold;
                            margin-left: 5px;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # 5열 그리드로 썸네일 표시
                        for i in range(0, len(filtered_results), 5):
                            cols = st.columns(5)
                            for j in range(5):
                                if i + j < len(filtered_results):
                                    row = filtered_results.iloc[i + j]
                                    with cols[j]:
                                        try:
                                            # 타임스탬프 포맷팅
                                            timestamp = pd.to_datetime(row['timestamp'])
                                            date_str = timestamp.strftime("%Y-%m-%d")
                                            
                                            # 코멘트 아이콘 표시 여부
                                            has_comment = pd.notna(row['comment']) and row['comment'] != ''
                                            comment_icon = ' <span class="comment-icon">💬</span>' if has_comment else ''
                                            
                                            # 썸네일 생성
                                            real_img = Image.open(row['real_image_path'])
                                            ai_img = Image.open(row['ai_image_path'])
                                            
                                            # 이미지 크기 조정 (썸네일)
                                            real_thumb = resize_and_pad(real_img, size=THUMBNAIL_SIZE)
                                            ai_thumb = resize_and_pad(ai_img, size=THUMBNAIL_SIZE)
                                            
                                            # 썸네일 헤더
                                            st.markdown(f"#### #{i+j+1}{comment_icon}", unsafe_allow_html=True)
                                            
                                            # 이미지 표시
                                            st.image(real_thumb, caption="실제 사진", width=THUMBNAIL_SIZE)
                                            st.image(ai_thumb, caption="AI 사진", width=THUMBNAIL_SIZE)
                                            
                                            # 유사도 및 날짜 표시
                                            st.markdown(f"<div class='similarity-badge'>유사도: {row['avg_score']:.1f}%</div>", unsafe_allow_html=True)
                                            st.markdown(f"<div class='date-text'>{date_str}</div>", unsafe_allow_html=True)
                                            
                                            if st.button("상세보기", key=f"view_{row['id']}"):
                                                st.session_state['selected_result_id'] = row['id']
                                                st.experimental_rerun()
                                        except Exception as e:
                                            st.error(f"이미지 오류: {str(e)}")
                    else:  # 상세 모드
                        # 결과를 테이블로 표시
                        st.write("### 비교 결과 상세 정보")
                        
                        # 데이터프레임 준비
                        display_df = pd.DataFrame({
                            "날짜": pd.to_datetime(filtered_results['timestamp']).dt.strftime("%Y-%m-%d %H:%M"),
                            "SSIM 점수": filtered_results['ssim_score'].round(2),
                            "PSNR 점수": filtered_results['psnr_score'].round(2),
                            "VGG 점수": filtered_results['vgg_score'].round(2),
                            "평균 점수": filtered_results['avg_score'].round(2),
                            "댓글": filtered_results['comment'].fillna('')
                        })
                        
                        st.dataframe(
                            display_df,
                            column_config={
                                "날짜": st.column_config.TextColumn("날짜"),
                                "SSIM 점수": st.column_config.NumberColumn("SSIM", format="%.2f%%"),
                                "PSNR 점수": st.column_config.NumberColumn("PSNR", format="%.2f%%"),
                                "VGG 점수": st.column_config.NumberColumn("VGG", format="%.2f%%"),
                                "평균 점수": st.column_config.NumberColumn("평균", format="%.2f%%"),
                                "댓글": st.column_config.TextColumn("댓글"),
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # 선택된 결과 상세보기
                    if 'selected_result_id' in st.session_state:
                        selected_id = st.session_state['selected_result_id']
                        selected_row = filtered_results[filtered_results['id'] == selected_id]
                        
                        if not selected_row.empty:
                            row = selected_row.iloc[0]
                            st.write("---")
                            st.write("## 선택된 결과 상세보기")
                            try:
                                col1, col2 = st.columns(2)
                                with col1:
                                    real_img = Image.open(row['real_image_path'])
                                    st.image(real_img, caption="실제 사진", use_column_width=True)
                                with col2:
                                    ai_img = Image.open(row['ai_image_path'])
                                    st.image(ai_img, caption="AI 사진", use_column_width=True)
                                
                                # 점수 표시
                                st.write("### 유사도 점수")
                                scores_df = pd.DataFrame({
                                    "점수 유형": ["SSIM", "PSNR", "VGG16", "평균"],
                                    "점수 (%)": [
                                        f"{row['ssim_score']:.2f}%", 
                                        f"{row['psnr_score']:.2f}%", 
                                        f"{row['vgg_score']:.2f}%", 
                                        f"{row['avg_score']:.2f}%"
                                    ]
                                })
                                st.table(scores_df)
                                
                                # 코멘트 표시 및 입력
                                st.write("### 💬 코멘트")
                                current_comment = row['comment'] if pd.notna(row['comment']) else ""
                                new_comment = st.text_area("코멘트 입력", value=current_comment, key=f"detail_comment_{row['id']}")
                                
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    if st.button("코멘트 저장"):
                                        save_comment(row['id'], new_comment)
                                        st.success("코멘트가 저장되었습니다!")
                                        st.experimental_rerun()
                                
                                with col2:
                                    if st.button("상세보기 닫기"):
                                        del st.session_state['selected_result_id']
                                        st.experimental_rerun()
                                    
                            except Exception as e:
                                st.error(f"이미지를 불러올 수 없습니다: {str(e)}")
                else:
                    st.info(f"선택한 최소 유사도({min_similarity}%) 이상의 결과가 없습니다.")
        else:
            st.info("저장된 비교 결과가 없습니다.")
    except Exception as e:
        st.error(f"결과를 불러오는 중 오류가 발생했습니다: {str(e)}")