import streamlit as st
from PIL import Image
import sqlite3
import os
import sys

# 현재 디렉토리를 상위 폴더로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db_utils import get_results

def app():
    st.title("유사도 비교 결과 조회")
    
    # 데이터베이스에서 결과 가져오기
    try:
        results = get_results()
        if len(results) > 0:
            st.write(f"총 {len(results)}개의 비교 결과가 있습니다.")
            
            # 결과 필터링
            min_similarity = st.slider("최소 유사도 점수", 0.0, 100.0, 0.0, 5.0)
            filtered_results = results[results['avg_score'] >= min_similarity]
            
            # 결과 표시
            if len(filtered_results) > 0:
                for index, row in filtered_results.iterrows():
                    with st.expander(f"결과 #{index+1} - {row['timestamp']} (유사도: {row['avg_score']:.2f}%)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            try:
                                real_img = Image.open(row['real_image_path'])
                                st.image(real_img, caption="실제 사진", width=300)
                            except Exception as e:
                                st.error(f"실제 사진을 불러올 수 없습니다. 오류: {str(e)}")
                        
                        with col2:
                            try:
                                ai_img = Image.open(row['ai_image_path'])
                                st.image(ai_img, caption="AI 생성 사진", width=300)
                            except Exception as e:
                                st.error(f"AI 생성 사진을 불러올 수 없습니다. 오류: {str(e)}")
                        
                        # 점수 표시
                        st.markdown("### 유사도 점수")
                        scores_df = {
                            "점수 유형": ["SSIM", "PSNR", "VGG16", "평균"],
                            "점수 (%)": [
                                f"{row['ssim_score']:.2f}%", 
                                f"{row['psnr_score']:.2f}%", 
                                f"{row['vgg_score']:.2f}%", 
                                f"{row['avg_score']:.2f}%"
                            ]
                        }
                        st.table(scores_df)
                        
                        # 결과 삭제 옵션
                        if st.button(f"결과 #{index+1} 삭제", key=f"delete_{row['id']}"):
                            conn = sqlite3.connect('similarity_results.db')
                            c = conn.cursor()
                            c.execute("DELETE FROM results WHERE id = ?", (row['id'],))
                            conn.commit()
                            conn.close()
                            
                            # 이미지 파일 삭제 시도
                            try:
                                if os.path.exists(row['real_image_path']):
                                    os.remove(row['real_image_path'])
                                if os.path.exists(row['ai_image_path']):
                                    os.remove(row['ai_image_path'])
                            except Exception as e:
                                st.warning(f"이미지 파일을 삭제하는 중 오류가 발생했습니다: {str(e)}")
                                
                            st.success("결과가 삭제되었습니다. 페이지를 새로고침하세요.")
                            st.experimental_rerun()
            else:
                st.info(f"선택한 최소 유사도({min_similarity}%) 이상의 결과가 없습니다.")
        else:
            st.info("저장된 비교 결과가 없습니다.")
    except Exception as e:
        st.error(f"결과를 불러오는 중 오류가 발생했습니다: {str(e)}")