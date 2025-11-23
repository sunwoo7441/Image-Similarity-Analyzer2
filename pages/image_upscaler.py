"""
이미지 업스케일링 페이지
Real-ESRGAN, SwinIR, Stable Diffusion x4 Upscaler를 사용한 고품질 이미지 업스케일링
"""

import streamlit as st
import numpy as np
from PIL import Image
import torch
import time
import io
import json
import os
import sys

# 현재 디렉토리를 상위 폴더로 변경
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_upscaler import ImageUpscaler, display_upscaling_comparison, save_upscaled_images
from db_utils import save_work_history  # 작업 히스토리 저장 함수 추가

def main():
    import time  # time 모듈 명시적 import
    
    st.title("🚀 AI 이미지 업스케일러")
    st.markdown("**Real-ESRGAN, SwinIR, Stable Diffusion x4 Upscaler를 사용한 고품질 이미지 업스케일링**")
    
    # GPU 사용 가능 여부 확인
    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"🔧 사용 중인 디바이스: {device_info}")
    
    if not torch.cuda.is_available():
        st.warning("⚠️ GPU가 감지되지 않았습니다. CPU로 실행하면 처리 시간이 오래 걸릴 수 있습니다.")
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 단일 모델 업스케일링", 
        "⚖️ 모델 비교 분석", 
        "🎛️ 고급 설정", 
        "📊 성능 벤치마크"
    ])
    
    # 업스케일러 초기화
    if 'upscaler' not in st.session_state:
        st.session_state.upscaler = ImageUpscaler()
    
    upscaler = st.session_state.upscaler
    
    with tab1:
        st.markdown("### 🖼️ 단일 모델로 이미지 업스케일링")
        
        # 이미지 업로드
        uploaded_file = st.file_uploader(
            "업스케일할 이미지를 선택하세요",
            type=["jpg", "png", "jpeg", "bmp", "tiff"],
            key="single_upscale"
        )
        
        if uploaded_file:
            # 원본 이미지 표시
            original_image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 원본 이미지")
                st.image(original_image, caption=f"크기: {original_image.size[0]}×{original_image.size[1]}")
                
                # 이미지 정보
                file_size = len(uploaded_file.getvalue())
                st.write(f"**파일명:** {uploaded_file.name}")
                st.write(f"**크기:** {original_image.size[0]} × {original_image.size[1]}")
                st.write(f"**파일 크기:** {file_size/1024/1024:.1f} MB")
                
                # 모델 선택
                st.markdown("#### 업스케일링 모델 선택")
                model_choice = st.selectbox(
                    "사용할 모델을 선택하세요:",
                    [
                        "Real-ESRGAN x4plus",
                        "Real-ESRGAN x2plus", 
                        "SwinIR",
                        "Stable Diffusion x4 Upscaler"
                    ]
                )
                
                # Stable Diffusion용 프롬프트
                prompt = ""
                if "Stable Diffusion" in model_choice:
                    prompt = st.text_input(
                        "업스케일링 프롬프트:",
                        value="high quality, detailed, sharp, clear",
                        help="업스케일링에 사용할 텍스트 프롬프트를 입력하세요"
                    )
                
                # 업스케일링 실행
                if st.button("🚀 업스케일링 시작", key="single_start"):
                    start_time = time.time()
                    
                    result_image = None
                    
                    if model_choice == "Real-ESRGAN x4plus":
                        result_image = upscaler.upscale_with_realesrgan(original_image, 'RealESRGAN_x4plus')
                    elif model_choice == "Real-ESRGAN x2plus":
                        result_image = upscaler.upscale_with_realesrgan(original_image, 'RealESRGAN_x2plus')
                    elif model_choice == "SwinIR":
                        result_image = upscaler.upscale_with_swinir(original_image)
                    elif model_choice == "Stable Diffusion x4 Upscaler":
                        result_image = upscaler.upscale_with_sd(original_image, prompt)
                    
                    if result_image is None:
                        result_image = upscaler.upscale_with_lanczos(original_image, 4)
                    
                    processing_time = time.time() - start_time
                    
                    if result_image:
                        # 이미지를 임시 저장
                        temp_dir = "Result/image_upscaling"
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        original_path = os.path.join(temp_dir, f"original_{uploaded_file.name}")
                        result_path = os.path.join(temp_dir, f"upscaled_{model_choice.lower().replace(' ', '_')}_{uploaded_file.name}")
                        
                        original_image.save(original_path)
                        result_image.save(result_path)
                        
                        # 성능 메트릭 계산
                        scale_factor = result_image.size[0] / original_image.size[0]
                        pixel_increase = (result_image.size[0] * result_image.size[1]) / (original_image.size[0] * original_image.size[1])
                        
                        # 작업 히스토리 저장
                        parameters = {
                            "model": model_choice,
                            "prompt": prompt if "Stable Diffusion" in model_choice else None,
                            "original_size": f"{original_image.size[0]}×{original_image.size[1]}"
                        }
                        
                        results = {
                            "scale_factor": f"{scale_factor:.1f}x",
                            "new_size": f"{result_image.size[0]}×{result_image.size[1]}",
                            "pixel_increase": f"{pixel_increase:.1f}x",
                            "processing_time": f"{processing_time:.1f}초"
                        }
                        
                        work_id = save_work_history(
                            work_type="image_upscaling",
                            title=f"이미지 업스케일링 - {model_choice}",
                            description=f"{uploaded_file.name} 파일을 {model_choice} 모델로 업스케일링",
                            input_images=json.dumps([original_path]),
                            output_images=json.dumps([result_path]),
                            parameters=json.dumps(parameters),
                            results=json.dumps(results)
                        )
                        
                        # 결과 저장
                        st.session_state.single_result = {
                            'original': original_image,
                            'result': result_image,
                            'model': model_choice,
                            'time': processing_time,
                            'filename': uploaded_file.name,
                            'work_id': work_id
                        }
            
            with col2:
                if 'single_result' in st.session_state:
                    result_data = st.session_state.single_result
                    
                    st.markdown("#### 업스케일링 결과")
                    st.image(
                        result_data['result'], 
                        caption=f"{result_data['model']}: {result_data['result'].size[0]}×{result_data['result'].size[1]}"
                    )
                    
                    # 성능 정보
                    st.markdown("#### 📈 성능 정보")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        scale_factor = result_data['result'].size[0] / result_data['original'].size[0]
                        st.metric("해상도 향상", f"{scale_factor:.1f}x")
                        st.metric("처리 시간", f"{result_data['time']:.1f}초")
                    
                    with col_b:
                        pixel_increase = (result_data['result'].size[0] * result_data['result'].size[1]) / (result_data['original'].size[0] * result_data['original'].size[1])
                        st.metric("픽셀 증가", f"{pixel_increase:.1f}x")
                        
                        # 파일 크기 추정
                        import io
                        buffer = io.BytesIO()
                        result_data['result'].save(buffer, format='PNG')
                        result_size = len(buffer.getvalue()) / 1024 / 1024
                        st.metric("예상 파일 크기", f"{result_size:.1f} MB")
                    
                    # 다운로드 버튼
                    st.markdown("#### 💾 결과 다운로드")
                    
                    # 작업 저장 완료 메시지
                    if 'work_id' in result_data:
                        st.success(f"✅ 작업이 히스토리에 저장되었습니다! (ID: {result_data['work_id'][:8]})")
                    
                    # 단일 이미지 다운로드
                    buffer = io.BytesIO()
                    result_data['result'].save(buffer, format='PNG')
                    
                    st.download_button(
                        label="📥 업스케일된 이미지 다운로드 (PNG)",
                        data=buffer.getvalue(),
                        file_name=f"{result_data['filename'].split('.')[0]}_{result_data['model'].lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )
    
    with tab2:
        st.markdown("### ⚖️ 모든 모델로 비교 업스케일링")
        st.markdown("*동일한 이미지를 여러 모델로 업스케일링하여 결과를 비교합니다.*")
        
        # 이미지 업로드
        uploaded_file_compare = st.file_uploader(
            "비교할 이미지를 선택하세요",
            type=["jpg", "png", "jpeg", "bmp", "tiff"],
            key="compare_upscale"
        )
        
        if uploaded_file_compare:
            original_image = Image.open(uploaded_file_compare).convert("RGB")
            
            # 원본 이미지 표시
            st.markdown("#### 원본 이미지")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(original_image, caption=f"원본: {original_image.size[0]}×{original_image.size[1]}")
            
            # 설정 옵션
            st.markdown("#### 🎛️ 업스케일링 설정")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_models = st.multiselect(
                    "비교할 모델들을 선택하세요:",
                    [
                        "Real-ESRGAN x4plus",
                        "Real-ESRGAN x2plus",
                        "SwinIR", 
                        "Stable Diffusion x4 Upscaler"
                    ],
                    default=["Real-ESRGAN x4plus", "SwinIR"]
                )
            
            with col2:
                sd_prompt = st.text_input(
                    "Stable Diffusion 프롬프트:",
                    value="high quality, detailed, sharp, clear",
                    help="Stable Diffusion 모델용 프롬프트"
                )
            
            # 일괄 업스케일링 실행
            if st.button("🚀 모든 모델로 업스케일링 시작", key="compare_start") and selected_models:
                import time  # time 모듈 명시적 import
                
                results = {}
                metrics = {}
                total_start_time = time.time()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model in enumerate(selected_models):
                    status_text.text(f"처리 중: {model}")
                    
                    start_time = time.time()
                    result_image = None
                    
                    try:
                        if model == "Real-ESRGAN x4plus":
                            result_image = upscaler.upscale_with_realesrgan(original_image, 'RealESRGAN_x4plus')
                        elif model == "Real-ESRGAN x2plus":
                            result_image = upscaler.upscale_with_realesrgan(original_image, 'RealESRGAN_x2plus')
                        elif model == "SwinIR":
                            result_image = upscaler.upscale_with_swinir(original_image)
                        elif model == "Stable Diffusion x4 Upscaler":
                            result_image = upscaler.upscale_with_sd(original_image, sd_prompt)
                        
                        processing_time = time.time() - start_time
                        
                        if result_image:
                            results[model] = result_image
                            # 메트릭 계산
                            metrics[model] = upscaler.calculate_enhancement_metrics(original_image, result_image)
                            metrics[model]['processing_time'] = f"{processing_time:.1f}초"
                        
                    except Exception as e:
                        st.error(f"{model} 처리 실패: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                total_time = time.time() - total_start_time
                status_text.text(f"모든 처리 완료! (총 {total_time:.1f}초)")
                
                # 결과 저장
                st.session_state.compare_results = {
                    'original': original_image,
                    'results': results,
                    'metrics': metrics,
                    'filename': uploaded_file_compare.name
                }
                
                # 결과 표시
                if results:
                    display_upscaling_comparison(original_image, results, metrics)
                    
                    # 일괄 다운로드
                    st.markdown("#### 💾 일괄 다운로드")
                    
                    zip_data = save_upscaled_images(uploaded_file_compare.name, results)
                    if zip_data:
                        st.download_button(
                            label="📦 모든 결과 이미지 다운로드 (ZIP)",
                            data=zip_data,
                            file_name=f"upscaled_results_{uploaded_file_compare.name.split('.')[0]}.zip",
                            mime="application/zip"
                        )
    
    with tab3:
        st.markdown("### 🎛️ 고급 설정 및 최적화")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 시스템 정보")
            
            # GPU 메모리 정보
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
                st.write(f"**GPU 메모리:** {gpu_memory:.1f} GB")
                
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    cached = torch.cuda.memory_reserved(0) / 1024**3
                    st.write(f"**사용 중인 메모리:** {allocated:.2f} GB")
                    st.write(f"**캐시된 메모리:** {cached:.2f} GB")
            else:
                import psutil
                cpu_count = psutil.cpu_count()
                memory = psutil.virtual_memory().total / 1024**3
                st.write(f"**CPU 코어:** {cpu_count}")
                st.write(f"**시스템 메모리:** {memory:.1f} GB")
            
            # 메모리 정리 버튼
            if st.button("🧹 메모리 정리"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    st.success("GPU 메모리 캐시를 정리했습니다.")
                else:
                    st.info("CPU 모드에서는 메모리 정리가 필요하지 않습니다.")
        
        with col2:
            st.markdown("#### ⚙️ 모델 관리")
            
            # 로드된 모델 확인
            if hasattr(st.session_state, 'upscaler') and st.session_state.upscaler.models:
                st.write("**로드된 모델:**")
                for model_name in st.session_state.upscaler.models.keys():
                    st.write(f"✅ {model_name}")
            else:
                st.write("로드된 모델이 없습니다.")
            
            # 모델 사전 로드
            st.markdown("**모델 사전 로드:**")
            preload_models = st.multiselect(
                "사전에 로드할 모델을 선택하세요:",
                [
                    "Real-ESRGAN x4plus",
                    "Real-ESRGAN x2plus",
                    "SwinIR",
                    "Stable Diffusion x4 Upscaler"
                ]
            )
            
            if st.button("📥 선택된 모델 사전 로드"):
                for model in preload_models:
                    try:
                        if model == "Real-ESRGAN x4plus":
                            upscaler.load_realesrgan_model('RealESRGAN_x4plus')
                        elif model == "Real-ESRGAN x2plus":
                            upscaler.load_realesrgan_model('RealESRGAN_x2plus')
                        elif model == "SwinIR":
                            upscaler.load_swinir_model()
                        elif model == "Stable Diffusion x4 Upscaler":
                            upscaler.load_sd_upscaler_model()
                        
                        st.success(f"{model} 로드 완료!")
                    except Exception as e:
                        st.error(f"{model} 로드 실패: {str(e)}")
    
    with tab4:
        st.markdown("### 📊 성능 벤치마크 및 분석")
        
        if 'compare_results' in st.session_state:
            results_data = st.session_state.compare_results
            
            st.markdown("#### 🏆 모델 성능 순위")
            
            if results_data['metrics']:
                import pandas as pd
                
                # 성능 데이터 정리
                benchmark_data = []
                for model, metrics in results_data['metrics'].items():
                    if metrics:
                        benchmark_data.append({
                            '모델': model,
                            '처리 시간': metrics.get('processing_time', 'N/A'),
                            '해상도 변화': metrics.get('resolution_change', 'N/A'),
                            '선명도 향상': metrics.get('sharpness_improvement', 'N/A'),
                            '대비 향상': metrics.get('contrast_improvement', 'N/A'),
                            '밝기 변화': metrics.get('brightness_change', 'N/A')
                        })
                
                if benchmark_data:
                    df = pd.DataFrame(benchmark_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # 차트 생성
                    st.markdown("#### 📈 성능 비교 차트")
                    
                    # 처리 시간 비교
                    try:
                        import matplotlib.pyplot as plt
                        
                        models = [data['모델'] for data in benchmark_data]
                        times = []
                        
                        for data in benchmark_data:
                            time_str = data['처리 시간'].replace('초', '')
                            try:
                                times.append(float(time_str))
                            except:
                                times.append(0)
                        
                        if times and any(t > 0 for t in times):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(models, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
                            ax.set_ylabel('처리 시간 (초)')
                            ax.set_title('모델별 처리 시간 비교')
                            ax.tick_params(axis='x', rotation=45)
                            
                            # 값 표시
                            for bar, time in zip(bars, times):
                                if time > 0:
                                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                           f'{time:.1f}s', ha='center', va='bottom')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    except Exception as e:
                        st.error(f"차트 생성 실패: {str(e)}")
            
            # 권장사항
            st.markdown("#### 💡 모델 선택 가이드")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Real-ESRGAN**
                - ✅ 사진에 최적화
                - ✅ 빠른 처리 속도
                - ✅ 안정적인 결과
                - ❌ 창작적 개선 제한적
                """)
                
                st.markdown("""
                **SwinIR**
                - ✅ 균형잡힌 성능
                - ✅ 다양한 이미지 타입
                - ✅ 고품질 결과
                - ❌ 상대적으로 느림
                """)
            
            with col2:
                st.markdown("""
                **Stable Diffusion x4 Upscaler**
                - ✅ 창작적 개선
                - ✅ 텍스트 프롬프트 제어
                - ✅ 예술적 효과
                - ❌ 느린 처리 속도
                - ❌ 입력 크기 제한
                """)
        else:
            st.info("벤치마크 데이터를 보려면 먼저 '모델 비교 분석' 탭에서 업스케일링을 실행하세요.")

if __name__ == "__main__":
    main()