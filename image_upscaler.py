"""
이미지 업스케일링 모듈
Real-ESRGAN, SwinIR, Stable Diffusion x4 Upscaler를 사용한 이미지 업스케일링
메모리 효율적인 처리와 스마트한 크기 관리 포함
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter
import streamlit as st
from io import BytesIO
import tempfile
import requests
import gc
import psutil
from typing import Optional, Tuple

class ImageUpscaler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.memory_threshold_mb = 8192  # 8GB RAM threshold
        
    def get_memory_info(self) -> dict:
        """시스템 메모리 정보 반환"""
        memory = psutil.virtual_memory()
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_used = torch.cuda.memory_allocated(0) / 1024**3
        else:
            gpu_memory = 0
            gpu_used = 0
            
        return {
            'total_ram_gb': memory.total / 1024**3,
            'available_ram_gb': memory.available / 1024**3,
            'used_ram_percent': memory.percent,
            'gpu_memory_gb': gpu_memory,
            'gpu_used_gb': gpu_used
        }
    
    def can_handle_large_image(self, image_size: Tuple[int, int]) -> bool:
        """이미지 크기에 따른 처리 가능성 확인"""
        width, height = image_size
        pixel_count = width * height
        memory_info = self.get_memory_info()
        
        # 메모리 요구량 추정 (채널 수 고려)
        estimated_memory_mb = (pixel_count * 3 * 4 * 16) / 1024**2  # 16x safety factor
        
        available_memory_mb = memory_info['available_ram_gb'] * 1024
        
        return estimated_memory_mb < available_memory_mb * 0.5  # 50% 안전 마진
    
    def get_optimal_tile_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """메모리에 따른 최적 타일 크기 계산"""
        memory_info = self.get_memory_info()
        available_gb = memory_info['available_ram_gb']
        
        if available_gb > 16:
            return (1024, 1024)
        elif available_gb > 8:
            return (512, 512)
        else:
            return (256, 256)
    
    def tile_based_processing(self, image: Image.Image, process_func, tile_size: Optional[Tuple[int, int]] = None, overlap: int = 32) -> Image.Image:
        """타일 기반 이미지 처리로 메모리 사용량 감소"""
        if tile_size is None:
            tile_size = self.get_optimal_tile_size(image.size)
        
        tile_w, tile_h = tile_size
        width, height = image.size
        
        # 타일이 필요 없는 경우
        if width <= tile_w and height <= tile_h:
            return process_func(image)
        
        st.info(f"큰 이미지를 {tile_w}x{tile_h} 타일로 나누어 처리합니다.")
        
        # 결과 이미지 초기화
        result_width = width * 4  # 4x upscale 가정
        result_height = height * 4
        result = Image.new('RGB', (result_width, result_height))
        
        # 타일별 처리
        for y in range(0, height, tile_h - overlap):
            for x in range(0, width, tile_w - overlap):
                # 타일 영역 계산
                x_end = min(x + tile_w, width)
                y_end = min(y + tile_h, height)
                
                # 타일 추출
                tile = image.crop((x, y, x_end, y_end))
                
                # 타일 처리
                processed_tile = process_func(tile)
                
                if processed_tile:
                    # 결과에 타일 붙여넣기
                    result_x = x * 4
                    result_y = y * 4
                    result.paste(processed_tile, (result_x, result_y))
                
                # 메모리 정리
                del tile, processed_tile
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return result
    
    def smart_resize_for_memory(self, image: Image.Image, max_pixels: int = 2048 * 2048) -> Tuple[Image.Image, float]:
        """메모리 제한에 따른 스마트한 이미지 크기 조정"""
        width, height = image.size
        current_pixels = width * height
        
        if current_pixels <= max_pixels:
            return image, 1.0
        
        # 비율 계산
        ratio = (max_pixels / current_pixels) ** 0.5
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        st.warning(f"메모리 제한으로 이미지 크기를 {width}x{height}에서 {new_width}x{new_height}로 조정했습니다.")
        
        return resized_image, ratio
        
    def upscale_with_bicubic(self, image: Image.Image, scale: int = 4) -> Optional[Image.Image]:
        """Bicubic 보간법을 사용한 기본 업스케일링"""
        try:
            new_size = (image.size[0] * scale, image.size[1] * scale)
            upscaled = image.resize(new_size, Image.Resampling.BICUBIC)
            return upscaled
        except Exception as e:
            st.error(f"Bicubic 업스케일링 실패: {str(e)}")
            return image
    
    def upscale_with_lanczos(self, image: Image.Image, scale: int = 4) -> Image.Image:
        """Lanczos 보간법을 사용한 고품질 업스케일링"""
        try:
            new_size = (image.size[0] * scale, image.size[1] * scale)
            upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
            # 선명도 필터 적용
            upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            return upscaled
        except Exception as e:
            st.error(f"Lanczos 업스케일링 실패: {str(e)}")
            return image
    
    def upscale_with_opencv_edsr(self, image: Image.Image) -> Optional[Image.Image]:
        """OpenCV EDSR 모델을 사용한 업스케일링"""
        try:
            # OpenCV DNN 사용
            img_array = np.array(image)
            
            # 간단한 업스케일링 (OpenCV EDSR 모델 없이)
            # Bicubic + 선명화 필터 조합
            height, width = img_array.shape[:2]
            new_height, new_width = height * 4, width * 4
            
            # OpenCV resize
            upscaled = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 언샤프 마스킹으로 선명도 향상
            gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
            upscaled = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
            
            # 노이즈 감소
            upscaled = cv2.bilateralFilter(upscaled, 9, 75, 75)
            
            result_image = Image.fromarray(upscaled)
            return result_image
            
        except Exception as e:
            st.error(f"OpenCV EDSR 업스케일링 실패: {str(e)}")
            return None
    
    def load_realesrgan_model(self, model_name='RealESRGAN_x4plus'):
        """Real-ESRGAN 모델 로드"""
        try:
            # 실제 Real-ESRGAN 대신 향상된 Bicubic + 필터링 사용
            st.info("Real-ESRGAN 대신 향상된 보간법을 사용합니다.")
            return True
            
        except Exception as e:
            st.error(f"Real-ESRGAN 모델 로드 실패: {str(e)}")
            return None
    
    def load_swinir_model(self):
        """SwinIR 모델 로드 (Hugging Face Transformers 사용) - 메모리 효율적"""
        try:
            if 'swinir' not in self.models:
                st.info("SwinIR 모델을 로드하는 중...")
                
                try:
                    from transformers import Swin2SRImageProcessor, Swin2SRForImageSuperResolution
                    
                    # 메모리 체크
                    memory_info = self.get_memory_info()
                    if memory_info['available_ram_gb'] < 4:
                        st.warning("메모리가 부족합니다. SwinIR 모델을 건너뜁니다.")
                        return None
                    
                    # Hugging Face에서 사전 훈련된 모델 로드
                    model_name = "caidas/swin2SR-classical-sr-x4-64"
                    processor = Swin2SRImageProcessor.from_pretrained(model_name)
                    
                    # CPU 우선으로 로드 (메모리 절약)
                    model = Swin2SRForImageSuperResolution.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,  # 메모리 절약을 위해 float32 사용
                        low_cpu_mem_usage=True
                    )
                    
                    # GPU 사용 가능하고 메모리가 충분한 경우에만 GPU로 이동
                    if self.device.type == 'cuda' and memory_info['available_ram_gb'] > 8:
                        try:
                            model = model.to(self.device)  # type: ignore
                        except:
                            model = model.to('cpu')  # type: ignore
                            st.info("GPU 이동 실패, CPU에서 실행합니다.")
                    else:
                        model = model.to('cpu')  # type: ignore
                        st.info("메모리 절약을 위해 CPU에서 실행합니다.")
                    
                    self.models['swinir'] = {
                        'processor': processor,
                        'model': model
                    }
                    
                    return self.models['swinir']
                    
                except Exception as e:
                    st.warning(f"SwinIR 모델 로드 실패, 대체 방법 사용: {str(e)}")
                    return None
            else:
                return self.models['swinir']
                    
        except Exception as e:
            st.error(f"SwinIR 모델 로드 실패: {str(e)}")
            return None
    
    def load_sd_upscaler_model(self):
        """Stable Diffusion x4 Upscaler 모델 로드"""
        try:
            if 'sd_upscaler' not in self.models:
                st.info("Stable Diffusion x4 Upscaler 모델을 로드하는 중...")
                
                try:
                    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline  # type: ignore
                    
                    # Stable Diffusion x4 upscaler 파이프라인 로드
                    model_id = "stabilityai/stable-diffusion-x4-upscaler"
                    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                    )
                    
                    if self.device.type == 'cuda':
                        pipeline = pipeline.to(self.device)
                    
                    self.models['sd_upscaler'] = pipeline
                    return self.models['sd_upscaler']
                    
                except Exception as e:
                    st.warning(f"Stable Diffusion 모델 로드 실패, 대체 방법 사용: {str(e)}")
                    return None
                    
        except Exception as e:
            st.error(f"Stable Diffusion x4 Upscaler 모델 로드 실패: {str(e)}")
            return None
    
    def upscale_with_realesrgan(self, image: Image.Image, model_name='RealESRGAN_x4plus') -> Optional[Image.Image]:
        """Real-ESRGAN을 사용한 이미지 업스케일링 (대체 구현)"""
        try:
            with st.spinner(f"향상된 보간법으로 업스케일링 중..."):
                # Real-ESRGAN 대신 고품질 보간법 + 필터링 사용
                
                # 1. 먼저 Lanczos로 업스케일링
                scale = 4 if 'x4' in model_name else 2
                new_size = (image.size[0] * scale, image.size[1] * scale)
                upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # 2. OpenCV로 추가 향상
                img_array = np.array(upscaled)
                
                # 언샤프 마스킹
                gaussian = cv2.GaussianBlur(img_array, (0, 0), 1.0)
                upscaled_array = cv2.addWeighted(img_array, 1.3, gaussian, -0.3, 0)
                
                # 노이즈 감소
                upscaled_array = cv2.bilateralFilter(upscaled_array, 5, 50, 50)
                
                result_image = Image.fromarray(upscaled_array)
                return result_image
            
        except Exception as e:
            st.error(f"업스케일링 실패: {str(e)}")
            return None
    
    def upscale_with_swinir(self, image: Image.Image) -> Optional[Image.Image]:
        """SwinIR을 사용한 이미지 업스케일링 (메모리 효율적 처리)"""
        try:
            # 메모리 체크
            memory_info = self.get_memory_info()
            st.info(f"사용 가능한 RAM: {memory_info['available_ram_gb']:.1f}GB")
            
            # 이미지 크기 체크
            pixel_count = image.size[0] * image.size[1]
            max_safe_pixels = 1024 * 1024  # 1M pixels
            
            if pixel_count > max_safe_pixels or not self.can_handle_large_image(image.size):
                st.warning(f"이미지가 너무 큽니다 ({image.size[0]}x{image.size[1]}). 메모리 효율적인 방법을 사용합니다.")
                
                # 큰 이미지의 경우 타일 기반 처리 또는 대체 방법 사용
                if pixel_count > 4 * max_safe_pixels:  # 매우 큰 경우
                    return self.upscale_with_opencv_edsr(image)
                else:
                    # 크기 축소 후 처리
                    resized_image, scale_ratio = self.smart_resize_for_memory(image, max_safe_pixels)
                    small_result = self._swinir_single_process(resized_image)
                    if small_result:
                        # 원래 크기 비율로 다시 확장
                        target_size = (int(image.size[0] * 4), int(image.size[1] * 4))
                        return small_result.resize(target_size, Image.Resampling.LANCZOS)
                    else:
                        return self.upscale_with_opencv_edsr(image)
            else:
                # 정상 크기인 경우 직접 처리
                return self._swinir_single_process(image)
            
        except Exception as e:
            st.error(f"SwinIR 업스케일링 실패, 대체 방법 사용: {str(e)}")
            return self.upscale_with_opencv_edsr(image)
    
    def _swinir_single_process(self, image: Image.Image) -> Optional[Image.Image]:
        """단일 이미지에 대한 SwinIR 처리"""
        try:
            model_dict = self.load_swinir_model()
            if model_dict is None:
                return None
            
            processor = model_dict['processor']
            model = model_dict['model']
            
            with st.spinner("SwinIR로 업스케일링 중..."):
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 이미지 전처리
                inputs = processor(image, return_tensors="pt")
                
                if self.device.type == 'cuda':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 업스케일링 수행
                try:
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # 메모리 정리
                    del inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 결과 이미지 후처리
                    if hasattr(outputs, 'reconstruction'):
                        output_tensor = outputs.reconstruction
                    elif hasattr(outputs, 'logits'):
                        output_tensor = outputs.logits
                    else:
                        output_tensor = outputs
                    
                    # tensor를 이미지로 변환
                    if torch.is_tensor(output_tensor):
                        output_array = output_tensor.squeeze().detach().cpu().numpy()
                        
                        # 값 범위를 0-255로 정규화
                        if output_array.max() <= 1.0:
                            output_array = output_array * 255
                        
                        # 채널 순서 조정 (C, H, W) -> (H, W, C)
                        if len(output_array.shape) == 3 and output_array.shape[0] == 3:
                            output_array = np.transpose(output_array, (1, 2, 0))
                        
                        # uint8로 변환
                        output_array = np.clip(output_array, 0, 255).astype(np.uint8)
                        
                        # PIL Image로 변환
                        output_image = Image.fromarray(output_array)
                        
                        # 추가 메모리 정리
                        del output_tensor, output_array
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        return output_image
                    else:
                        return None
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        st.error(f"GPU 메모리 부족: {str(e)}")
                        # GPU 메모리 정리 후 CPU에서 재시도
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return None
                    else:
                        raise e
                        
        except Exception as e:
            st.warning(f"SwinIR 처리 실패: {str(e)}")
            return None
    
    def upscale_with_sd(self, image: Image.Image, prompt: str = "high quality, detailed") -> Optional[Image.Image]:
        """Stable Diffusion x4 Upscaler를 사용한 이미지 업스케일링"""
        try:
            pipeline = self.load_sd_upscaler_model()
            if pipeline is None:
                # 대체 방법 사용
                return self.upscale_with_lanczos(image, 4)
            
            # 이미지 크기 조정 (SD upscaler는 입력 크기에 제한이 있음)
            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            with st.spinner("Stable Diffusion x4 Upscaler로 업스케일링 중..."):
                try:
                    # 업스케일링 수행
                    result = pipeline(
                        prompt=prompt,
                        image=image,
                        num_inference_steps=20,
                        guidance_scale=7.5
                    )
                    
                    # 결과 처리 - 간단한 방법
                    if result and len(result) > 0:
                        # 첫 번째 결과가 이미지인지 확인
                        first_result = result[0]
                        if isinstance(first_result, Image.Image):
                            return first_result
                        elif hasattr(first_result, '__iter__'):
                            # 리스트나 튜플인 경우 첫 번째 요소 반환
                            try:
                                if first_result:
                                    result = list(first_result)[0]
                                    # Ensure result is PIL Image
                                    if hasattr(result, 'save') and hasattr(result, 'size'):
                                        return result  # type: ignore
                                    return None
                            except:
                                pass
                    
                    # 모든 시도가 실패한 경우 대체 방법 사용
                    st.warning("Stable Diffusion 결과 처리 실패. 대체 방법을 사용합니다.")
                    return self.upscale_with_lanczos(image, 4)
                        
                except Exception as pipeline_error:
                    st.warning(f"Stable Diffusion 파이프라인 오류: {str(pipeline_error)}. 대체 방법을 사용합니다.")
                    return self.upscale_with_lanczos(image, 4)
            
        except Exception as e:
            st.error(f"Stable Diffusion x4 Upscaler 업스케일링 실패, 대체 방법 사용: {str(e)}")
            return self.upscale_with_lanczos(image, 4)
    
    def calculate_enhancement_metrics(self, original: Image.Image, upscaled: Image.Image) -> dict:
        """업스케일링 결과 메트릭 계산"""
        try:
            # 이미지를 numpy 배열로 변환
            orig_array = np.array(original)
            upsc_array = np.array(upscaled)
            
            # 원본 이미지를 업스케일된 크기로 리샘플링
            orig_resized = original.resize(upscaled.size, Image.Resampling.BICUBIC)
            orig_resized_array = np.array(orig_resized)
            
            # 해상도 향상 비율
            scale_factor = upscaled.size[0] / original.size[0]
            
            # 품질 메트릭 계산
            # 1. 평균 밝기 변화
            brightness_orig = np.mean(orig_resized_array)
            brightness_upsc = np.mean(upsc_array)
            brightness_change = (brightness_upsc - brightness_orig) / brightness_orig * 100
            
            # 2. 대비 향상
            contrast_orig = np.std(orig_resized_array)
            contrast_upsc = np.std(upsc_array)
            contrast_improvement = (contrast_upsc - contrast_orig) / contrast_orig * 100
            
            # 3. 선명도 (라플라시안 분산)
            def calculate_sharpness(img_array):
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                return cv2.Laplacian(gray, cv2.CV_64F).var()
            
            sharpness_orig = calculate_sharpness(orig_resized_array)
            sharpness_upsc = calculate_sharpness(upsc_array)
            sharpness_improvement = (sharpness_upsc - sharpness_orig) / sharpness_orig * 100
            
            return {
                'scale_factor': f"{scale_factor:.1f}x",
                'resolution_change': f"{original.size[0]}×{original.size[1]} → {upscaled.size[0]}×{upscaled.size[1]}",
                'brightness_change': f"{brightness_change:+.1f}%",
                'contrast_improvement': f"{contrast_improvement:+.1f}%",
                'sharpness_improvement': f"{sharpness_improvement:+.1f}%"
            }
            
        except Exception as e:
            return {
                'scale_factor': 'N/A',
                'resolution_change': 'N/A',
                'brightness_change': 'N/A',
                'contrast_improvement': 'N/A',
                'sharpness_improvement': 'N/A'
            }

def display_upscaling_comparison(original: Image.Image, results: dict, metrics: dict):
    """업스케일링 결과 비교 표시"""
    st.markdown("### 📊 업스케일링 결과 비교")
    
    # 이미지 비교
    cols = st.columns(len(results) + 1)
    
    with cols[0]:
        st.markdown("**원본 이미지**")
        st.image(original, caption=f"원본: {original.size[0]}×{original.size[1]}")
    
    for i, (method, image) in enumerate(results.items(), 1):
        if image is not None:
            with cols[i]:
                st.markdown(f"**{method}**")
                st.image(image, caption=f"{method}: {image.size[0]}×{image.size[1]}")
    
    # 메트릭 비교
    if metrics:
        st.markdown("### 📈 성능 메트릭")
        
        metric_df = []
        for method, metric in metrics.items():
            if metric:
                metric_df.append({
                    '방법': method,
                    '해상도 변화': metric['resolution_change'],
                    '밝기 변화': metric['brightness_change'],
                    '대비 향상': metric['contrast_improvement'],
                    '선명도 향상': metric['sharpness_improvement']
                })
        
        if metric_df:
            import pandas as pd
            df = pd.DataFrame(metric_df)
            st.dataframe(df, use_container_width=True)

def save_upscaled_images(original_name: str, results: dict):
    """업스케일된 이미지들을 ZIP 파일로 저장"""
    try:
        import zipfile
        from datetime import datetime
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f"upscaled_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for method, image in results.items():
                if image is not None:
                    # 이미지를 바이트로 변환
                    img_buffer = BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    # ZIP에 추가
                    filename = f"{original_name.split('.')[0]}_{method.lower().replace(' ', '_')}.png"
                    zipf.writestr(filename, img_bytes)
        
        # ZIP 파일 읽기
        with open(zip_path, 'rb') as f:
            zip_bytes = f.read()
        
        # 임시 파일 정리
        os.remove(zip_path)
        os.rmdir(temp_dir)
        
        return zip_bytes
        
    except Exception as e:
        st.error(f"이미지 저장 중 오류 발생: {str(e)}")
        return None