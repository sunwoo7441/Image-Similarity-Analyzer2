import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import io
import base64
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# PIL 이미지 크기 제한 증가 (보안을 위해 적절한 수준으로 설정)
Image.MAX_IMAGE_PIXELS = 300_000_000  # 300M 픽셀까지 허용

def safe_image_open(image_path_or_buffer, max_size=(4096, 4096)):
    """
    안전하게 이미지를 열고 크기가 너무 큰 경우 자동으로 리샘플링
    
    Args:
        image_path_or_buffer: 이미지 경로 또는 버퍼
        max_size: 최대 허용 크기 (width, height)
    
    Returns:
        PIL Image 객체
    """
    try:
        image = Image.open(image_path_or_buffer)
        
        # 이미지 크기 확인
        width, height = image.size
        max_width, max_height = max_size
        
        # 크기가 너무 큰 경우 비율을 유지하면서 축소
        if width > max_width or height > max_height:
            # 비율 계산
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            print(f"이미지 크기가 큽니다 ({width}x{height}). {new_width}x{new_height}로 리샘플링합니다.")
            
            # 고품질 리샘플링
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
        
    except Image.DecompressionBombError as e:
        print(f"이미지가 너무 커서 보안상 차단되었습니다: {e}")
        print("이미지 크기를 줄이거나 다른 이미지를 사용해주세요.")
        raise
    except Exception as e:
        print(f"이미지 로드 중 오류 발생: {e}")
        raise

# 배경 제거 함수 개선
def remove_background(image, threshold=240):
    """Remove background from image using rembg or fallback to threshold method"""
    if REMBG_AVAILABLE:
        try:
            # rembg를 사용한 AI 기반 배경 제거
            img_array = np.array(image)
            output = remove(img_array)
            return Image.fromarray(output)
        except Exception as e:
            print(f"rembg 사용 중 오류 발생: {e}, 임계값 방식으로 대체합니다.")
    
    # 임계값 기반 배경 제거 (fallback)
    img_array = np.array(image)
    
    # RGB 이미지를 처리
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # 밝은 배경 픽셀 마스크 생성
        mask = np.all(img_array > threshold, axis=2)
        
        # RGBA 이미지 생성 (알파 채널 포함)
        rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = img_array
        rgba[:, :, 3] = np.where(mask, 0, 255)  # 배경은 투명하게, 객체는 불투명하게
        
        return Image.fromarray(rgba)
    return image

# 영역 지정 배경 제거 함수
def remove_background_with_mask(image, mask_coords=None, mask_type="rectangle", threshold=240, invert_mask=False):
    """
    지정된 영역에만 배경 제거를 적용하는 함수
    
    Args:
        image: PIL 이미지
        mask_coords: 마스크 좌표 (사각형: [x1, y1, x2, y2], 폴리곤: [(x1,y1), (x2,y2), ...])
        mask_type: "rectangle" 또는 "polygon"
        threshold: 임계값 (rembg 미사용시)
        invert_mask: True면 마스크 영역 외부에 배경제거 적용, False면 마스크 영역 내부에 적용
    
    Returns:
        배경이 제거된 PIL 이미지 (RGBA)
    """
    
    if mask_coords is None:
        # 마스크가 없으면 전체 이미지에 배경 제거 적용
        return remove_background(image, threshold)
    
    # 원본 이미지를 RGBA로 변환
    if image.mode != 'RGBA':
        original_rgba = image.convert('RGBA')
    else:
        original_rgba = image.copy()
    
    # 마스크 생성
    mask = create_mask(image.size, mask_coords, mask_type)
    
    if invert_mask:
        # 마스크 반전 (지정 영역 외부에 배경제거 적용)
        mask = Image.eval(mask, lambda x: 255 - x)
    
    # 마스크된 영역만 잘라내어 배경 제거 적용
    masked_image = apply_mask_to_image(image, mask)
    
    # 배경 제거 적용
    if REMBG_AVAILABLE:
        try:
            # rembg를 사용한 AI 기반 배경 제거
            masked_array = np.array(masked_image)
            bg_removed_result = remove(masked_array)
            if isinstance(bg_removed_result, np.ndarray):
                bg_removed_image = Image.fromarray(bg_removed_result)
            else:
                # bytes나 다른 타입인 경우 처리
                bg_removed_image = remove_background_threshold(masked_image, threshold)
        except Exception as e:
            print(f"rembg 사용 중 오류 발생: {e}, 임계값 방식으로 대체합니다.")
            bg_removed_image = remove_background_threshold(masked_image, threshold)
    else:
        # 임계값 기반 배경 제거
        bg_removed_image = remove_background_threshold(masked_image, threshold)
    
    # 배경 제거된 영역을 원본 이미지와 합성
    result = original_rgba.copy()
    
    # 마스크 영역에만 배경 제거 결과 적용
    mask_array = np.array(mask)
    result_array = np.array(result)
    bg_removed_array = np.array(bg_removed_image.convert('RGBA'))
    
    # 마스크가 적용된 영역에만 배경 제거 결과를 합성
    for i in range(4):  # RGBA 채널
        result_array[:, :, i] = np.where(
            mask_array > 128,  # 마스크가 활성화된 영역
            bg_removed_array[:, :, i],  # 배경 제거된 이미지
            result_array[:, :, i]  # 원본 이미지
        )
    
    return Image.fromarray(result_array)

def create_mask(image_size, mask_coords, mask_type):
    """마스크 이미지 생성"""
    width, height = image_size
    mask = Image.new('L', (width, height), 0)  # 검은색 배경
    draw = ImageDraw.Draw(mask)
    
    if mask_type == "rectangle" and len(mask_coords) == 4:
        # 사각형 마스크
        x1, y1, x2, y2 = mask_coords
        draw.rectangle([x1, y1, x2, y2], fill=255)
    
    elif mask_type == "polygon" and len(mask_coords) >= 3:
        # 폴리곤 마스크
        draw.polygon(mask_coords, fill=255)
    
    return mask

def apply_mask_to_image(image, mask):
    """이미지에 마스크 적용"""
    if image.mode != 'RGBA':
        image_rgba = image.convert('RGBA')
    else:
        image_rgba = image.copy()
    
    # 마스크를 알파 채널로 사용
    image_array = np.array(image_rgba)
    mask_array = np.array(mask)
    
    # 마스크되지 않은 영역은 투명하게 만듦
    image_array[:, :, 3] = np.where(mask_array > 128, image_array[:, :, 3], 0)
    
    return Image.fromarray(image_array)

def remove_background_threshold(image, threshold=240):
    """임계값 기반 배경 제거"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            # RGB 채널만 처리
            rgb = img_array[:, :, :3]
            mask = np.all(rgb > threshold, axis=2)
            rgba = img_array.copy()
            rgba[:, :, 3] = np.where(mask, 0, rgba[:, :, 3])
        else:  # RGB
            # 밝은 배경 픽셀 마스크 생성
            mask = np.all(img_array > threshold, axis=2)
            rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = img_array
            rgba[:, :, 3] = np.where(mask, 0, 255)
        
        return Image.fromarray(rgba)
    
    return image

# 이미지 크기 조정 함수
def resize_image(image, size):
    """Resize an image to the specified size"""
    image = image.resize(size)
    return np.array(image)

# 이미지 회전 함수 - 짤림 방지를 위해 expand=True 추가
def rotate_image(image, angle):
    """Rotate an image by the given angle"""
    return image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

# 이미지 좌우 반전 함수
def flip_image_horizontal(image):
    """Flip an image horizontally"""
    return ImageOps.mirror(image)

# 이미지 밝기 조정 함수
def adjust_brightness(image, factor):
    """Adjust the brightness of an image"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# 이미지 대비 조정 함수
def adjust_contrast(image, factor):
    """Adjust the contrast of an image"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

# 이미지 색상 조정 함수
def adjust_color(image, factor):
    """Adjust the color saturation of an image"""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

# 이미지 선명도 조정 함수
def adjust_sharpness(image, factor):
    """Adjust the sharpness of an image"""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

# 이미지 크롭 함수
def crop_image(image, left, top, right, bottom):
    """Crop an image to the specified coordinates"""
    return image.crop((left, top, right, bottom))

# 이미지를 base64로 변환하는 함수
def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# base64를 이미지로 변환하는 함수
def base64_to_image(base64_str):
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))