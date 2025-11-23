import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10
from scipy.spatial.distance import cosine
import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
from PIL import Image
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("경고: LPIPS 패키지가 설치되지 않았습니다. pip install lpips로 설치하세요.")

# 이미지 형식 통일 함수
def normalize_image_format(img):
    """이미지를 RGB 형식으로 통일"""
    if isinstance(img, Image.Image):
        # PIL Image인 경우
        if img.mode == 'RGBA':
            # RGBA를 RGB로 변환 (투명 배경을 흰색으로)
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])  # 알파 채널을 마스크로 사용
            return np.array(background)
        elif img.mode == 'RGB':
            return np.array(img)
        else:
            return np.array(img.convert('RGB'))
    else:
        # numpy array인 경우
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                # 알파 채널 제거하고 RGB로 변환
                rgb_array = img_array[:, :, :3]
                alpha = img_array[:, :, 3] / 255.0
                # 투명한 부분을 흰색 배경으로 합성
                for i in range(3):
                    rgb_array[:, :, i] = rgb_array[:, :, i] * alpha + 255 * (1 - alpha)
                return rgb_array.astype(np.uint8)
            elif img_array.shape[2] == 3:  # RGB
                return img_array
            else:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 2:  # Grayscale
            return cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"지원하지 않는 이미지 형식: {img_array.shape}")

# SSIM 비교 함수
def compare_ssim(img1, img2):
    # 이미지 형식 통일
    img1_normalized = normalize_image_format(img1)
    img2_normalized = normalize_image_format(img2)
    
    # 그레이스케일로 변환
    gray1 = cv2.cvtColor(img1_normalized, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_normalized, cv2.COLOR_RGB2GRAY)
    
    # 이미지 크기 통일
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    result = ssim(gray1, gray2)
    # ssim 함수가 tuple을 반환하는 경우 첫 번째 값만 사용
    if isinstance(result, tuple):
        score = result[0]
    else:
        score = result
    return float(score * 100)  # float32를 일반 float으로 변환

# PSNR 비교 함수 (MAX_PSNR을 기준으로 백분율 계산)
def compare_psnr(img1, img2):
    # 이미지 형식 통일
    img1_normalized = normalize_image_format(img1)
    img2_normalized = normalize_image_format(img2)
    
    # 이미지 크기 통일
    if img1_normalized.shape != img2_normalized.shape:
        img2_normalized = cv2.resize(img2_normalized, (img1_normalized.shape[1], img1_normalized.shape[0]))
    
    mse = np.mean((img1_normalized.astype(np.float64) - img2_normalized.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    psnr = 20 * log10(255.0 / np.sqrt(mse))
    # PSNR을 백분율로 변환 (50dB를 100%로 가정)
    MAX_PSNR = 50.0
    percentage = min(psnr / MAX_PSNR * 100, 100)
    return float(percentage)  # numpy 타입을 일반 float으로 변환

# VGG16 기반 Cosine 유사도 비교 함수 (PyTorch 사용)
def compare_vgg_cosine(img1, img2):
    # 이미지를 PIL Image로 변환
    if isinstance(img1, np.ndarray):
        img1_normalized = normalize_image_format(img1)
        img1 = Image.fromarray(img1_normalized)
    elif img1.mode != 'RGB':
        img1_normalized = normalize_image_format(img1)
        img1 = Image.fromarray(img1_normalized)
    
    if isinstance(img2, np.ndarray):
        img2_normalized = normalize_image_format(img2)
        img2 = Image.fromarray(img2_normalized)
    elif img2.mode != 'RGB':
        img2_normalized = normalize_image_format(img2)
        img2 = Image.fromarray(img2_normalized)
    
    # Load pretrained model
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    # Remove the classifier to get features only
    model = torch.nn.Sequential(*list(model.children())[:-1])
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process images
    try:
        img1_tensor = preprocess(img1)
        img2_tensor = preprocess(img2)
        
        # Add batch dimension if needed
        if torch.is_tensor(img1_tensor) and len(img1_tensor.shape) == 3:
            img1_tensor = img1_tensor.unsqueeze(0)
        if torch.is_tensor(img2_tensor) and len(img2_tensor.shape) == 3:
            img2_tensor = img2_tensor.unsqueeze(0)
    except Exception as e:
        print(f"이미지 전처리 중 오류: {e}")
        return 0.0
    
    # Get features
    with torch.no_grad():
        feat1 = model(img1_tensor).flatten().numpy()
        feat2 = model(img2_tensor).flatten().numpy()
    
    return float((1 - cosine(feat1, feat2)) * 100)  # numpy 타입을 일반 float으로 변환

# LPIPS 비교 함수 (Learned Perceptual Image Patch Similarity)
def compare_lpips(img1, img2, net='alex'):
    """
    LPIPS를 사용한 지각적 이미지 유사도 측정
    net: 'alex', 'vgg', 'squeeze' 중 선택
    반환값: 유사도 백분율 (낮을수록 유사함)
    """
    if not LPIPS_AVAILABLE:
        print("LPIPS 패키지가 설치되지 않았습니다. VGG Cosine 유사도를 대신 사용합니다.")
        return compare_vgg_cosine(img1, img2)
    
    try:
        # LPIPS 모델 로드 (캐싱을 위해 global 변수 사용)
        if not hasattr(compare_lpips, 'lpips_model'):
            compare_lpips.lpips_model = lpips.LPIPS(net=net, verbose=False)
            compare_lpips.lpips_model.eval()
        
        model = compare_lpips.lpips_model
        
        # 이미지를 PIL Image로 변환
        if isinstance(img1, np.ndarray):
            img1_normalized = normalize_image_format(img1)
            img1 = Image.fromarray(img1_normalized)
        elif img1.mode != 'RGB':
            img1_normalized = normalize_image_format(img1)
            img1 = Image.fromarray(img1_normalized)
        
        if isinstance(img2, np.ndarray):
            img2_normalized = normalize_image_format(img2)
            img2 = Image.fromarray(img2_normalized)
        elif img2.mode != 'RGB':
            img2_normalized = normalize_image_format(img2)
            img2 = Image.fromarray(img2_normalized)
        
        # LPIPS를 위한 전처리 (-1~1 범위로 정규화)
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),  # LPIPS는 보통 256x256 사용
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] 범위로
        ])
        
        # 이미지 텐서 변환
        img1_tensor = preprocess(img1)
        img2_tensor = preprocess(img2)
        
        # 배치 차원 추가
        if torch.is_tensor(img1_tensor):
            img1_tensor = img1_tensor.unsqueeze(0)
        if torch.is_tensor(img2_tensor):
            img2_tensor = img2_tensor.unsqueeze(0)
        
        # LPIPS 계산
        with torch.no_grad():
            lpips_distance = model(img1_tensor, img2_tensor)
            lpips_score = float(lpips_distance.item())
        
        # LPIPS는 거리 메트릭 (0: 동일, 1: 완전히 다름)
        # 원래 특성에 맞게 0-1 사이의 값으로 반환
        return float(lpips_score)
        
    except Exception as e:
        print(f"LPIPS 계산 중 오류 발생: {e}")
        print("VGG Cosine 유사도를 대신 사용합니다.")
        return compare_vgg_cosine(img1, img2)

# 고급 LPIPS 비교 함수 (여러 네트워크 앙상블)
def compare_lpips_ensemble(img1, img2):
    """
    여러 LPIPS 네트워크의 앙상블을 사용한 더 정확한 유사도 측정
    """
    if not LPIPS_AVAILABLE:
        return compare_vgg_cosine(img1, img2)
    
    try:
        networks = ['alex', 'vgg']
        scores = []
        
        for net in networks:
            try:
                score = compare_lpips(img1, img2, net=net)
                scores.append(score)
            except Exception as e:
                print(f"LPIPS {net} 네트워크에서 오류: {e}")
                continue
        
        if scores:
            # LPIPS 점수들의 평균 (0-1 거리값)
            return float(np.mean(scores))
        else:
            # 모든 LPIPS 네트워크가 실패한 경우 VGG Cosine을 LPIPS 스케일로 변환
            vgg_similarity = compare_vgg_cosine(img1, img2)
            # VGG Cosine 유사도(0-100%)를 LPIPS 거리값(0-1)으로 변환
            return float(max(0, min(1, (100 - vgg_similarity) / 100)))
            
    except Exception as e:
        print(f"LPIPS 앙상블 계산 중 오류: {e}")
        return compare_vgg_cosine(img1, img2)