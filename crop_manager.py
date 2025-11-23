# crop_manager.py - 크롭 이미지 관리 모듈

import streamlit as st
from PIL import Image
import os
import json
from datetime import datetime
import uuid

class CropManager:
    """크롭된 이미지들을 관리하는 클래스"""
    
    def __init__(self, storage_dir="Result/crops"):
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "crop_metadata.json")
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """저장 디렉토리가 없으면 생성"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def save_crop(self, image, source_image_name, crop_coords, description=""):
        """크롭된 이미지를 저장하고 메타데이터 기록"""
        crop_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crop_{timestamp}_{crop_id}.png"
        filepath = os.path.join(self.storage_dir, filename)
        
        # 이미지 저장
        image.save(filepath)
        
        # 메타데이터 저장
        metadata = {
            "id": crop_id,
            "filename": filename,
            "filepath": filepath,
            "source_image": source_image_name,
            "crop_coords": crop_coords,
            "description": description,
            "timestamp": timestamp,
            "size": f"{image.width}x{image.height}",
            "created_at": datetime.now().isoformat()
        }
        
        self._save_metadata(metadata)
        return crop_id, filepath
    
    def _save_metadata(self, metadata):
        """메타데이터를 JSON 파일에 저장"""
        all_metadata = self.load_all_metadata()
        all_metadata.append(metadata)
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    def load_all_metadata(self):
        """모든 크롭 메타데이터 로드"""
        if not os.path.exists(self.metadata_file):
            return []
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def get_crop_by_id(self, crop_id):
        """ID로 크롭 이미지 정보 조회"""
        metadata_list = self.load_all_metadata()
        for metadata in metadata_list:
            if metadata["id"] == crop_id:
                return metadata
        return None
    
    def load_crop_image(self, crop_id):
        """ID로 크롭 이미지 로드"""
        metadata = self.get_crop_by_id(crop_id)
        if metadata and os.path.exists(metadata["filepath"]):
            return Image.open(metadata["filepath"])
        return None
    
    def delete_crop(self, crop_id):
        """크롭 이미지 삭제"""
        metadata_list = self.load_all_metadata()
        updated_metadata = []
        deleted = False
        
        for metadata in metadata_list:
            if metadata["id"] == crop_id:
                # 파일 삭제
                if os.path.exists(metadata["filepath"]):
                    os.remove(metadata["filepath"])
                deleted = True
            else:
                updated_metadata.append(metadata)
        
        if deleted:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(updated_metadata, f, ensure_ascii=False, indent=2)
        
        return deleted
    
    def get_crops_by_source(self, source_image_name):
        """특정 소스 이미지에서 나온 크롭들 조회"""
        metadata_list = self.load_all_metadata()
        return [meta for meta in metadata_list if meta["source_image"] == source_image_name]

def display_crop_gallery():
    """크롭 갤러리 UI 표시"""
    st.markdown("### 🖼️ 크롭 이미지 갤러리")
    
    crop_manager = CropManager()
    all_crops = crop_manager.load_all_metadata()
    
    if not all_crops:
        st.info("저장된 크롭 이미지가 없습니다.")
        return
    
    # 최신순으로 정렬
    all_crops.sort(key=lambda x: x["created_at"], reverse=True)
    
    # 페이지네이션 설정
    items_per_page = 6
    total_pages = (len(all_crops) + items_per_page - 1) // items_per_page
    
    if total_pages > 1:
        page = st.selectbox(
            f"페이지 선택 (총 {len(all_crops)}개 크롭)",
            range(1, total_pages + 1),
            format_func=lambda x: f"페이지 {x}"
        )
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(all_crops))
        page_crops = all_crops[start_idx:end_idx]
    else:
        page_crops = all_crops
    
    # 그리드 형태로 크롭 이미지 표시
    cols = st.columns(3)
    for idx, crop_meta in enumerate(page_crops):
        col = cols[idx % 3]
        
        with col:
            crop_image = crop_manager.load_crop_image(crop_meta["id"])
            if crop_image:
                st.image(crop_image, caption=f"ID: {crop_meta['id']}", use_column_width=True)
                
                with st.expander(f"상세 정보 - {crop_meta['id']}"):
                    st.write(f"**소스:** {crop_meta['source_image']}")
                    st.write(f"**크기:** {crop_meta['size']}")
                    st.write(f"**생성일:** {crop_meta['timestamp']}")
                    if crop_meta.get('description'):
                        st.write(f"**설명:** {crop_meta['description']}")
                    
                    # 삭제 버튼
                    if st.button(f"🗑️ 삭제", key=f"delete_{crop_meta['id']}"):
                        if crop_manager.delete_crop(crop_meta["id"]):
                            st.success("크롭 이미지가 삭제되었습니다.")
                            st.rerun()
                        else:
                            st.error("삭제에 실패했습니다.")

def crop_comparison_interface():
    """크롭 비교 전용 인터페이스"""
    st.markdown("### 🔍 저장된 크롭 이미지 비교")
    
    crop_manager = CropManager()
    all_crops = crop_manager.load_all_metadata()
    
    if len(all_crops) < 2:
        st.warning("비교하려면 최소 2개의 크롭 이미지가 필요합니다.")
        return None, None, None, None
    
    # 크롭 선택 드롭다운
    crop_options = {f"{crop['id']} - {crop['source_image']} ({crop['size']})": crop['id'] 
                   for crop in all_crops}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 첫 번째 크롭")
        selected_crop1_key = st.selectbox("첫 번째 크롭 선택", list(crop_options.keys()), key="crop_select1")
        crop1_id = crop_options[selected_crop1_key]
        crop1_image = crop_manager.load_crop_image(crop1_id)
        if crop1_image:
            st.image(crop1_image, caption=f"크롭 ID: {crop1_id}", use_column_width=True)
    
    with col2:
        st.markdown("#### 두 번째 크롭")
        selected_crop2_key = st.selectbox("두 번째 크롭 선택", list(crop_options.keys()), key="crop_select2")
        crop2_id = crop_options[selected_crop2_key]
        crop2_image = crop_manager.load_crop_image(crop2_id)
        if crop2_image:
            st.image(crop2_image, caption=f"크롭 ID: {crop2_id}", use_column_width=True)
    
    return crop1_image, crop2_image, crop1_id, crop2_id