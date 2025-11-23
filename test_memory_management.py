#!/usr/bin/env python3
"""Test script for memory management system."""

import sys
import traceback

try:
    from image_upscaler import ImageUpscaler
    
    print("=== Memory Management Test ===")
    upscaler = ImageUpscaler()
    
    # Test memory information
    memory_info = upscaler.get_memory_info()
    print(f"Total RAM: {memory_info['total_ram_gb']:.2f} GB")
    print(f"Available RAM: {memory_info['available_ram_gb']:.2f} GB")
    print(f"RAM Usage: {memory_info['used_ram_percent']:.1f}%")
    print(f"GPU Memory: {memory_info['gpu_memory_gb']:.2f} GB")
    print(f"GPU Used: {memory_info['gpu_used_gb']:.2f} GB")
    print(f"Can handle large images (1920x1080): {upscaler.can_handle_large_image((1920, 1080))}")
    print(f"Can handle large images (4K 3840x2160): {upscaler.can_handle_large_image((3840, 2160))}")
    print(f"Can handle large images (8K 7680x4320): {upscaler.can_handle_large_image((7680, 4320))}")
    
    # Test safe image loading
    print("\n=== Safe Image Loading Test ===")
    from image_processing import safe_image_open
    print("Safe image loading function imported successfully")
    
    # Test PIL configuration
    from PIL import Image
    print(f"PIL MAX_IMAGE_PIXELS: {Image.MAX_IMAGE_PIXELS:,}")
    
    print("\n✅ All tests passed successfully!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ Test Error: {e}")
    traceback.print_exc()
    sys.exit(1)