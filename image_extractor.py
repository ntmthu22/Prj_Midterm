"""
IMAGE EXTRACTOR VỚI BOUNDING BOX TỪ PDF
Extract hình ảnh từ PDF và lưu với thông tin bounding box
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import json
import fitz  # PyMuPDF - Tốt hơn cho extract images từ PDF

class PDFImageExtractor:
    """Extract hình ảnh từ PDF với bounding box chính xác"""
    
    def __init__(self, min_width=100, min_height=100):
        """
        Args:
            min_width: Chiều rộng tối thiểu của ảnh (pixels)
            min_height: Chiều cao tối thiểu của ảnh (pixels)
        """
        self.min_width = min_width
        self.min_height = min_height
        
    def extract_images_pymupdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract images trực tiếp từ PDF bằng PyMuPDF
        Phương pháp này chính xác hơn vì lấy metadata trực tiếp từ PDF
        
        Returns:
            List of dict: {
                'page': số trang,
                'image': numpy array,
                'bbox': (x0, y0, x1, y1),
                'width': width,
                'height': height,
                'image_index': index trong trang
            }
        """
        print(f"📄 Đang extract images từ PDF bằng PyMuPDF...")
        
        doc = fitz.open(pdf_path)
        all_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_num_display = page_num + 1
            
            print(f"  📑 Trang {page_num_display}/{len(doc)}", end='\r')
            
            # Lấy danh sách images trong trang
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]  # xref number
                
                # Lấy bounding box của image trong trang
                # get_image_rects trả về list các rectangle chứa image này
                rects = page.get_image_rects(xref)
                
                if not rects:
                    continue
                
                # Lấy image data
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Chuyển bytes thành numpy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    h, w = img.shape[:2]
                    
                    # Lọc theo kích thước
                    if w < self.min_width or h < self.min_height:
                        continue
                    
                    # Lấy bbox đầu tiên (nếu image xuất hiện nhiều lần thì có nhiều rects)
                    rect = rects[0]
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    
                    all_images.append({
                        'page': page_num_display,
                        'image': img,
                        'bbox': bbox,
                        'width': w,
                        'height': h,
                        'image_index': img_index + 1,
                        'format': base_image.get("ext", "png")
                    })
                    
                except Exception as e:
                    print(f"\n⚠️  Lỗi extract image {img_index} trang {page_num_display}: {e}")
                    continue
        
        doc.close()
        print(f"\n✅ Đã extract {len(all_images)} hình ảnh")
        return all_images
    
    def extract_images_opencv(self, pdf_path: str, dpi: int = 300) -> List[Dict]:
        """
        Extract images bằng cách convert PDF sang ảnh rồi phát hiện vùng
        Phương pháp này tốt cho các hình ảnh embedded/scanned
        
        Returns:
            List of dict tương tự extract_images_pymupdf
        """
        print(f"📄 Đang extract images bằng OpenCV...")
        
        # Chuyển PDF sang ảnh
        print("  🔄 Chuyển đổi PDF sang ảnh...")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        all_extracted = []
        
        for page_num, page_img in enumerate(images, 1):
            print(f"  📑 Phân tích trang {page_num}/{len(images)}", end='\r')
            
            # Chuyển sang numpy array
            img_array = np.array(page_img)
            
            # Phát hiện vùng ảnh
            detected_regions = self._detect_image_regions(img_array)
            
            for img_index, region in enumerate(detected_regions, 1):
                x, y, w, h = region['bbox']
                
                # Lọc theo kích thước
                if w < self.min_width or h < self.min_height:
                    continue
                
                # Crop ảnh
                cropped = img_array[y:y+h, x:x+w]
                
                all_extracted.append({
                    'page': page_num,
                    'image': cropped,
                    'bbox': (x, y, x+w, y+h),
                    'width': w,
                    'height': h,
                    'image_index': img_index,
                    'confidence': region.get('confidence', 1.0),
                    'format': 'png'
                })
        
        print(f"\n✅ Đã extract {len(all_extracted)} hình ảnh")
        return all_extracted
    
    def _detect_image_regions(self, image: np.ndarray) -> List[Dict]:
        """Phát hiện các vùng ảnh trong trang"""
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Threshold để tách foreground/background
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Tính các đặc điểm
            area = w * h
            contour_area = cv2.contourArea(contour)
            solidity = contour_area / area if area > 0 else 0
            aspect_ratio = w / h if h > 0 else 0
            
            # Lọc các region giống ảnh
            # Ảnh thường có solidity cao và kích thước hợp lý
            if solidity > 0.7 and 50 < w and 50 < h:
                confidence = self._calculate_image_confidence(
                    image[y:y+h, x:x+w], solidity, aspect_ratio, area
                )
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'solidity': solidity,
                    'aspect_ratio': aspect_ratio
                })
        
        # Sắp xếp theo confidence
        regions.sort(key=lambda r: r['confidence'], reverse=True)
        
        return regions
    
    def _calculate_image_confidence(self, region: np.ndarray, 
                                   solidity: float, aspect_ratio: float, 
                                   area: int) -> float:
        """Tính confidence score cho vùng ảnh"""
        
        confidence = 0.0
        
        # Điểm cho solidity (ảnh thường có đường viền rõ ràng)
        if solidity > 0.9:
            confidence += 0.3
        elif solidity > 0.8:
            confidence += 0.2
        
        # Điểm cho aspect ratio (ảnh thường không quá méo)
        if 0.3 < aspect_ratio < 3.0:
            confidence += 0.2
        
        # Điểm cho kích thước (ảnh lớn thường quan trọng hơn)
        if area > 100000:
            confidence += 0.3
        elif area > 50000:
            confidence += 0.2
        elif area > 20000:
            confidence += 0.1
        
        # Điểm cho độ phức tạp (ảnh thường có nhiều chi tiết)
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.1:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def save_extracted_images(self, extracted_images: List[Dict], 
                             output_dir: str,
                             draw_bbox: bool = True,
                             save_metadata: bool = True):
        """
        Lưu các ảnh đã extract
        
        Args:
            extracted_images: List các ảnh từ extract_images_*
            output_dir: Thư mục output
            draw_bbox: Vẽ bounding box lên ảnh gốc
            save_metadata: Lưu file metadata JSON
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Tạo thư mục con
        images_dir = os.path.join(output_dir, 'extracted_images')
        bbox_dir = os.path.join(output_dir, 'bbox_visualizations')
        os.makedirs(images_dir, exist_ok=True)
        if draw_bbox:
            os.makedirs(bbox_dir, exist_ok=True)
        
        metadata = {
            'total_images': len(extracted_images),
            'images': []
        }
        
        print(f"\n💾 Đang lưu {len(extracted_images)} hình ảnh...")
        
        # Group theo trang
        pages = {}
        for img_data in extracted_images:
            page = img_data['page']
            if page not in pages:
                pages[page] = []
            pages[page].append(img_data)
        
        # Lưu từng ảnh
        for page_num in sorted(pages.keys()):
            page_images = pages[page_num]
            
            for img_data in page_images:
                img_index = img_data['image_index']
                img = img_data['image']
                bbox = img_data['bbox']
                
                # Tên file
                filename = f"page_{page_num:03d}_img_{img_index:02d}.png"
                filepath = os.path.join(images_dir, filename)
                
                # Lưu ảnh
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, img_bgr)
                
                # Metadata
                metadata['images'].append({
                    'filename': filename,
                    'page': page_num,
                    'image_index': img_index,
                    'bbox': {
                        'x0': float(bbox[0]),
                        'y0': float(bbox[1]),
                        'x1': float(bbox[2]),
                        'y1': float(bbox[3])
                    },
                    'width': img_data['width'],
                    'height': img_data['height'],
                    'format': img_data.get('format', 'png')
                })
                
                print(f"  ✓ Đã lưu: {filename}", end='\r')
        
        print(f"\n✅ Đã lưu tất cả ảnh vào: {images_dir}")
        
        # Lưu metadata
        if save_metadata:
            metadata_file = os.path.join(output_dir, 'images_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"✅ Đã lưu metadata: {metadata_file}")
        
        return metadata
    
    def visualize_bboxes_on_pdf(self, pdf_path: str, 
                               extracted_images: List[Dict],
                               output_dir: str,
                               dpi: int = 150):
        """
        Vẽ bounding boxes lên các trang PDF để visualize
        
        Args:
            pdf_path: Đường dẫn PDF gốc
            extracted_images: List ảnh đã extract
            output_dir: Thư mục lưu visualization
            dpi: Độ phân giải để render PDF
        """
        print(f"\n🎨 Đang tạo visualization...")
        
        bbox_dir = os.path.join(output_dir, 'bbox_visualizations')
        os.makedirs(bbox_dir, exist_ok=True)
        
        # Group theo trang
        pages_data = {}
        for img_data in extracted_images:
            page = img_data['page']
            if page not in pages_data:
                pages_data[page] = []
            pages_data[page].append(img_data)
        
        # Render PDF pages
        print("  🔄 Rendering PDF pages...")
        pdf_images = convert_from_path(pdf_path, dpi=dpi)
        
        # Vẽ bbox cho từng trang
        for page_num, page_img in enumerate(pdf_images, 1):
            if page_num not in pages_data:
                continue
            
            img_array = np.array(page_img)
            
            # Vẽ từng bbox
            for img_data in pages_data[page_num]:
                bbox = img_data['bbox']
                img_index = img_data['image_index']
                
                # Scale bbox nếu DPI khác
                scale = dpi / 72  # PDF thường 72 DPI
                x0, y0, x1, y1 = bbox
                x0, y0, x1, y1 = int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)
                
                # Vẽ rectangle
                cv2.rectangle(img_array, (x0, y0), (x1, y1), (0, 255, 0), 3)
                
                # Vẽ label
                label = f"IMG {img_index}"
                font_scale = 0.8
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                       font_scale, thickness)
                
                # Background cho text
                cv2.rectangle(img_array, (x0, y0 - text_h - 10), 
                            (x0 + text_w + 10, y0), (0, 255, 0), -1)
                
                # Text
                cv2.putText(img_array, label, (x0 + 5, y0 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            # Lưu ảnh
            output_file = os.path.join(bbox_dir, f'page_{page_num:03d}_bbox.png')
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_file, img_bgr)
            
            print(f"  ✓ Trang {page_num}: {len(pages_data[page_num])} ảnh", end='\r')
        
        print(f"\n✅ Đã lưu visualization vào: {bbox_dir}")


def main():
    """Ví dụ sử dụng"""
    
    print("=" * 70)
    print("🖼️  PDF IMAGE EXTRACTOR VỚI BOUNDING BOX")
    print("=" * 70)
    print()
    
    # ========== CẤU HÌNH ==========
    pdf_file = "input/document.pdf"
    output_directory = "output/extracted"
    
    use_pymupdf = True  # True = dùng PyMuPDF (chính xác hơn)
                        # False = dùng OpenCV (tốt cho scanned PDF)
    
    min_image_size = (100, 100)  # (width, height) tối thiểu
    # ==============================
    
    # Kiểm tra file
    if not os.path.exists(pdf_file):
        print(f"❌ Không tìm thấy file: {pdf_file}")
        input("Nhấn Enter để thoát...")
        return
    
    try:
        # Khởi tạo extractor
        extractor = PDFImageExtractor(
            min_width=min_image_size[0],
            min_height=min_image_size[1]
        )
        
        # Extract images
        print(f"\n📖 Đang xử lý: {pdf_file}\n")
        
        if use_pymupdf:
            extracted = extractor.extract_images_pymupdf(pdf_file)
        else:
            extracted = extractor.extract_images_opencv(pdf_file, dpi=300)
        
        if not extracted:
            print("\n⚠️  Không tìm thấy hình ảnh nào trong PDF!")
            input("Nhấn Enter để thoát...")
            return
        
        # Lưu ảnh và metadata
        metadata = extractor.save_extracted_images(
            extracted,
            output_directory,
            draw_bbox=True,
            save_metadata=True
        )
        
        # Tạo visualization
        extractor.visualize_bboxes_on_pdf(
            pdf_file,
            extracted,
            output_directory,
            dpi=150
        )
        
        # Thống kê
        print("\n" + "=" * 70)
        print("📊 THỐNG KÊ")
        print("=" * 70)
        
        pages_count = {}
        for img in extracted:
            page = img['page']
            pages_count[page] = pages_count.get(page, 0) + 1
        
        print(f"\n✅ Tổng số ảnh: {len(extracted)}")
        print(f"📄 Số trang có ảnh: {len(pages_count)}")
        print(f"\n📑 Chi tiết theo trang:")
        for page in sorted(pages_count.keys()):
            print(f"   - Trang {page}: {pages_count[page]} ảnh")
        
        print(f"\n📂 Kết quả đã lưu:")
        print(f"   - Ảnh gốc: {output_directory}/extracted_images/")
        print(f"   - Visualization: {output_directory}/bbox_visualizations/")
        print(f"   - Metadata: {output_directory}/images_metadata.json")
        
        print("\n" + "=" * 70)
        print("🎉 HOÀN THÀNH!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nNhấn Enter để thoát...")


if __name__ == "__main__":
    main()