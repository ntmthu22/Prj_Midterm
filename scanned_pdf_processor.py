"""
XỬ LÝ PDF SCAN - PHÁT HIỆN VÙNG ẢNH TRONG TRANG
Dành cho các PDF được scan (mỗi trang là 1 ảnh lớn)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from pdf2image import convert_from_path
from PIL import Image
import json
import fitz  # PyMuPDF


class ScannedPDFImageDetector:
    """Phát hiện vùng ảnh trong PDF scan"""
    
    def __init__(self, min_width=150, min_height=150):
        self.min_width = min_width
        self.min_height = min_height
        
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Kiểm tra PDF có phải scan không
        Nếu mỗi trang chỉ có 1 ảnh lớn = PDF scan
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Lấy vài trang đầu để test
            pages_to_check = min(3, len(doc))
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                images = page.get_images(full=True)
                
                # Nếu trang có nhiều hơn 1 ảnh = không phải scan đơn giản
                if len(images) > 1:
                    doc.close()
                    return False
                
                # Nếu có 1 ảnh, kiểm tra kích thước
                if len(images) == 1:
                    xref = images[0][0]
                    base_image = doc.extract_image(xref)
                    img_width = base_image["width"]
                    img_height = base_image["height"]
                    
                    page_rect = page.rect
                    page_width = page_rect.width
                    page_height = page_rect.height
                    
                    # Nếu ảnh chiếm gần như toàn bộ trang = scan
                    if img_width > page_width * 0.9 and img_height > page_height * 0.9:
                        continue
                    else:
                        doc.close()
                        return False
            
            doc.close()
            return True
            
        except Exception as e:
            print(f"⚠️  Lỗi kiểm tra PDF: {e}")
            return False
    
    def preprocess_scanned_page(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý trang scan để phát hiện vùng ảnh tốt hơn
        """
        # Chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Khử nhiễu
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Tăng độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def detect_image_regions_advanced(self, image: np.ndarray) -> List[Dict]:
        """
        Phát hiện vùng ảnh nâng cao cho PDF scan
        Sử dụng nhiều kỹ thuật kết hợp
        """
        h, w = image.shape[:2]
        
        # Tiền xử lý
        processed = self.preprocess_scanned_page(image)
        
        # PHƯƠNG PHÁP 1: Edge Detection + Contours
        regions_edge = self._detect_by_edges(processed, image)
        
        # PHƯƠNG PHÁP 2: Color/Texture Analysis
        regions_texture = self._detect_by_texture(image)
        
        # PHƯƠNG PHÁP 3: Connected Components
        regions_components = self._detect_by_components(processed, image)
        
        # Kết hợp các phương pháp
        all_regions = regions_edge + regions_texture + regions_components
        
        # Loại bỏ trùng lặp và merge overlapping boxes
        merged_regions = self._merge_overlapping_regions(all_regions, image)
        
        # Lọc và sắp xếp
        filtered = [r for r in merged_regions if self._is_valid_image_region(r, image)]
        filtered.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))
        
        return filtered
    
    def _detect_by_edges(self, processed: np.ndarray, original: np.ndarray) -> List[Dict]:
        """Phát hiện bằng edge detection"""
        regions = []
        
        # Canny edge detection
        edges = cv2.Canny(processed, 30, 100)
        
        # Dilate để kết nối các cạnh gần nhau
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Tìm contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < self.min_width or h < self.min_height:
                continue
            
            # Tính các đặc trưng
            area = w * h
            contour_area = cv2.contourArea(contour)
            solidity = contour_area / area if area > 0 else 0
            aspect_ratio = w / h if h > 0 else 0
            
            # Lọc: ảnh thường có đường viền rõ ràng
            if solidity > 0.5:
                confidence = self._calculate_confidence(
                    original[y:y+h, x:x+w], 'edge', solidity, aspect_ratio
                )
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'method': 'edge',
                    'solidity': solidity
                })
        
        return regions
    
    def _detect_by_texture(self, image: np.ndarray) -> List[Dict]:
        """Phát hiện bằng phân tích texture"""
        regions = []
        
        # Chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Tính variance để phát hiện vùng có nhiều chi tiết
        # Ảnh thường có variance cao hơn text
        window_size = 50
        variance_map = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(0, gray.shape[0] - window_size, window_size):
            for j in range(0, gray.shape[1] - window_size, window_size):
                window = gray[i:i+window_size, j:j+window_size]
                variance_map[i:i+window_size, j:j+window_size] = np.var(window)
        
        # Threshold: vùng có variance cao = có thể là ảnh
        _, texture_mask = cv2.threshold(
            variance_map.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Morphology để làm sạch
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < self.min_width or h < self.min_height:
                continue
            
            confidence = self._calculate_confidence(
                image[y:y+h, x:x+w], 'texture', 0, w/h if h > 0 else 0
            )
            
            regions.append({
                'bbox': (x, y, w, h),
                'confidence': confidence,
                'method': 'texture'
            })
        
        return regions
    
    def _detect_by_components(self, processed: np.ndarray, original: np.ndarray) -> List[Dict]:
        """Phát hiện bằng connected components"""
        regions = []
        
        # Threshold adaptive
        binary = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Loại bỏ text bằng morphology
        # Text thường có kích thước nhỏ và mỏng
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        without_text = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Kết nối các thành phần gần nhau
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        connected = cv2.morphologyEx(without_text, cv2.MORPH_CLOSE, kernel_large)
        
        # Tìm connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
        
        for i in range(1, num_labels):  # Bỏ background (0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            if w < self.min_width or h < self.min_height:
                continue
            
            # Tính solidity
            bbox_area = w * h
            solidity = area / bbox_area if bbox_area > 0 else 0
            
            if solidity > 0.3:  # Lọc vùng quá thưa
                confidence = self._calculate_confidence(
                    original[y:y+h, x:x+w], 'component', solidity, w/h if h > 0 else 0
                )
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'method': 'component',
                    'solidity': solidity
                })
        
        return regions
    
    def _calculate_confidence(self, region: np.ndarray, method: str,
                             solidity: float, aspect_ratio: float) -> float:
        """Tính confidence score"""
        confidence = 0.0
        
        # Base confidence theo method
        method_weights = {
            'edge': 0.3,
            'texture': 0.2,
            'component': 0.25
        }
        confidence += method_weights.get(method, 0.2)
        
        # Solidity
        if solidity > 0.8:
            confidence += 0.2
        elif solidity > 0.6:
            confidence += 0.15
        
        # Aspect ratio (ảnh thường không quá méo)
        if 0.3 < aspect_ratio < 3.0:
            confidence += 0.15
        
        # Kích thước
        h, w = region.shape[:2]
        area = h * w
        if area > 100000:
            confidence += 0.2
        elif area > 50000:
            confidence += 0.15
        elif area > 20000:
            confidence += 0.1
        
        # Phân tích chi tiết region
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Edge density (ảnh có nhiều cạnh)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > 0.1:
            confidence += 0.1
        
        # Color variance (ảnh thường đa sắc màu hơn text)
        if len(region.shape) == 3:
            std_per_channel = [np.std(region[:,:,i]) for i in range(3)]
            if np.mean(std_per_channel) > 30:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _merge_overlapping_regions(self, regions: List[Dict], 
                                  image: np.ndarray, 
                                  overlap_threshold: float = 0.5) -> List[Dict]:
        """Merge các region trùng lặp"""
        if not regions:
            return []
        
        # Sort theo confidence
        regions = sorted(regions, key=lambda r: r['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
            
            bbox1 = region1['bbox']
            x1, y1, w1, h1 = bbox1
            
            # Tìm các region overlap
            overlapping = [region1]
            
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                bbox2 = region2['bbox']
                x2, y2, w2, h2 = bbox2
                
                # Tính IoU (Intersection over Union)
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > overlap_threshold:
                    overlapping.append(region2)
                    used.add(j)
            
            # Merge tất cả overlapping regions
            if len(overlapping) > 1:
                merged_bbox = self._merge_bboxes([r['bbox'] for r in overlapping])
                merged_confidence = max(r['confidence'] for r in overlapping)
            else:
                merged_bbox = bbox1
                merged_confidence = region1['confidence']
            
            merged.append({
                'bbox': merged_bbox,
                'confidence': merged_confidence,
                'method': 'merged' if len(overlapping) > 1 else region1.get('method', 'unknown')
            })
            
            used.add(i)
        
        return merged
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Tính Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Tọa độ intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        # Diện tích intersection
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Diện tích union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _merge_bboxes(self, bboxes: List[Tuple]) -> Tuple:
        """Merge nhiều bounding boxes thành 1"""
        x_min = min(bbox[0] for bbox in bboxes)
        y_min = min(bbox[1] for bbox in bboxes)
        x_max = max(bbox[0] + bbox[2] for bbox in bboxes)
        y_max = max(bbox[1] + bbox[3] for bbox in bboxes)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _is_valid_image_region(self, region: Dict, image: np.ndarray) -> bool:
        """Kiểm tra region có phải ảnh thực sự không"""
        x, y, w, h = region['bbox']
        
        # Kiểm tra kích thước
        if w < self.min_width or h < self.min_height:
            return False
        
        # Không quá lớn (không phải toàn trang)
        img_h, img_w = image.shape[:2]
        if w > img_w * 0.95 and h > img_h * 0.95:
            return False
        
        # Confidence tối thiểu
        if region['confidence'] < 0.4:
            return False
        
        # Aspect ratio hợp lý
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            return False
        
        return True
    
    def process_scanned_pdf(self, pdf_path: str, output_dir: str, 
                          dpi: int = 300, visualize: bool = True) -> Dict:
        """
        Xử lý PDF scan hoàn chỉnh
        
        Returns:
            Dict với thống kê và danh sách images
        """
        print(f"\n📄 Xử lý PDF scan: {os.path.basename(pdf_path)}")
        
        # Kiểm tra có phải PDF scan không
        is_scanned = self.is_scanned_pdf(pdf_path)
        print(f"   Loại: {'PDF Scan' if is_scanned else 'PDF Digital'}")
        
        # Convert PDF sang images
        print(f"   🔄 Chuyển đổi PDF (DPI: {dpi})...")
        pdf_images = convert_from_path(pdf_path, dpi=dpi)
        
        # Xử lý từng trang
        all_results = []
        total_images = 0
        
        for page_num, page_img in enumerate(pdf_images, 1):
            print(f"   📑 Trang {page_num}/{len(pdf_images)}", end='\r')
            
            img_array = np.array(page_img)
            
            # Phát hiện vùng ảnh
            detected_regions = self.detect_image_regions_advanced(img_array)
            
            # Lưu thông tin
            page_result = {
                'page': page_num,
                'regions': detected_regions,
                'count': len(detected_regions)
            }
            all_results.append(page_result)
            total_images += len(detected_regions)
            
            # Extract và lưu từng ảnh
            for idx, region in enumerate(detected_regions, 1):
                x, y, w, h = region['bbox']
                extracted_img = img_array[y:y+h, x:x+w]
                
                # Lưu ảnh
                img_filename = f"page_{page_num:03d}_img_{idx:02d}.png"
                img_path = os.path.join(output_dir, 'extracted_images', img_filename)
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
                img_bgr = cv2.cvtColor(extracted_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, img_bgr)
                
                # Thêm thông tin file vào region
                region['filename'] = img_filename
            
            # Visualization
            if visualize and detected_regions:
                vis_img = img_array.copy()
                for region in detected_regions:
                    x, y, w, h = region['bbox']
                    conf = region['confidence']
                    
                    # Vẽ rectangle
                    cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Label
                    label = f"{conf:.2f}"
                    cv2.putText(vis_img, label, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Lưu visualization
                vis_path = os.path.join(output_dir, 'bbox_visualizations', 
                                       f'page_{page_num:03d}_bbox.png')
                os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(vis_path, vis_bgr)
        
        print(f"\n   ✅ Đã phát hiện {total_images} vùng ảnh")
        
        # Lưu metadata
        metadata = {
            'pdf_file': os.path.basename(pdf_path),
            'is_scanned': is_scanned,
            'total_pages': len(pdf_images),
            'total_images': total_images,
            'dpi': dpi,
            'pages': all_results
        }
        
        metadata_path = os.path.join(output_dir, 'scanned_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return metadata


def main():
    """Demo sử dụng"""
    
    print("=" * 70)
    print("🔍 PDF SCAN - PHÁT HIỆN VÙNG ẢNH")
    print("=" * 70)
    
    # Cấu hình
    pdf_file = "input/scanned_document.pdf"
    output_dir = "output/scanned_test"
    
    if not os.path.exists(pdf_file):
        print(f"\n❌ Không tìm thấy: {pdf_file}")
        print("💡 Đặt file PDF scan vào thư mục input/")
        input("\nNhấn Enter để thoát...")
        return
    
    try:
        # Khởi tạo detector
        detector = ScannedPDFImageDetector(
            min_width=150,   # Điều chỉnh theo nhu cầu
            min_height=150
        )
        
        # Xử lý PDF
        result = detector.process_scanned_pdf(
            pdf_path=pdf_file,
            output_dir=output_dir,
            dpi=300,
            visualize=True
        )
        
        # Thống kê
        print("\n" + "=" * 70)
        print("📊 KẾT QUẢ")
        print("=" * 70)
        print(f"\n✅ Tổng số ảnh: {result['total_images']}")
        print(f"📄 Tổng số trang: {result['total_pages']}")
        print(f"\n📑 Chi tiết:")
        
        for page_data in result['pages']:
            if page_data['count'] > 0:
                print(f"   Trang {page_data['page']}: {page_data['count']} ảnh")
        
        print(f"\n📂 Kết quả:")
        print(f"   - Ảnh: {output_dir}/extracted_images/")
        print(f"   - Visualization: {output_dir}/bbox_visualizations/")
        print(f"   - Metadata: {output_dir}/scanned_metadata.json")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n\nNhấn Enter để thoát...")


if __name__ == "__main__":
    main()