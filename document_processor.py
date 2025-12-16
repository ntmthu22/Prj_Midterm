"""
Enhanced Document Processor với OCR cải tiến
- Tăng độ phân giải cao hơn
- Preprocessing image tốt hơn
- Xử lý scan PDF hiệu quả
"""

import os
import cv2
import fitz
import numpy as np
from PIL import Image, ImageEnhance
from paddleocr import PaddleOCR
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path
import re
from typing import List, Dict, Tuple


class ImagePreprocessor:
    """Tiền xử lý hình ảnh trước khi OCR"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Cải thiện chất lượng hình ảnh"""
        
        # Chuyển sang grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Loại bỏ noise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 2. Tăng contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        
        # 3. Thresholding - chuyển sang binary
        # Adaptive threshold tốt cho văn bản scan
        binary = cv2.adaptiveThreshold(
            contrast, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # 4. Morphological operations - làm sạch
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """Xoay thẳng hình ảnh bị nghiêng"""
        try:
            coords = np.column_stack(np.where(image > 0))
            
            # Nếu không có điểm nào (ảnh toàn đen/trắng), return nguyên
            if coords.shape[0] == 0:
                return image
            
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) < 0.5:  # Không cần xoay nếu gần như thẳng
                return image
            
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return rotated
        except Exception as e:
            # Nếu lỗi, return ảnh gốc
            return image
    
    @staticmethod
    def resize_for_ocr(image: np.ndarray, target_height=2000) -> np.ndarray:
        """Resize image về kích thước tối ưu cho OCR"""
        h, w = image.shape[:2]
        
        if h < target_height:
            # Scale up nếu quá nhỏ
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return resized
        elif h > target_height * 1.5:
            # Scale down nếu quá lớn
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized
        
        return image


class VietnameseSpellChecker:
    """Kiểm tra chính tả tiếng Việt"""
    
    def __init__(self):
        self.dictionary = set([
            'và', 'của', 'có', 'trong', 'là', 'được', 'với', 'các', 'cho', 'để',
            'đã', 'này', 'theo', 'những', 'người', 'từ', 'một', 'năm', 'khi', 'về',
            'nội', 'dung', 'hình', 'ảnh', 'đồ', 'thị', 'tài', 'liệu', 'chương', 'mục',
            'phần', 'bảng', 'số', 'liệu', 'thông', 'tin', 'dữ', 'kiểm', 'tra', 'xử', 'lý',
            'phương', 'pháp', 'kết', 'quả', 'nghiên', 'cứu', 'phát', 'triển', 'hệ', 'thống',
        ])
        self.load_extended_dictionary()
    
    def load_extended_dictionary(self, dict_file='vietnamese_dict.txt'):
        if os.path.exists(dict_file):
            with open(dict_file, 'r', encoding='utf-8') as f:
                words = f.read().splitlines()
                self.dictionary.update(words)
    
    def check_text(self, text: str) -> Dict:
        words = re.findall(r'[\wàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+', text.lower())
        errors = []
        
        for idx, word in enumerate(words):
            if len(word) > 2 and word not in self.dictionary:
                errors.append({'word': word, 'position': idx, 'suggestions': []})
        
        return {
            'total_words': len(words),
            'errors': errors,
            'error_rate': len(errors) / len(words) if len(words) > 0 else 0
        }


class EnhancedDocumentProcessor:
    """Document processor với OCR cải tiến"""
    
    def __init__(self, output_dir='output'):
        print("🔧 Đang khởi tạo PaddleOCR...")
        print("   (Lần đầu sẽ tải model ~100MB, chờ vài phút...)")
        
        # Khởi tạo OCR với config mới
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='vi')
        
        self.preprocessor = ImagePreprocessor()
        self.spell_checker = VietnameseSpellChecker()
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/debug", exist_ok=True)
        
        self.results = {
            'sections': [],
            'images': [],
            'spelling_check': {},
            'metadata': {}
        }
        
        print("✅ PaddleOCR sẵn sàng!\n")
    
    def process_page_image(self, image: np.ndarray, page_num: int, debug=False) -> List[Dict]:
        """Xử lý một trang - BỎ preprocessing để tránh lỗi"""
        
        print(f"   🔍 Đang OCR trang {page_num}...")
        
        # OCR trực tiếp - KHÔNG preprocessing
        try:
            ocr_result = self.ocr.predict(image)  # Dùng ảnh gốc luôn
        except Exception as e:
            print(f"   ❌ Lỗi OCR: {e}")
            return []
        
        layout_elements = []
        
        if not ocr_result or len(ocr_result) == 0:
            print(f"   ⚠️  Không phát hiện text trong trang {page_num}")
            return layout_elements
        
        # Parse kết quả mới - result là list of dict
        text_count = 0
        result_dict = ocr_result[0]  # Lấy dict đầu tiên
        
        # Lấy dữ liệu từ dict
        texts = result_dict.get('rec_texts', [])
        scores = result_dict.get('rec_scores', [])
        polys = result_dict.get('rec_polys', [])
        
        print(f"   📊 Phát hiện {len(texts)} dòng text")
        
        for i in range(len(texts)):
            try:
                text = texts[i]
                confidence = scores[i] if i < len(scores) else 1.0
                poly = polys[i] if i < len(polys) else None
                
                # Chỉ lấy text có confidence > 0.5
                if confidence < 0.5:
                    continue
                
                text_count += 1
                
                # Xử lý bounding box
                if poly is not None and len(poly) >= 4:
                    # poly là numpy array shape (4, 2): [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x_coords = [float(p[0]) for p in poly]
                    y_coords = [float(p[1]) for p in poly]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    width = x_max - x_min
                    height = y_max - y_min
                else:
                    x_min, y_min, width, height = 0, 0, 100, 20
                
                element_type = self.classify_element(text, width, height, y_min)
                
                layout_elements.append({
                    'type': element_type,
                    'bbox': {
                        'x': int(x_min),
                        'y': int(y_min),
                        'width': int(width),
                        'height': int(height)
                    },
                    'text': text,
                    'confidence': float(confidence)
                })
                
            except Exception as e:
                print(f"   ⚠️  Bỏ qua dòng lỗi: {e}")
                continue
        
        print(f"   ✅ Phát hiện {text_count} dòng text (confidence > 0.5)")
        return layout_elements
    
    def classify_element(self, text: str, width: float, height: float, y_pos: float) -> str:
        """Phân loại element"""
        if height > 30 and y_pos < 200:
            return 'title'
        if re.match(r'^\d+\.\d+', text.strip()):
            return 'heading'
        return 'text'
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Trích xuất hình ảnh từ PDF"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                img_filename = f"img_{page_num+1:03d}_{img_idx+1:03d}.{image_ext}"
                img_path = os.path.join(self.output_dir, "images", img_filename)
                
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                
                img_rects = page.get_image_rects(xref)
                bbox = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                
                if img_rects:
                    rect = img_rects[0]
                    bbox = {
                        'x': int(rect.x0),
                        'y': int(rect.y0),
                        'width': int(rect.width),
                        'height': int(rect.height)
                    }
                
                images.append({
                    'id': f'img_{page_num+1:03d}_{img_idx+1:03d}',
                    'filename': img_filename,
                    'path': img_path,
                    'page': page_num + 1,
                    'format': image_ext.upper(),
                    'bbox': bbox
                })
        
        doc.close()
        return images
    
    def process_pdf(self, pdf_path: str, debug=False) -> Dict:
        """Xử lý PDF với OCR cải tiến"""
        print(f"\n{'='*80}")
        print(f"📄 ĐANG XỬ LÝ: {os.path.basename(pdf_path)}")
        print(f"{'='*80}\n")
        
        # Trích xuất hình ảnh
        print("🖼️  BƯỚC 1: Trích xuất hình ảnh...")
        self.results['images'] = self.extract_images_from_pdf(pdf_path)
        print(f"✅ Đã trích xuất {len(self.results['images'])} hình ảnh\n")
        
        # OCR
        print("🔍 BƯỚC 2: OCR văn bản...")
        doc = fitz.open(pdf_path)
        
        all_text = ""
        section_counter = 1
        total_pages = len(doc)  # Lưu số trang TRƯỚC khi close
        
        for page_num in range(total_pages):
            print(f"\n📄 Trang {page_num + 1}/{total_pages}")
            
            page = doc[page_num]
            
            # Chuyển page sang image với độ phân giải cao
            zoom = 3  # Tăng độ phân giải lên 3x
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Chuyển sang numpy array
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img)
            
            # Process với preprocessing
            layout_elements = self.process_page_image(img_array, page_num + 1, debug=debug)
            
            # Xử lý kết quả
            current_section = ""
            
            for element in layout_elements:
                element['page'] = page_num + 1
                
                if element['type'] == 'heading':
                    if current_section:
                        self.results['sections'].append({
                            'id': f'section_{section_counter}',
                            'content': current_section.strip() + '\n\n</break>\n',
                            'page': page_num + 1
                        })
                        section_counter += 1
                        current_section = ""
                    
                    current_section = f"\n{element['text']}\n\n"
                else:
                    current_section += element['text'] + " "
                
                all_text += element['text'] + " "
            
            if current_section:
                self.results['sections'].append({
                    'id': f'section_{section_counter}',
                    'content': current_section.strip(),
                    'page': page_num + 1
                })
                section_counter += 1
        
        doc.close()
        
        # Kiểm tra chính tả
        print(f"\n📝 BƯỚC 3: Kiểm tra chính tả...")
        self.results['spelling_check'] = self.spell_checker.check_text(all_text)
        
        # Metadata (dùng total_pages đã lưu)
        self.results['metadata'] = {
            'total_pages': total_pages,
            'total_sections': len(self.results['sections']),
            'total_images': len(self.results['images']),
            'total_words': self.results['spelling_check']['total_words'],
            'spelling_errors': len(self.results['spelling_check']['errors'])
        }
        
        print(f"\n{'='*80}")
        print("✅ HOÀN TẤT XỬ LÝ")
        print(f"{'='*80}")
        print(f"📊 Tổng số trang: {self.results['metadata']['total_pages']}")
        print(f"📝 Tổng số phần: {self.results['metadata']['total_sections']}")
        print(f"🖼️  Tổng số hình: {self.results['metadata']['total_images']}")
        print(f"📖 Tổng số từ: {self.results['metadata']['total_words']}")
        
        return self.results
    
    def export_to_word(self, output_filename='output_document.docx'):
        """Xuất Word"""
        print(f"\n💾 Đang xuất file Word: {output_filename}")
        
        doc = Document()
        style = doc.styles['Normal']
        style.font.name = 'Times New Roman'
        style.font.size = Pt(13)
        
        title = doc.add_heading('TÀI LIỆU ĐÃ XỬ LÝ', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_heading('Thông tin tổng quan', 1)
        metadata_text = f"""
Tổng số trang: {self.results['metadata']['total_pages']}
Tổng số phần: {self.results['metadata']['total_sections']}
Tổng số hình ảnh: {self.results['metadata']['total_images']}
Tổng số từ: {self.results['metadata']['total_words']}
Lỗi chính tả: {self.results['metadata']['spelling_errors']}
        """
        doc.add_paragraph(metadata_text)
        
        doc.add_heading('Nội dung', 1)
        
        for section in self.results['sections']:
            content = section['content']
            
            if re.match(r'^\d+\.\d+', content.strip()):
                lines = content.split('\n', 1)
                heading_text = lines[0].strip()
                doc.add_heading(heading_text, 2)
                
                if len(lines) > 1:
                    body_text = lines[1].replace('</break>', '').strip()
                    if body_text:
                        p = doc.add_paragraph(body_text)
                        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            else:
                text = content.replace('</break>', '').strip()
                if text:
                    p = doc.add_paragraph(text)
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            
            if '</break>' in content:
                doc.add_page_break()
        
        if self.results['images']:
            doc.add_page_break()
            doc.add_heading('Danh sách hình ảnh', 1)
            
            for img in self.results['images']:
                doc.add_heading(f"Hình {img['id']}", 2)
                try:
                    doc.add_picture(img['path'], width=Inches(4))
                except:
                    doc.add_paragraph(f"[Không thể thêm hình ảnh: {img['filename']}]")
        
        output_path = os.path.join(self.output_dir, output_filename)
        doc.save(output_path)
        print(f"✅ Đã lưu file Word: {output_path}\n")
        
        return output_path


def main():
    pdf_path = "input_document.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Không tìm thấy file: {pdf_path}")
        print("📝 Đặt file PDF cần xử lý vào thư mục hiện tại và đổi tên thành 'input_document.pdf'")
        return
    
    processor = EnhancedDocumentProcessor(output_dir='output')
    results = processor.process_pdf(pdf_path, debug=True)  # debug=True để xem ảnh đã xử lý
    processor.export_to_word('tai_lieu_da_xu_ly.docx')


if __name__ == "__main__":
    main()