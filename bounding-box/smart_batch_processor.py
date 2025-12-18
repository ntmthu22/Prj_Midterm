"""
SMART BATCH PROCESSOR
Tự động phát hiện PDF digital hoặc scan và xử lý phù hợp
"""

import os
from pathlib import Path
import json
from datetime import datetime
import fitz  # PyMuPDF

# Import các class đã tạo
try:
    from image_extractor import PDFImageExtractor
    from scanned_pdf_processor import ScannedPDFImageDetector
except ImportError:
    print("⚠️  Cần có file image_extractor.py và scanned_pdf_processor.py")
    print("   Lưu các artifact trước đó vào cùng thư mục!")


class SmartPDFProcessor:
    """Tự động xử lý cả PDF digital và scan"""
    
    def __init__(self, min_size_digital=(100, 100), min_size_scanned=(150, 150)):
        self.digital_extractor = PDFImageExtractor(
            min_width=min_size_digital[0],
            min_height=min_size_digital[1]
        )
        self.scanned_detector = ScannedPDFImageDetector(
            min_width=min_size_scanned[0],
            min_height=min_size_scanned[1]
        )
    
    def detect_pdf_type(self, pdf_path: str) -> str:
        """
        Phát hiện loại PDF: 'digital' hoặc 'scanned'
        
        Returns:
            'digital': PDF có text layer và images embedded
            'scanned': PDF scan (mỗi trang là 1 ảnh lớn)
            'mixed': Có cả 2 loại trang
        """
        try:
            doc = fitz.open(pdf_path)
            
            digital_pages = 0
            scanned_pages = 0
            total_pages = len(doc)
            
            # Kiểm tra vài trang đại diện
            pages_to_check = min(5, total_pages)
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                
                # Kiểm tra text
                text = page.get_text().strip()
                has_text = len(text) > 50  # Có ít nhất 50 ký tự text
                
                # Kiểm tra images
                images = page.get_images(full=True)
                
                if not images:
                    # Không có ảnh = digital với text
                    digital_pages += 1
                elif len(images) == 1:
                    # 1 ảnh duy nhất - kiểm tra kích thước
                    xref = images[0][0]
                    try:
                        base_image = doc.extract_image(xref)
                        img_width = base_image["width"]
                        img_height = base_image["height"]
                        
                        page_rect = page.rect
                        page_width = page_rect.width
                        page_height = page_rect.height
                        
                        # Ảnh chiếm >90% trang = scan
                        coverage = (img_width * img_height) / (page_width * page_height)
                        
                        if coverage > 0.9 and not has_text:
                            scanned_pages += 1
                        else:
                            digital_pages += 1
                    except:
                        digital_pages += 1
                else:
                    # Nhiều ảnh = digital
                    digital_pages += 1
            
            doc.close()
            
            # Quyết định loại PDF
            if scanned_pages > digital_pages:
                return 'scanned'
            elif digital_pages > scanned_pages:
                return 'digital'
            else:
                return 'mixed'
                
        except Exception as e:
            print(f"⚠️  Lỗi phát hiện loại PDF: {e}")
            return 'unknown'
    
    def process_pdf(self, pdf_path: str, output_dir: str, 
                   pdf_type: str = None, dpi: int = 300) -> dict:
        """
        Xử lý PDF với phương pháp phù hợp
        
        Args:
            pdf_path: Đường dẫn PDF
            output_dir: Thư mục output
            pdf_type: 'digital', 'scanned', hoặc None (tự động phát hiện)
            dpi: Độ phân giải cho scanned PDF
        """
        pdf_name = os.path.basename(pdf_path)
        
        # Tự động phát hiện nếu chưa biết
        if pdf_type is None:
            print(f"🔍 Đang phát hiện loại PDF...")
            pdf_type = self.detect_pdf_type(pdf_path)
            print(f"   📋 Loại: {pdf_type.upper()}")
        
        # Xử lý theo loại
        if pdf_type == 'digital':
            return self._process_digital(pdf_path, output_dir)
        elif pdf_type == 'scanned':
            return self._process_scanned(pdf_path, output_dir, dpi)
        elif pdf_type == 'mixed':
            return self._process_mixed(pdf_path, output_dir, dpi)
        else:
            raise ValueError(f"Không xác định được loại PDF: {pdf_name}")
    
    def _process_digital(self, pdf_path: str, output_dir: str) -> dict:
        """Xử lý PDF digital (có embedded images)"""
        print(f"   📄 Xử lý bằng phương pháp DIGITAL...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract images bằng PyMuPDF
        extracted = self.digital_extractor.extract_images_pymupdf(pdf_path)
        
        if not extracted:
            print(f"   ⚠️  Không tìm thấy embedded images, thử phương pháp SCAN...")
            return self._process_scanned(pdf_path, output_dir, dpi=300)
        
        # Lưu kết quả
        metadata = self.digital_extractor.save_extracted_images(
            extracted, output_dir, draw_bbox=True, save_metadata=True
        )
        
        # Visualization
        self.digital_extractor.visualize_bboxes_on_pdf(
            pdf_path, extracted, output_dir, dpi=150
        )
        
        return {
            'method': 'digital',
            'total_images': len(extracted),
            'metadata': metadata
        }
    
    def _process_scanned(self, pdf_path: str, output_dir: str, dpi: int) -> dict:
        """Xử lý PDF scan (detect vùng ảnh trong trang)"""
        print(f"   🖼️  Xử lý bằng phương pháp SCAN...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Phát hiện vùng ảnh
        result = self.scanned_detector.process_scanned_pdf(
            pdf_path, output_dir, dpi=dpi, visualize=True
        )
        
        return {
            'method': 'scanned',
            'total_images': result['total_images'],
            'metadata': result
        }
    
    def _process_mixed(self, pdf_path: str, output_dir: str, dpi: int) -> dict:
        """Xử lý PDF mixed (thử cả 2 phương pháp)"""
        print(f"   🔀 PDF hỗn hợp, thử cả 2 phương pháp...")
        
        # Thử digital trước
        try:
            result_digital = self._process_digital(pdf_path, 
                                                  os.path.join(output_dir, 'digital'))
            images_digital = result_digital['total_images']
        except Exception as e:
            print(f"      ⚠️  Digital failed: {e}")
            images_digital = 0
        
        # Thử scanned
        try:
            result_scanned = self._process_scanned(pdf_path, 
                                                  os.path.join(output_dir, 'scanned'), 
                                                  dpi)
            images_scanned = result_scanned['total_images']
        except Exception as e:
            print(f"      ⚠️  Scanned failed: {e}")
            images_scanned = 0
        
        # Chọn phương pháp tốt hơn
        if images_digital > images_scanned:
            print(f"   ✓ Sử dụng kết quả DIGITAL ({images_digital} ảnh)")
            return result_digital
        else:
            print(f"   ✓ Sử dụng kết quả SCAN ({images_scanned} ảnh)")
            return result_scanned


def batch_process_smart(input_dir='input', output_base_dir='output', dpi=300):
    """
    Batch processing thông minh cho tất cả PDF
    """
    
    print("=" * 80)
    print("🚀 SMART BATCH PROCESSOR - TỰ ĐỘNG PHÁT HIỆN LOẠI PDF")
    print("=" * 80)
    print()
    
    # Danh sách file
    PDF_FILES = {
        '500_bai_thuoc.pdf': {'owner': 'Thư', 'pages': 250},
        'ban_thao_van_dap.pdf': {'owner': 'Chị Ngọc', 'pages': 83},
        'so_tay_thuoc_nam.pdf': {'owner': 'Chị Ngọc', 'pages': 179},
        'benh_ngoai_khoa.pdf': {'owner': 'Anh Hiếu', 'pages': 148},
        'dong_y.pdf': {'owner': 'Anh Hiếu', 'pages': 139}
    }
    
    # Khởi tạo processor
    processor = SmartPDFProcessor(
        min_size_digital=(100, 100),
        min_size_scanned=(150, 150)
    )
    
    # Thống kê
    stats = {
        'total_pdfs': len(PDF_FILES),
        'processed': 0,
        'failed': 0,
        'total_images': 0,
        'by_type': {'digital': 0, 'scanned': 0, 'mixed': 0},
        'results': []
    }
    
    start_time = datetime.now()
    
    # Xử lý từng file
    for idx, (filename, info) in enumerate(PDF_FILES.items(), 1):
        pdf_path = os.path.join(input_dir, filename)
        
        print("\n" + "=" * 80)
        print(f"📄 [{idx}/{len(PDF_FILES)}] {filename}")
        print(f"   Người xử lý: {info['owner']} | Số trang: {info['pages']}")
        print("=" * 80)
        
        if not os.path.exists(pdf_path):
            print(f"❌ Không tìm thấy file!")
            stats['failed'] += 1
            stats['results'].append({
                'filename': filename,
                'status': 'FAILED',
                'reason': 'File not found'
            })
            continue
        
        try:
            file_start = datetime.now()
            
            # Tạo output dir
            output_dir = os.path.join(output_base_dir, Path(filename).stem)
            
            # Xử lý
            result = processor.process_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                pdf_type=None,  # Tự động phát hiện
                dpi=dpi
            )
            
            file_end = datetime.now()
            processing_time = (file_end - file_start).total_seconds()
            
            # Thống kê
            method = result['method']
            images_count = result['total_images']
            
            stats['processed'] += 1
            stats['total_images'] += images_count
            stats['by_type'][method] = stats['by_type'].get(method, 0) + 1
            
            print(f"\n   ✅ Hoàn thành:")
            print(f"      - Phương pháp: {method.upper()}")
            print(f"      - Số ảnh: {images_count}")
            print(f"      - Thời gian: {processing_time:.1f}s")
            print(f"      - Output: {output_dir}")
            
            stats['results'].append({
                'filename': filename,
                'status': 'SUCCESS',
                'method': method,
                'images_count': images_count,
                'processing_time': processing_time,
                'output_dir': output_dir
            })
            
        except Exception as e:
            print(f"\n   ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
            
            stats['failed'] += 1
            stats['results'].append({
                'filename': filename,
                'status': 'FAILED',
                'reason': str(e)
            })
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Tổng kết
    print("\n\n" + "=" * 80)
    print("📊 TỔNG KẾT")
    print("=" * 80)
    
    print(f"\n✅ Thành công: {stats['processed']}/{stats['total_pdfs']}")
    print(f"❌ Thất bại: {stats['failed']}/{stats['total_pdfs']}")
    print(f"🖼️  Tổng số ảnh: {stats['total_images']}")
    print(f"⏱️  Tổng thời gian: {total_time:.1f}s ({total_time/60:.1f} phút)")
    
    print(f"\n📋 Phân loại:")
    print(f"   - Digital PDF: {stats['by_type']['digital']} files")
    print(f"   - Scanned PDF: {stats['by_type']['scanned']} files")
    print(f"   - Mixed PDF: {stats['by_type']['mixed']} files")
    
    print(f"\n📑 Chi tiết từng file:")
    for result in stats['results']:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"\n{status_icon} {result['filename']}")
        
        if result['status'] == 'SUCCESS':
            print(f"   - Phương pháp: {result['method'].upper()}")
            print(f"   - Số ảnh: {result['images_count']}")
            print(f"   - Thời gian: {result['processing_time']:.1f}s")
            print(f"   - Output: {result['output_dir']}")
        else:
            print(f"   - Lỗi: {result.get('reason', 'Unknown')}")
    
    # Lưu báo cáo
    report_file = os.path.join(output_base_dir, 'smart_processing_report.json')
    stats['start_time'] = start_time.isoformat()
    stats['end_time'] = end_time.isoformat()
    stats['total_time'] = total_time
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 Báo cáo: {report_file}")
    
    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH!")
    print("=" * 80)
    
    return stats


if __name__ == "__main__":
    print("\n🚀 SMART BATCH PROCESSING\n")
    
    INPUT_DIR = 'input'
    OUTPUT_DIR = 'output'
    DPI = 300  # Cho scanned PDF
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        stats = batch_process_smart(
            input_dir=INPUT_DIR,
            output_base_dir=OUTPUT_DIR,
            dpi=DPI
        )
        
        print(f"\n✅ Tất cả kết quả trong: {OUTPUT_DIR}")
        print(f"\n💡 GỢI Ý:")
        print(f"   - Digital PDF: Ảnh rõ nét, bbox chính xác")
        print(f"   - Scanned PDF: Có thể cần điều chỉnh min_size")
        print(f"   - Review bbox_visualizations/ để kiểm tra")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n\nNhấn Enter để thoát...")