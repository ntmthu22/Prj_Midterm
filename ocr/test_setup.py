"""
Script kiểm tra các thư viện đã cài đặt đúng chưa
"""

import sys

def test_imports():
    """Kiểm tra tất cả các thư viện cần thiết"""
    
    results = []
    
    # Danh sách thư viện cần kiểm tra
    libraries = {
        'paddle': 'PaddlePaddle',
        'paddleocr': 'PaddleOCR',
        'fitz': 'PyMuPDF',
        'PIL': 'Pillow',
        'docx': 'python-docx',
        'cv2': 'OpenCV',
        'numpy': 'NumPy'
    }
    
    print("="*60)
    print("KIỂM TRA CÀI ĐẶT THƯ VIỆN")
    print("="*60)
    
    all_ok = True
    
    for module_name, display_name in libraries.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'N/A')
            print(f"✅ {display_name:20s} - Version: {version}")
            results.append((display_name, True, version))
        except ImportError as e:
            print(f"❌ {display_name:20s} - CHƯA CÀI ĐẶT")
            results.append((display_name, False, str(e)))
            all_ok = False
    
    print("="*60)
    
    if all_ok:
        print("🎉 TẤT CẢ THƯ VIỆN ĐÃ SẴN SÀNG!")
        print("\nBạn có thể chạy:")
        print("  python document_processor.py")
    else:
        print("⚠️  MỘT SỐ THƯ VIỆN CHƯA ĐƯỢC CÀI ĐẶT")
        print("\nVui lòng cài đặt các thư viện còn thiếu:")
        for name, ok, info in results:
            if not ok:
                print(f"  pip install {name.lower().replace(' ', '-')}")
    
    print("="*60)
    
    return all_ok


def test_paddleocr():
    """Test PaddleOCR cơ bản"""
    print("\n" + "="*60)
    print("TEST PADDLEOCR")
    print("="*60)
    
    try:
        from paddleocr import PaddleOCR
        print("✅ Import PaddleOCR thành công")
        
        # Khởi tạo OCR (sẽ tải model lần đầu)
        print("Đang khởi tạo PaddleOCR (có thể mất vài phút lần đầu)...")
        ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)
        print("✅ Khởi tạo PaddleOCR thành công")
        
        print("\n🎉 PaddleOCR đã sẵn sàng sử dụng!")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi test PaddleOCR: {e}")
        return False


def create_sample_pdf():
    """Tạo file PDF mẫu để test"""
    print("\n" + "="*60)
    print("TẠO FILE PDF MẪU")
    print("="*60)
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # Tạo PDF đơn giản
        pdf_file = "test_sample.pdf"
        c = canvas.Canvas(pdf_file, pagesize=A4)
        
        # Thêm nội dung
        c.setFont("Helvetica-Bold", 24)
        c.drawString(100, 800, "Test Document")
        
        c.setFont("Helvetica", 14)
        c.drawString(100, 750, "2.1 Section One")
        c.drawString(100, 720, "This is the content of section 2.1")
        c.drawString(100, 690, "with some text for testing OCR.")
        
        c.drawString(100, 650, "2.2 Section Two")
        c.drawString(100, 620, "This is the content of section 2.2")
        c.drawString(100, 590, "with more text for testing.")
        
        c.save()
        
        print(f"✅ Đã tạo file PDF mẫu: {pdf_file}")
        print("Bạn có thể dùng file này để test:")
        print(f"  python document_processor.py")
        return True
        
    except ImportError:
        print("⚠️  Cần cài reportlab để tạo PDF mẫu:")
        print("  pip install reportlab")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi tạo PDF: {e}")
        return False


def main():
    """Chạy tất cả các test"""
    print("\n🚀 BẮT ĐẦU KIỂM TRA HỆ THỐNG\n")
    
    # Test 1: Kiểm tra thư viện
    libs_ok = test_imports()
    
    if not libs_ok:
        print("\n⚠️  Vui lòng cài đặt đủ thư viện trước khi tiếp tục")
        return
    
    # Test 2: Test PaddleOCR
    print("\n")
    input("Nhấn Enter để test PaddleOCR (sẽ tải model ~50MB lần đầu)...")
    paddleocr_ok = test_paddleocr()
    
    # Test 3: Tạo PDF mẫu
    print("\n")
    create_pdf = input("Bạn có muốn tạo file PDF mẫu để test? (y/n): ")
    if create_pdf.lower() == 'y':
        create_sample_pdf()
    
    print("\n" + "="*60)
    print("HOÀN TẤT KIỂM TRA")
    print("="*60)
    print("\n📝 HƯỚNG DẪN SỬ DỤNG:")
    print("1. Đặt file PDF cần xử lý vào thư mục hiện tại")
    print("2. Đổi tên thành 'input_document.pdf' hoặc sửa trong code")
    print("3. Chạy: python document_processor.py")
    print("4. Kết quả sẽ ở trong thư mục 'output/'\n")


if __name__ == "__main__":
    main()