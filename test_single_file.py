"""
Test xử lý 1 file PDF đơn
"""

import os
from document_processor import EnhancedDocumentProcessor
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


def auto_correct_spelling(results, spell_checker):
    """Tự động sửa lỗi chính tả"""
    
    # Lấy toàn bộ text
    full_text = ""
    for section in results['sections']:
        full_text += section['content'] + " "
    
    # Tạo từ điển thay thế
    replacements = {}
    for error in results['spelling_check']['errors']:
        word = error['word']
        suggestions = error['suggestions']
        
        # Chọn gợi ý đầu tiên nếu có
        if suggestions and len(suggestions) > 0:
            replacements[word] = suggestions[0]
    
    # Thay thế từng từ
    corrected_text = full_text
    for wrong_word, correct_word in replacements.items():
        # Thay thế với boundaries (tránh thay thế một phần của từ khác)
        import re
        pattern = r'\b' + re.escape(wrong_word) + r'\b'
        corrected_text = re.sub(pattern, correct_word, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text


def export_corrected_word(corrected_text, output_dir, filename, results):
    """Xuất văn bản đã sửa ra Word"""
    
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(13)
    
    # Tiêu đề
    title = doc.add_heading('TÀI LIỆU ĐÃ SỬA LỖI CHÍNH TẢ', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Thông tin
    doc.add_heading('Thông tin', 1)
    info_text = f"""
Số lỗi đã sửa: {len(results['spelling_check']['errors'])}
Tổng số từ: {results['metadata']['total_words']}
Tỷ lệ lỗi ban đầu: {results['spelling_check']['error_rate']:.2%}
    """
    doc.add_paragraph(info_text)
    
    # Nội dung đã sửa
    doc.add_heading('Nội dung đã sửa', 1)
    
    # Chia thành đoạn
    paragraphs = corrected_text.split('\n\n')
    for para in paragraphs:
        if para.strip():
            p = doc.add_paragraph(para.strip())
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    # Lưu file
    output_path = os.path.join(output_dir, filename)
    doc.save(output_path)
    return output_path


def main():
    """Test với 1 file"""
    
    print("="*80)
    print("🧪 TEST XỬ LÝ 1 FILE PDF")
    print("="*80)
    
    # CHỌN FILE CẦN TEST (thay đổi tên file ở đây)
    pdf_file = "input_pdfs/ban_thao_van_dap.pdf"  # File nhỏ nhất - 83 trang
    
    # Hoặc chọn file khác:
    # pdf_file = "input_pdfs/dong_y_nhap_mon.pdf"  # 139 trang
    # pdf_file = "input_pdfs/500_bai_thuoc_hay_chua_benh_cao_huyet_ap.pdf"  # 250 trang
    
    # Kiểm tra file tồn tại
    if not os.path.exists(pdf_file):
        print(f"❌ Không tìm thấy file: {pdf_file}")
        print("\n📝 Vui lòng:")
        print("1. Kiểm tra tên file đúng chưa")
        print("2. Kiểm tra file có trong thư mục input_pdfs/ không")
        return
    
    print(f"\n📄 File test: {os.path.basename(pdf_file)}")
    
    # Tạo thư mục output
    output_dir = "output_test_single"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Kết quả sẽ lưu trong: {output_dir}/")
    
    # Xác nhận
    confirm = input("\n▶️  Bắt đầu xử lý? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Đã hủy")
        return
    
    try:
        # Khởi tạo processor
        processor = EnhancedDocumentProcessor(output_dir=output_dir)
        
        # Xử lý PDF (debug=False để không lưu ảnh debug, chạy nhanh hơn)
        print("\n⏳ Đang xử lý...")
        results = processor.process_pdf(pdf_file, debug=False)
        
        # Xuất Word
        output_filename = f"{os.path.basename(pdf_file).replace('.pdf', '')}_processed.docx"
        output_path = processor.export_to_word(output_filename)
        
        # SỬA LỖI CHÍNH TẢ
        print("\n" + "="*80)
        print("🔧 SỬA LỖI CHÍNH TẢ")
        print("="*80)
        
        if results['spelling_check']['errors']:
            print(f"\n📝 Tìm thấy {len(results['spelling_check']['errors'])} lỗi chính tả")
            
            # Hỏi người dùng có muốn sửa không
            fix_spelling = input("\n▶️  Bạn có muốn tự động sửa lỗi chính tả? (y/n): ")
            
            if fix_spelling.lower() == 'y':
                corrected_text = auto_correct_spelling(results, processor.spell_checker)
                
                # Lưu văn bản đã sửa
                corrected_file = os.path.join(output_dir, "ocr_result_corrected.txt")
                with open(corrected_file, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"KẾT QUẢ OCR ĐÃ SỬA LỖI: {os.path.basename(pdf_file)}\n")
                    f.write("="*80 + "\n\n")
                    f.write(corrected_text)
                
                print(f"\n✅ Đã sửa {len(results['spelling_check']['errors'])} lỗi")
                print(f"📄 File đã sửa: {corrected_file}")
                
                # Xuất Word đã sửa lỗi
                corrected_word_file = f"{os.path.basename(pdf_file).replace('.pdf', '')}_corrected.docx"
                export_corrected_word(corrected_text, output_dir, corrected_word_file, results)
                print(f"💾 File Word đã sửa: {os.path.join(output_dir, corrected_word_file)}")
            else:
                print("⏭️  Bỏ qua sửa lỗi chính tả")
        else:
            print("\n✅ Không có lỗi chính tả!")
        
        # Tổng kết
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH!")
        print("="*80)
        print(f"\n📊 Thống kê:")
        print(f"   📄 Tổng số trang: {results['metadata']['total_pages']}")
        print(f"   📝 Tổng số phần: {results['metadata']['total_sections']}")
        print(f"   🖼️  Tổng số hình: {results['metadata']['total_images']}")
        print(f"   📖 Tổng số từ: {results['metadata']['total_words']}")
        print(f"   ❌ Lỗi chính tả: {results['metadata']['spelling_errors']}")
        
        # Hiển thị preview nội dung OCR
        print(f"\n📝 PREVIEW NỘI DUNG OCR (5 phần đầu):")
        print("-"*80)
        for i, section in enumerate(results['sections'][:5], 1):
            content_preview = section['content'][:200].replace('\n', ' ')
            print(f"\n{i}. Trang {section['page']}:")
            print(f"   {content_preview}...")
        
        # Hiển thị danh sách hình ảnh
        print(f"\n🖼️  DANH SÁCH HÌNH ẢNH ({len(results['images'])} ảnh):")
        print("-"*80)
        for i, img in enumerate(results['images'][:10], 1):  # Hiển thị 10 ảnh đầu
            print(f"{i}. {img['filename']}")
            print(f"   Trang: {img['page']} | Format: {img['format']} | Path: {img['path']}")
        
        if len(results['images']) > 10:
            print(f"   ... và {len(results['images']) - 10} hình ảnh khác")
        
        # Lưu kết quả OCR ra file text
        ocr_text_file = os.path.join(output_dir, "ocr_result.txt")
        with open(ocr_text_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"KẾT QUẢ OCR: {os.path.basename(pdf_file)}\n")
            f.write("="*80 + "\n\n")
            
            for section in results['sections']:
                f.write(f"\n{'='*60}\n")
                f.write(f"Trang {section['page']} - Section {section['id']}\n")
                f.write(f"{'='*60}\n")
                f.write(section['content'])
                f.write("\n\n")
        
        print(f"\n📄 File text OCR: {ocr_text_file}")
        
        # Lưu danh sách hình ảnh ra file
        images_list_file = os.path.join(output_dir, "images_list.txt")
        with open(images_list_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"DANH SÁCH HÌNH ẢNH: {os.path.basename(pdf_file)}\n")
            f.write(f"Tổng số: {len(results['images'])} hình ảnh\n")
            f.write("="*80 + "\n\n")
            
            for i, img in enumerate(results['images'], 1):
                f.write(f"{i}. {img['filename']}\n")
                f.write(f"   ID: {img['id']}\n")
                f.write(f"   Trang: {img['page']}\n")
                f.write(f"   Format: {img['format']}\n")
                f.write(f"   Path: {img['path']}\n")
                f.write(f"   Bbox: x={img['bbox']['x']}, y={img['bbox']['y']}, ")
                f.write(f"w={img['bbox']['width']}, h={img['bbox']['height']}\n")
                f.write("\n")
        
        print(f"🖼️  File danh sách ảnh: {images_list_file}")
        
        # Lưu thống kê chi tiết
        stats_file = os.path.join(output_dir, "statistics.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"THỐNG KÊ CHI TIẾT: {os.path.basename(pdf_file)}\n")
            f.write("="*80 + "\n\n")
            
            f.write("TỔNG QUAN:\n")
            f.write(f"  Tổng số trang: {results['metadata']['total_pages']}\n")
            f.write(f"  Tổng số phần: {results['metadata']['total_sections']}\n")
            f.write(f"  Tổng số hình ảnh: {results['metadata']['total_images']}\n")
            f.write(f"  Tổng số từ: {results['metadata']['total_words']}\n")
            f.write(f"  Lỗi chính tả: {results['metadata']['spelling_errors']}\n")
            f.write(f"  Tỷ lệ lỗi: {results['spelling_check']['error_rate']:.2%}\n\n")
            
            if results['spelling_check']['errors']:
                f.write("LỖI CHÍNH TẢ (20 lỗi đầu):\n")
                for i, error in enumerate(results['spelling_check']['errors'][:20], 1):
                    f.write(f"  {i}. '{error['word']}' (vị trí: {error['position']})\n")
                    if error['suggestions']:
                        f.write(f"     Gợi ý: {', '.join(error['suggestions'])}\n")
        
        print(f"📊 File thống kê: {stats_file}")
        
        print(f"\n💾 File Word: {output_path}")
        print(f"📁 Thư mục hình ảnh: {output_dir}/images/")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()