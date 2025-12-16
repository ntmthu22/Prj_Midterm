"""
Batch Document Processor - Xử lý nhiều file PDF cùng lúc
Tự động xử lý tất cả file PDF trong thư mục và xuất ra Word
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Import DocumentProcessor từ file document_processor.py
try:
    from document_processor import EnhancedDocumentProcessor, VietnameseSpellChecker
    print("✅ Import DocumentProcessor thành công")
except ImportError as e:
    print(f"❌ Lỗi: Không tìm thấy file document_processor.py")
    print(f"   Vui lòng đảm bảo file document_processor.py nằm cùng thư mục")
    sys.exit(1)


class BatchProcessor:
    """Xử lý batch nhiều file PDF"""
    
    def __init__(self, input_folder='input_pdfs', output_folder='output_batch'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Tạo thư mục nếu chưa có
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        self.results = []
        
    def find_pdf_files(self):
        """Tìm tất cả file PDF trong thư mục input"""
        pdf_files = []
        
        # Tìm trong thư mục input
        for file in os.listdir(self.input_folder):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.input_folder, file))
        
        # Nếu không có file trong thư mục input, tìm trong thư mục hiện tại
        if not pdf_files:
            print(f"⚠️  Không tìm thấy file PDF trong thư mục '{self.input_folder}'")
            print(f"   Đang tìm trong thư mục hiện tại...")
            
            for file in os.listdir('.'):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(file)
        
        return pdf_files
    
    def process_single_file(self, pdf_path, index, total):
        """Xử lý một file PDF"""
        filename = os.path.basename(pdf_path)
        
        print("\n" + "="*80)
        print(f"📄 ĐANG XỬ LÝ FILE {index}/{total}: {filename}")
        print("="*80)
        
        try:
            start_time = datetime.now()
            
            # Tạo thư mục output riêng cho mỗi file
            file_output_dir = os.path.join(
                self.output_folder, 
                Path(filename).stem
            )
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Khởi tạo processor
            processor = EnhancedDocumentProcessor(output_dir=file_output_dir)
            
            # Xử lý PDF
            print(f"⏳ Đang phân tích và OCR file...")
            results = processor.process_pdf(pdf_path)
            
            # Xuất Word
            output_filename = f"{Path(filename).stem}_processed.docx"
            print(f"⏳ Đang xuất file Word...")
            output_path = processor.export_to_word(output_filename)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Lưu kết quả
            result = {
                'filename': filename,
                'status': 'success',
                'output_path': output_path,
                'duration': duration,
                'stats': results['metadata']
            }
            
            print(f"\n✅ HOÀN THÀNH: {filename}")
            print(f"   ⏱️  Thời gian: {duration:.1f}s")
            print(f"   📊 Trang: {results['metadata']['total_pages']}")
            print(f"   📝 Phần: {results['metadata']['total_sections']}")
            print(f"   🖼️  Hình: {results['metadata']['total_images']}")
            print(f"   💾 Output: {output_path}")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            
            print(f"\n❌ LỖI KHI XỬ LÝ: {filename}")
            print(f"   Lỗi: {error_msg}")
            
            result = {
                'filename': filename,
                'status': 'error',
                'error': error_msg,
                'traceback': traceback_str,
                'duration': duration
            }
            
            return result
    
    def process_all(self):
        """Xử lý tất cả file PDF"""
        print("\n" + "="*80)
        print("🚀 BATCH DOCUMENT PROCESSOR - XỬ LÝ NHIỀU FILE PDF")
        print("="*80)
        
        # Tìm file PDF
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            print(f"\n❌ Không tìm thấy file PDF nào!")
            print(f"\nHướng dẫn:")
            print(f"1. Tạo thư mục '{self.input_folder}' trong thư mục hiện tại")
            print(f"2. Đặt các file PDF vào thư mục '{self.input_folder}'")
            print(f"3. Chạy lại script này")
            print(f"\nHoặc đặt file PDF trực tiếp trong thư mục hiện tại")
            return
        
        print(f"\n📋 Tìm thấy {len(pdf_files)} file PDF:")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"   {i}. {os.path.basename(pdf_file)}")
        
        # Xác nhận
        print(f"\n📁 Kết quả sẽ được lưu trong: {self.output_folder}/")
        confirm = input("\n▶️  Bắt đầu xử lý? (y/n): ")
        
        if confirm.lower() != 'y':
            print("❌ Đã hủy")
            return
        
        # Xử lý từng file
        start_time = datetime.now()
        
        for i, pdf_path in enumerate(pdf_files, 1):
            result = self.process_single_file(pdf_path, i, len(pdf_files))
            self.results.append(result)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Tổng kết
        self.print_summary(total_duration)
        
        # Lưu log
        self.save_log()
    
    def print_summary(self, total_duration):
        """In tổng kết kết quả"""
        print("\n" + "="*80)
        print("📊 TỔNG KẾT KẾT QUẢ")
        print("="*80)
        
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        error_count = sum(1 for r in self.results if r['status'] == 'error')
        
        print(f"\n✅ Thành công: {success_count}/{len(self.results)} file")
        print(f"❌ Lỗi: {error_count}/{len(self.results)} file")
        print(f"⏱️  Tổng thời gian: {total_duration:.1f}s")
        print(f"📁 Thư mục kết quả: {self.output_folder}/")
        
        if success_count > 0:
            print(f"\n✅ CÁC FILE THÀNH CÔNG:")
            for result in self.results:
                if result['status'] == 'success':
                    print(f"   • {result['filename']}")
                    print(f"     → {result['output_path']}")
        
        if error_count > 0:
            print(f"\n❌ CÁC FILE BỊ LỖI:")
            for result in self.results:
                if result['status'] == 'error':
                    print(f"   • {result['filename']}")
                    print(f"     Lỗi: {result['error']}")
        
        print("\n" + "="*80)
    
    def save_log(self):
        """Lưu log chi tiết"""
        log_file = os.path.join(self.output_folder, 'processing_log.txt')
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BATCH PROCESSING LOG\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for result in self.results:
                f.write(f"\nFile: {result['filename']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Duration: {result['duration']:.1f}s\n")
                
                if result['status'] == 'success':
                    f.write(f"Output: {result['output_path']}\n")
                    f.write(f"Stats:\n")
                    for key, value in result['stats'].items():
                        f.write(f"  - {key}: {value}\n")
                else:
                    f.write(f"Error: {result['error']}\n")
                    f.write(f"Traceback:\n{result['traceback']}\n")
                
                f.write("-"*80 + "\n")
        
        print(f"\n📝 Log đã được lưu: {log_file}")


def main():
    """Hàm main"""
    
    # Kiểm tra file document_processor.py
    if not os.path.exists('document_processor.py'):
        print("❌ Lỗi: Không tìm thấy file 'document_processor.py'")
        print("   Vui lòng đảm bảo cả 2 file nằm cùng thư mục:")
        print("   - document_processor.py")
        print("   - batch_processor.py")
        return
    
    # Khởi tạo batch processor
    processor = BatchProcessor(
        input_folder='input_pdfs',  # Thư mục chứa file PDF cần xử lý
        output_folder='output_batch'  # Thư mục lưu kết quả
    )
    
    # Xử lý tất cả file
    processor.process_all()


if __name__ == "__main__":
    main()