# Tesseract OCR Pipeline cho Tài Liệu Y Học Cổ Truyền Tiếng Việt

Branch: `feature/vietnamese-post-ocr-correction`

### Mô tả
- Pipeline OCR sử dụng **Tesseract OCR** (phù hợp sách scan cũ, tiếng Việt có dấu).
- Tính năng chính:
  - Tiền xử lý ảnh nâng cao (crop lề, xóa bóng gáy sách, Gaussian blur + Otsu threshold).
  - Phát hiện tự động trang 1 cột / 2 cột và tách cột.
  - Thử nhiều config PSM (6 normal, 4 table, 11 sparse text).
  - Làm sạch text cơ bản (xóa ký tự lạ, normalize dòng).
  - Xuất ra file .docx có phân trang.
- Đây là nền tảng để phát triển **post-OCR error correction** (sửa lỗi tiếng Việt: dấu sai, chữ không dấu, từ y học cổ như "huyet ap" → "huyết áp").

### Output mẫu
Thư mục `ocr_outputs/` chứa kết quả OCR thực tế từ 3 sách:
- `500_bai_thuoc_hay_chua_benh_cao_huyet_ap_ocr.docx`
- `ban_thao_van_dap_ocr.docx`
- `thuoc_nam_chua_benh_tuyen_co_so_ocr.docx`

### Dataset minh họa (PDF scan do giảng viên cung cấp - không public PDF gốc)
**Sổ Tay Thuốc Nam Chữa Bệnh Tuyến Cơ Sở** (NXB Quân Đội Nhân Dân)

<grok-card data-id="03bc0c" data-type="image_card"></grok-card>



<grok-card data-id="a045cc" data-type="image_card"></grok-card>


**500 Bài Thuốc Hay Chữa Bệnh Cao Huyết Áp** (NXB Y Học)

<grok-card data-id="2d1682" data-type="image_card"></grok-card>


**Các Bệnh Ngoại Khoa Thường Gặp Và Cách Chữa Trị Bằng Y Học Cổ Truyền** (NXB Chính Trị Quốc Gia Sự Thật)

<grok-card data-id="93ea5a" data-type="image_card"></grok-card>


**Đông Y Nhập Môn** (Lương y Nguyễn Thiên Quyền, NXB Văn Hóa Dân Tộc)

<grok-card data-id="5ae17c" data-type="image_card"></grok-card>


### Cách chạy
1. Install package chính của repo: `pip install -r ../requirements.txt`
2. Install thêm cho phần Tesseract: `pip install -r requirements_tesseract.txt`
3. Sửa đường dẫn Tesseract/Poppler và danh sách PDF trong `vietnamese_ocr_pipeline.py`
4. Chạy: `python vietnamese_ocr_pipeline.py`

**Lưu ý**: Code dùng cho mục đích học tập, đồ án Vietnamese Text Mining.