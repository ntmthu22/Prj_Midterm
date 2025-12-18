from paddleocr import PaddleOCR
import fitz
from PIL import Image
import numpy as np

# Init OCR
ocr = PaddleOCR(use_textline_orientation=True, lang='vi')

# Mở PDF
pdf_path = "input_pdfs/ban_thao_van_dap.pdf"  # File nhỏ nhất
doc = fitz.open(pdf_path)
page = doc[0]

# Chuyển sang image
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img_array = np.array(img)

print("Đang OCR...")

# OCR TRỰC TIẾP - KHÔNG có preprocessing
result = ocr.predict(img_array)

print(f"Type: {type(result)}")
print(f"Length: {len(result)}")

if result and len(result) > 0:
    r = result[0]
    print(f"\nKeys: {r.keys()}")
    print(f"\nTexts: {r.get('rec_texts', [])[:5]}")  # 5 dòng đầu
    print(f"Scores: {r.get('rec_scores', [])[:5]}")
    
print("\n✅ Test thành công!")