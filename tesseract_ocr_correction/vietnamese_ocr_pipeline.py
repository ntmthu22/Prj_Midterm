import cv2
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from tqdm import tqdm
import os
import re
import numpy as np

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\PC\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# Các tham số chung
DPI = 300
LANG = "vie"

OCR_CONFIG_NORMAL = "--oem 3 --psm 6"
OCR_CONFIG_SPECIAL = "--oem 3 --psm 11"
OCR_CONFIG_TABLE = "--oem 3 --psm 4" 

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

PDF_FILES = [
    "benh_ngoai_khoa_thuong_gap_va_cach_chua_tri_bang_y_hoc_co_truyen (1).pdf",
    "dong_y_nhap_mon.pdf",
    "500_bai_thuoc_hay_chua_benh_cao_huyet_ap.pdf",
    "ban_thao_van_dap.pdf",
    "thuoc_nam_chua_benh_tuyen_co_so.pdf"
]

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def crop_margins(img):
    h, w = img.shape[:2]
    return img[60:h-60, 90:w-90]


def remove_book_spine(img, ratio=0.04):
    h, w = img.shape[:2]
    mid = w // 2
    sw = int(w * ratio)
    img[:, mid-sw:mid+sw] = 255
    return img


def detect_columns(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    profile = np.sum(thresh, axis=0) 
    mid = profile.shape[0] // 2
    window_size = 200 
    if mid - window_size // 2 < 0 or mid + window_size // 2 > len(profile):
        return 1 
    min_in_mid = np.min(profile[mid - window_size // 2:mid + window_size // 2])
    max_profile = np.max(profile)
    if min_in_mid < 0.1 * max_profile: 
        return 2
    return 1


def split_columns(img):
    h, w = img.shape[:2]
    mid = w // 2
    return img[:, :mid], img[:, mid:]


def ocr(img, config):
    processed = preprocess(img)
    return pytesseract.image_to_string(
        processed,
        lang=LANG,
        config=config
    )


def clean_text(text):
    text = re.sub(r'[®©~]', '', text)
    text = re.sub(r'[^\x00-\x7FÀ-ỹ\n ]+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_special_page(text):
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) < 0.4


def process_pdf(pdf_path):
    output_docx = pdf_path.replace(".pdf", "_ocr.docx")
    print(f"🔄 Bắt đầu OCR file: {pdf_path} → {output_docx}")

    pages = convert_from_path(pdf_path, dpi=DPI, poppler_path=POPPLER_PATH)
    doc = Document()

    for i, page in enumerate(tqdm(pages, desc=f"📖 OCR {pdf_path}")):
        img_name = f"page_{i}.png"
        page.save(img_name)

        img = cv2.imread(img_name)
        img = crop_margins(img)

        num_columns = detect_columns(img)

        if num_columns == 2:
            img = remove_book_spine(img)
            left, right = split_columns(img)
            text_normal = ocr(left, OCR_CONFIG_NORMAL) + "\n\n" + ocr(right, OCR_CONFIG_NORMAL)
        else:
            text_normal = ocr(img, OCR_CONFIG_NORMAL)

        if is_special_page(text_normal):
            text_table = ocr(img, OCR_CONFIG_TABLE) if num_columns == 1 else text_normal
            if is_special_page(text_table):
                text = ocr(img, OCR_CONFIG_SPECIAL)
            else:
                text = text_table
        else:
            text = text_normal

        text = clean_text(text)

        if text:
            for line in text.split("\n"):
                doc.add_paragraph(line)

        doc.add_page_break()
        os.remove(img_name)

    doc.save(output_docx)
    print(f"OCR hoàn tất cho file: {pdf_path}")


def main():
    for pdf in PDF_FILES:
        if os.path.exists(pdf):
            process_pdf(pdf)
        else:
            print(f"File không tồn tại: {pdf}")


if __name__ == "__main__":
    main()