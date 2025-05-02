from pdf2image import convert_from_path
import pytesseract

# Optional: Set the tesseract path on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Path to your PDF
pdf_path = 'sem 1 questions.pdf'

# Convert PDF pages to images
pages = convert_from_path(pdf_path)

# Extract text from each page image
for i, page in enumerate(pages):
    print(f"\n--- Text from page {i + 1} ---")
    text = pytesseract.image_to_string(page)
    print(text)
