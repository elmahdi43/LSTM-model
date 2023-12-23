import PyPDF2


def extract_text_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extractText()
    return text
