# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:02:10 2019

@author: Helena Olafsdottir
"""

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, HTMLConverter, XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.image import ImageWriter
from io import BytesIO, StringIO

def pdf_to_text(path):
    manager = PDFResourceManager(caching=True)
    retstr = BytesIO()  
    laparams = LAParams()
    device = HTMLConverter(manager, retstr, laparams=laparams)
    filepath = open(path, 'rb')
    interpreter = PDFPageInterpreter(manager, device)

    for page in PDFPage.get_pages(filepath, set(), maxpages=0, caching=True, check_extractable=True):
        interpreter.process_page(page)
    device.close()
    text = retstr.getvalue()
    
    

    filepath.close()
    
    retstr.close()
    
    text_file = open("Output.txt", "w")
    text_file.write(str(text))
    text_file.close()
    return text


if __name__ == "__main__":
    text = pdf_to_text("upright_hdfs_design.pdf")
    print(text)