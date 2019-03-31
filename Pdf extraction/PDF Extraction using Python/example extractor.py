from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, HTMLConverter, XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.image import ImageWriter
from io import BytesIO, StringIO

def extractTextAndImagesFromPDF(rawFile):
    laparams = LAParams()
    imagewriter = ImageWriter('extractedImageFolder/')    
    resourceManager = PDFResourceManager()

    outfp = BytesIO()  # Use StringIO to catch the output later.
    device = TextConverter(resourceManager, outfp, laparams=laparams, imagewriter=imagewriter)
    interpreter = PDFPageInterpreter(resourceManager, device)
    for page in PDFPage.get_pages(rawFile, set(), maxpages=0, caching=True, check_extractable=True):
        interpreter.process_page(page)
    device.close()    
    extractedText = outfp.getvalue()  # Get the text from the StringIO
    outfp.close()
    return extractedText 
text = extractTextAndImagesFromPDF("upright_hdfs_design.pdf")
print(text)