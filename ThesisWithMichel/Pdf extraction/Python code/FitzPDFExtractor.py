import fitz
from tabula import read_pdf

def retrieve_text(filepath):
    doc = fitz.open(filepath)
    
    metadata = doc.metadata
    ToC = doc.getToC()

    noPages = doc.pageCount
    paragraphs = []
    
    for n in range(noPages):
        #print("Current page: ", n)
        currPage = doc[n]
        #links = currPage.getLinks()
        #print('Links: ', links)
        text = currPage.getText("dict")
        #print('---------')
        #print('The text: ')
        if "blocks" in text:
            blocks = text["blocks"]
            
            for block in blocks:
                #print('----- Block: -----')
                #print(block)
                if "lines" in block:
                    lines = block["lines"]
                    #print('--- NEW PARAGRAPH STARTING ---')
                    #print('paragraphNo: ', paragraphNo)
                    paragraph = ''
                    for line in lines:
                        if "spans" in line:
                            spans = line["spans"]
                            for span in spans:
                                if "text" in span: 
                                    #print(span["text"]) 
                                    paragraph = paragraph + ' ' + span["text"]
                                    
                    paragraphs.append(paragraph)            

    return metadata, ToC, paragraphs



def retrieve_tables(filepath):
    tables = read_pdf(filepath, pages="all", multiple_tables=True)
    return tables

#REF: https://github.com/rk700/PyMuPDF/wiki/How-to-Extract-Images-from-a-PDFhttps://github.com/rk700/PyMuPDF/wiki/How-to-Extract-Images-from-a-PDF
def retrieve_figures(filepath):
    doc = fitz.open(filepath)
    #for i in range(len(doc)):
    for img in doc.getPageImageList(0):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n < 5:       # this is GRAY or RGB
            pix.writePNG("extractedImages/p%s-%s.png" % (0, xref))
        else:          
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1.writePNG("extractedImages/p%s-%s.png" % (0, xref))
            pix1 = None
        pix = None

filepath = "../SADs/Good/symphony-core2_Symphony_requirements_and_architecture.pdf"
metadata, ToC, paragraphs = retrieve_text(filepath)
tables = retrieve_tables(filepath)
retrieve_figures(filepath)

print('')
print('paragraphs: ')
for paragraph in paragraphs:
    print('---')
    print(paragraph)
#print(paragraphs)

print('')
print('Metadata:')
print(metadata)
print('')
print('Table of Contents:')
print(ToC)

print('')
print('')
print('Identified tables')
print(tables)