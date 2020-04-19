from PyPDF2 import PdfFileMerger
import os
os.chdir(r'C:\Users\hzxhz\Desktop')
pdfs = ['a.pdf','b.pdf']
merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(pdf)
merger.write(r'C:\Users\hzxhz\Desktop\Result.pdf')
merger.close()
