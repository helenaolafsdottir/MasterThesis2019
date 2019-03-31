# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:09:39 2019

@author: Lenovo
"""

from Snowflake_SAD import *

sentence_list = []

def formulate_structure_w_sentences(sections):
    
    f = open("sentences.txt", "a", encoding="utf-8")
    i = 1
    for section in sections:
        
        if(section['included']=='yes'):
            #f.write(str(i))
            #f.write("\t")
            f.write(section['h1'])
            #i+=1
            f.write("\n\n")
            sentence_list.append(section['h1'])
        
            paragraphs = section['paragraphs']
            for paragraph in paragraphs:
                fullParagraph = ''
                
                for sentence in range(len(paragraph)):
                    #f.write(str(i))
                    #f.write("\t")
                    f.write(paragraph[sentence])
                    #i+=1
                    f.write("\n\n")                
                    sentence_list.append(paragraph[sentence])                
                    
                    fullParagraph += (' ' + paragraph[sentence])

            
        subsections =  section['sub-sections']

        if(subsections):
            for subsection in subsections:
                if(subsection['included']=='yes'):
                    #f.write(str(i))
                   # f.write("\t")
                    f.write(subsection['h2'])
                    #i+=1
                    f.write("\n\n")
                    sentence_list.append(subsection['h2'])
                    
                    paragraphs = subsection['paragraphs']
                    for paragraph in paragraphs:
                        fullParagraph = ''
                        for sentence in range(len(paragraph)):
                            
                           # f.write(str(i))
                           # f.write("\t")
                            f.write(paragraph[sentence])
                            #i+=1
                            f.write("\n\n")
                            sentence_list.append(paragraph[sentence])  
                            
                            fullParagraph += (' ' + paragraph[sentence])
                
                subsubsections = subsection['sub-sections']
                
                if(subsubsections):
                    for subsubsection in subsubsections:
                        if(subsubsection['included']=='yes'):
                           #f.write(str(i))
                           # f.write("\t")   
                            f.write(subsubsection['h3'])
                           # i+=1
                            f.write("\n\n")
                            sentence_list.append(subsubsection['h3'])
                        
                            paragraphs = subsubsection['paragraphs']
                            for paragraph in paragraphs:
                                fullParagraph = ''
                                for sentence in range(len(paragraph)):
                                    
                                 #   f.write(str(i))
                                  #  f.write("\t")
                                    f.write(paragraph[sentence])
                                  #  i+=1
                                    f.write("\n\n")
                                    sentence_list.append(paragraph[sentence])  
                                    
                                    fullParagraph += (' ' + paragraph[sentence])
                                
                        subsubsubsections = subsubsection['sub-sections']
                        
                        if(subsubsubsections):
                            for subsubsubsection in subsubsubsections:
                                if(subsubsubsection['included']=='yes'):
                                   # f.write(str(i))
                                  #  f.write("\t")
                                    f.write(subsubsubsection['h4'])
                                  #  i+=1
                                    f.write("\n\n")
                                    sentence_list.append(subsubsubsection['h4'])
                                    
                                    paragraphs = subsubsubsection['paragraphs']
                                    for paragraph in paragraphs:
                                        fullParagraph = ''
                                        for sentence in range(len(paragraph)):
                                            
                                            #f.write(str(i))
                                            #f.write("\t")
                                            f.write(paragraph[sentence])
                                           # i+=1
                                            f.write("\n\n")
                                            sentence_list.append(paragraph[sentence])  
                                            
                                            fullParagraph += (' ' + paragraph[sentence])
      


formulate_structure_w_sentences(sections)
print(sentence_list)