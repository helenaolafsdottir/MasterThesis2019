# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:41:26 2019

@author: Lenovo
"""

from Snowflake_SAD import *

  
    
paragraph_list = []
sentence_list = []

def list_paragraphs(sections,i):
    heading = 'h'+str(i)
    for section in sections:
        paragraph_list.append(section[heading])
        paragraphs = section['paragraphs']
        if(paragraphs):
            for paragraph in paragraphs:
                fullParagraph = ''
                for sentence in range(len(paragraph)):
                    fullParagraph += (' ' + paragraph[sentence])
                paragraph_list.append(fullParagraph)
            
        subsections =  section['sub-sections']
        if(subsections):
            i+=1
            list_paragraphs(subsections,i)
        i=1
    
def list_sentences(sections,i):
    for section in sections:
        heading = 'h'+str(i)
        print('------------1111111111-------------')
        print(section)
        if heading in section:
            print('heading found')
            print('------------2222222-------------')
            print(section)
            print('-- current heading --')
            print(heading)
            print('-- current section --')
            print(section)
            print('')
            
            sentence_list.append(section[heading])
            paragraphs = section['paragraphs']
            if(paragraphs):
                for paragraph in paragraphs:
                    for sentence in range(len(paragraph)):
                        sentence_list.append(paragraph[sentence])
            subsections =  section['sub-sections']
            if(subsections):
                print('new sub-section')
                i+=1
                print('i taken to new iteration: ', i)
                list_sentences(subsections,i)
        else:
            print('heading NOT found')
            i-=1
            print('i taken to new iteration: ', i)
            print('--- testing1--------')
            print(section)
            
    print('---------new section----------')
    #i=1



#list_paragraphs(sections,1)
list_sentences(sections,1)


#for paragraph in paragraph_list:
#    print('')
#    print(paragraph)
#
#print(sentence_list)
#for sentence in sentence_list:
#    print('')
#    print(sentence)



#print(len(paragraph_list))
#print('')
#print(paragraph_list)
#print('')
#print(len(paragraph_list))
#print('')
#print('')
#print(sentence_list)
#print('')
#print(len(sentence_list))