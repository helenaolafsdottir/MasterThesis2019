# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:23:39 2019

@author: Lenovo
"""


sections = [{'h1': 'section A',
             'paragraphs':['blabla','Hola Amigo. Como estas?','Ég er eins og ég eeeer. Páll Óskar.'], 
             'sub-sections':[{
                     'h2':'section A.a', 
                     'paragraphs':['halló','hallóó','haaallóóóóó!'], 
                     'sub-sections':[]}]},
            {'h1': 'Section B', 
             'paragraphs':['ég er hluti af section b'], 
             'sub-sections': []}]



def formulate_structure(sections):
    for section in sections:
        print('--- NEW SECTION ---')
        print('Heading: ', section['h1'])
        print('paragraphs in section: ', section['paragraphs'])
        print('subsections: ', section['sub-sections'])
        subsections =  section['sub-sections']
        for subsection in subsections:
            print('-- LOOKING AT SUBSECTIONS --')
            #print(subsection)
            print('Heading: ', subsection['h2'])
            print('paragraphs in section: ', subsection['paragraphs'])
            print('subsections: ', subsection['sub-sections'])