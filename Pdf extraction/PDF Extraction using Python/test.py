# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:53:46 2019

@author: Lenovo
"""

from xml.dom import minidom

xml_file = open('../SADs/Good/XMLs/upright_hdfs_design.xml', encoding="utf8")
xmldoc = minidom.parse(xml_file)
xml_file.close()

print(xmldoc)

print(xmldoc.documentElement.tagName)

headings1 = xmldoc.getElementsByTagName("h1")
headings2 = xmldoc.getElementsByTagName("h2")
headings3 = xmldoc.getElementsByTagName("h3")
headings4 = xmldoc.getElementsByTagName("h4")

for heading in headings1:
    print('h1 contents:', heading.firstChild.data)
print('-')
for heading in headings2:
    print('h2 contents:', heading.firstChild.data)
print('-')
for heading in headings3:
    print('h3 contents:', heading.firstChild.data)
print('-')
for heading in headings4:
    print('h4 contents:', heading.firstChild.data)
    
    
items = ['Tag1','Tag2']
for tag in items:
    for value in xml_file.getElementsByTagName(tag):
        project[tag].append(value.firstChild.data)











def printNode(node):
  print('node: ',node)
  headings = node.getElementsByTagName("h1")
  for child in node.childNodes:
       printNode(child)
   

#printNode(xmldoc.documentElement)
