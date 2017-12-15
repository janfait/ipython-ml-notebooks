# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:31:24 2016

@author: jf186031
"""

import re
import string
from prettytable import PrettyTable


#define translations for a elements in string.punctuation object
def remove_punctuation(s):
    table = s.maketrans({key: None for key in string.punctuation})
    return s.translate(table)

#ingest text and     
def tokenize(text):
    text = remove_punctuation(text)
    text = text.lower()
    return re.split("\W+", text)

def count_words(words):
    wc = {}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0
    return wc

s = "Ever wanted a russian wife? Buy a russian wife online now."
count_words(tokenize(s))
