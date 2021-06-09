#!/usr/bin/env python
"""
Mapper tokenizes and emits words with their class.
INPUT:
    ID \t SPAM \t SUBJECT \t CONTENT \n
OUTPUT:
    word \t class \t count 
"""
import re
import sys
from collections import defaultdict

counts_spam = defaultdict(int)
counts_ham = defaultdict(int)

# read from standard input
for line in sys.stdin:
    # parse input
    docID, _class, subject, body = line.split('\t')
    # tokenize
    words = re.findall(r'[a-z]+', subject + ' ' + body)
    
############ YOUR CODE HERE #########
    for word in words:
        if _class == "0":
            counts_ham[word] += 1
        else:
            counts_spam[word] += 1

for word in counts_ham:
    print(f"{word}\t0\t{counts_ham[word]}\r")
for word in counts_spam:
    print(f"{word}\t1\t{counts_spam[word]}\r")

############ (END) YOUR CODE #########