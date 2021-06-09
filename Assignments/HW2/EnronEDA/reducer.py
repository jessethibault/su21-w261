#!/usr/bin/env python
"""
Reducer takes words with their class and partial counts and computes totals.
INPUT:
    word \t class \t partialCount 
OUTPUT:
    word \t class \t totalCount  
"""
import re
import sys

# initialize trackers
current_word = None
spam_count, ham_count = 0,0

# read from standard input
for line in sys.stdin:
    # parse input
    word, is_spam, count = line.split('\t')
    
############ YOUR CODE HERE #########
    
    if current_word == None:
        current_word = word
    
    # if word matches current word, add to current sum
    if current_word == word:
        if is_spam == "0":
            ham_count += int(count)
        else:
            spam_count += int(count)
    else:
        # print count for finished word
        if ham_count > 0:
            print(f"{current_word}\t0\t{ham_count}")
        if spam_count > 0:
            print(f"{current_word}\t1\t{spam_count}")
            
        current_word = word
        
        if is_spam == "0":
            ham_count = int(count)
            spam_count = 0
        else:
            ham_count = 0
            spam_count = int(count)

if ham_count > 0:
    print(f"{current_word}\t0\t{ham_count}")
if spam_count > 0:
    print(f"{current_word}\t1\t{spam_count}")

############ (END) YOUR CODE #########