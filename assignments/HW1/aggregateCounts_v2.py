#!/usr/bin/env python
"""
This script reads word counts from STDIN and aggregates
the counts for any duplicated words.

INPUT & OUTPUT FORMAT:
    word \t count
USAGE (standalone):
    python aggregateCounts_v2.py < yourCountsFile.txt

Instructions:
    For Q7 - Your solution should not use a dictionary or store anything   
             other than a single total count - just print them as soon as  
             you've added them. HINT: you've modified the framework script 
             to ensure that the input is alphabetized; how can you 
             use that to your advantage?
"""

# imports
import sys

########### PROVIDED IMPLEMENTATION ##############  

current_sum = 0
current_word = ""

# stream over lines from Standard Input
for line in sys.stdin:
    # extract words & counts
    word, count  = line.split()
    
    if current_word == "":
        current_word = word
    
    # if word matches current word, add to current sum
    if current_word == word:
        current_sum += int(count)
    else:
        # print count for finished word
        print(f"{current_word}\t{current_sum}")
        current_word = word
        current_sum = int(count)
    
########## (END) PROVIDED IMPLEMENTATION #########
