import sys
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from alignment_HW3 import HelperFunctions

def test_seq_io():
    seq1,seq2=read_seq('sequences/prot-0004.fa','sequences/prot-0008.fa')
    assert seq1 == 'SLEAAQKSNVTSSWAKASAAWGTAGPEFFMALFDAHDDVFAKFSGLFSGAAKGTVKNTPEMAAQAQSFKGLVSNWVDNLDNAGALEGQCKTFAANHKARGISAGQLEAAFKVLSGFMKSYGGDEGAWTAVAGALMGEIEPDM'
    assert seq2 == 'ANKTRELCMKSLEHAKVDTSNEARQDGIDLYKHMFENYPPLRKYFKSREEYTAEDVQNDPFFAKQGQKILLACHVLCATYDDRETFNAYTRELLDRHARDHVHMPPEVWTDFWKLFEEYLGKKTTLDEPTKQAWHEIGREFAKEINK'

def test_matrix_io():
    BLOSUM50 = read_matrix('BLOSUM50')
    assert BLOSUM50[0] == [5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0, -2, -1, -1, -5]
    
def test_match_score():
    BLOSUM50 = read_matrix('BLOSUM50')
    score = match_score('P','K',BLOSUM50) 
    assert score == -1

def test_sw_alignment_part1():
    seq1,seq2=read_seq('sequences/prot-0004.fa','sequences/prot-0008.fa')
    BLOSUM50 = read_matrix('BLOSUM50')
    score,align1,align2 = alignment(seq1,seq2,-5,-1,BLOSUM50)
    print(score)
    assert score == 123.0

def test_sw_alignment_part2():
    '''Try with different gap penalties'''
    seq1,seq2=read_seq('sequences/prot-0004.fa','sequences/prot-0008.fa')
    BLOSUM50 = read_matrix('BLOSUM50')
    score,align1,align2 = alignment(seq1,seq2,-10,-5,BLOSUM50)
    assert score == 46.0

def test_symmetric_optimized_matrix():
    optimized_matrix = sim_annealing(static,BLOSUM50,5)[1]
    assert optimized_matrix[6][3] == optimized_matrix[3][6]
    assert optimized_matrix[12][4] == optimized_matrix[4][12]
