import sys
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from scripts import HelperFunctions

def test_seq_io():
    seq1,seq2=HelperFunctions.read_seq('sequences/prot-0004.fa','sequences/prot-0008.fa')
    assert seq1 == 'SLEAAQKSNVTSSWAKASAAWGTAGPEFFMALFDAHDDVFAKFSGLFSGAAKGTVKNTPEMAAQAQSFKGLVSNWVDNLDNAGALEGQCKTFAANHKARGISAGQLEAAFKVLSGFMKSYGGDEGAWTAVAGALMGEIEPDM'
    assert seq2 == 'ANKTRELCMKSLEHAKVDTSNEARQDGIDLYKHMFENYPPLRKYFKSREEYTAEDVQNDPFFAKQGQKILLACHVLCATYDDRETFNAYTRELLDRHARDHVHMPPEVWTDFWKLFEEYLGKKTTLDEPTKQAWHEIGREFAKEINK'

def test_matrix_io():
    BLOSUM50 = HelperFunctions.read_matrix('BLOSUM50')
    assert BLOSUM50[0] == [5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0, -2, -1, -1, -5]
    
def test_match_score():
    BLOSUM50 = HelperFunctions.read_matrix('BLOSUM50')
    score = HelperFunctions.match_score('P','K',BLOSUM50) 
    assert score == -1

def test_sw_alignment_part1():
    seq1,seq2=HelperFunctions.read_seq('sequences/prot-0004.fa','sequences/prot-0008.fa')
    BLOSUM50 = HelperFunctions.read_matrix('BLOSUM50')
    score,align1,align2 = HelperFunctions.alignment(seq1,seq2,-5,-1,BLOSUM50)
    print(score)
    assert score == 123.0

def test_sw_alignment_part2():
    '''Try with different gap penalties'''
    seq1,seq2=HelperFunctions.read_seq('sequences/prot-0004.fa','sequences/prot-0008.fa')
    BLOSUM50 = HelperFunctions.read_matrix('BLOSUM50')
    score,align1,align2 = HelperFunctions.alignment(seq1,seq2,-10,-5,BLOSUM50)
    assert score == 46.0

def test_sim_annealing():
    BLOSUM50 = HelperFunctions.read_matrix('BLOSUM50')
    pospairs = []
    negpairs = []
    with open('Pospairs.txt', 'r') as f1:
        for line in f1:
            line = line.strip()
            line = line.split(' ')
            pospairs.append(line)
    with open('Negpairs.txt', 'r') as f2:
        for line in f2:
            line = line.strip()
            line = line.split(' ')
            negpairs.append(line)
    score_pos,static_align1_pos,static_align2_pos = HelperFunctions.align_set(pospairs,-7,-3,BLOSUM50)
    score_neg,static_align1_neg,static_align2_neg = HelperFunctions.align_set(negpairs,-7,-3,BLOSUM50)
    static = []
    static.append(static_align1_pos)
    static.append(static_align2_pos)
    static.append(static_align1_neg)
    static.append(static_align2_neg)
    obj_values = HelperFunctions.sim_annealing(static,BLOSUM50,10)[0]
    # check that there are 4013 iterations
    assert len(obj_values) == 4013 
    # check that there are 2 values per iteration (old objective value, new objective value)
    assert len(obj_values[0]) == 2

def test_symmetric_optimized_matrix():
    BLOSUM50 = HelperFunctions.read_matrix('BLOSUM50')
    pospairs = []
    negpairs = []
    with open('Pospairs.txt', 'r') as f1:
        for line in f1:
            line = line.strip()
            line = line.split(' ')
            pospairs.append(line)
    with open('Negpairs.txt', 'r') as f2:
        for line in f2:
            line = line.strip()
            line = line.split(' ')
            negpairs.append(line)
    score_pos,static_align1_pos,static_align2_pos = HelperFunctions.align_set(pospairs,-7,-3,BLOSUM50)
    score_neg,static_align1_neg,static_align2_neg = HelperFunctions.align_set(negpairs,-7,-3,BLOSUM50)
    static = []
    static.append(static_align1_pos)
    static.append(static_align2_pos)
    static.append(static_align1_neg)
    static.append(static_align2_neg)
    optimized_matrix = HelperFunctions.sim_annealing(static,BLOSUM50,5)[1]
    assert optimized_matrix[6][3] == optimized_matrix[3][6]
    assert optimized_matrix[12][4] == optimized_matrix[4][12]
