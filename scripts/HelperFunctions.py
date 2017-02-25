import sys
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

def read_seq(fname_a,fname_b):
    '''read the first sequence, starting from 2nd line to avoid header'''
    with open(fname_a, 'r') as f1:
        next(f1) # skip header
        seq1 = ''.join(line.strip() for line in f1)
        seq1=seq1.strip() #remove newline chars, etc. 
        f1.close()
    
    #read the second sequence, starting from 2nd line to avoid header
    with open(fname_b, 'r') as f2:
        next(f2) # skip header
        seq2 = ''.join(line.strip() for line in f2)
        seq2=seq2.strip() #remove newline chars, etc. 
        f2.close()
    return seq1,seq2

def read_matrix(fname):
    ''' reads in file for the matrix and turns it into lists of mapped objects. '''
    matrix = []
    with open(fname, 'r') as f4:
        for line in f4:
            if line[1] != 'A' and line[0] != '#': # remove header, makes sure it starts at 
                # amino acid 'A'
                matrix.append(list(map(int, line.split())))
    return matrix

def alphabet_to_number(letter1,letter2):
    ''' Input are the two letters you want the pairwise score for. Returns the indices i,j for the substitution matrix'''
    # assigns each AA a unique number
    alphabet={}
    alphabet["A"] = 0
    alphabet["R"] = 1
    alphabet["N"] = 2
    alphabet["D"] = 3
    alphabet["C"] = 4
    alphabet["Q"] = 5
    alphabet["E"] = 6
    alphabet["G"] = 7
    alphabet["H"] = 8
    alphabet["I"] = 9
    alphabet["L"] = 10
    alphabet["K"] = 11
    alphabet["M"] = 12
    alphabet["F"] = 13
    alphabet["P"] = 14
    alphabet["S"] = 15
    alphabet["T"] = 16
    alphabet["W"] = 17
    alphabet["Y"] = 18
    alphabet["V"] = 19
    alphabet["B"] = 20
    alphabet["Z"] = 21
    alphabet["X"] = 22
    alphabet["x"] = 22
    lut_x=alphabet[letter1]
    lut_y=alphabet[letter2]
    return lut_x,lut_y

def match_score(alpha,beta,subs_matrix):
    ''' finds match/mismatch score from subs matrix based on letters of AAs'''
    i,j = alphabet_to_number(alpha,beta)
    return subs_matrix[i][j]

def alignment(seq1,seq2,gap_open,gap_extend,subs_matrix):
    ''' adapted code from Forest Bao, modified for this assignment.
    Link to the original code in the homework pdf.
    This function performs sw alignment based on a specified pair of sequences,
    gap opening penalty, gap extension penalty, and substitution matrix.
    Returns alignment score, as well as the aligned sequences'''
    m,n =  len(seq1),len(seq2) #length of the two sequences
    
    gap_open = float(gap_open);   #define the gap penalties
    gap_extend = float(gap_extend)
    
    # generate DP table and traceback path pointer matrix
    score=np.zeros((m+1,n+1))   #the DP table
    pointer=np.zeros((m+1,n+1))  #to store the traceback path
    
    # initialize penalty to zero
    P=0;
    max_score=P;  #initial maximum score in DP table
    
    #calculate DP table and mark pointers
    for i in range(m):
        for j in range(n):
            # start at top left of matrix
    
            if pointer[i-1][j] == 1 or pointer[i-1][j] == 2: #if there was a previous gap, 
            #apply gap extension penalty
                score_up = score[i-1][j]+gap_extend
            else: #if new gap, apply gap opening penalty:
                score_up = score[i-1][j]+gap_open
                
            # score "left" is marked by score_down, which is the other way to incur a gap penalty
            if pointer[i][j-1] == 1 or pointer[i][j-1] == 2:
                score_down = score[i][j-1] + gap_extend
            else:
                score_down = score[i][j-1] + gap_open
            
            score_diagonal=score[i-1][j-1]+match_score(seq1[i-1],seq2[j-1],subs_matrix);
            # update the score in DP table
            score[i][j]=max(0,score_up,score_down,score_diagonal);
            if score[i][j]==0:
                pointer[i][j]=0; #0 means end of the path
            if score[i][j]==score_up:
                pointer[i][j]=1; #1 means trace up
            if score[i][j]==score_down:
                pointer[i][j]=2; #2 means trace left
            if score[i][j]==score_diagonal:
                pointer[i][j]=3; #3 means trace diagonal
            if score[i][j]>=max_score:
                # update max score and max_i, max_j for traceback
                max_i=i;
                max_j=j;
                max_score=score[i][j];
    #END of DP table
    #now start to traceback to get sequence alignment
    align1,align2='',''; #initial sequences
    
    i,j=max_i,max_j; #indices of path starting point
    #traceback, follow pointers
    while pointer[i][j]!=0:
        if pointer[i][j]==3:
            #move diagonally
            align1=align1+seq1[i-1];
            align2=align2+seq2[j-1];
            i=i-1;
            j=j-1;
        elif pointer[i][j]==2:
            #move left, account for gap in sequence 1
            align1=align1+'-';
            align2=align2+seq2[j-1];
            j=j-1;
        elif pointer[i][j]==1:
            #move up, account for gap in sequence 2
            align1=align1+seq1[i-1];
            align2=align2+'-';
            i=i-1;

    #END of traceback
    
    align1=align1[::-1]; #reverse sequence 1
    align2=align2[::-1]; #reverse sequence 2
    
    return max_score,align1,align2

def align_set(pairs_set,gap_open,gap_extend,subs_matrix,norm=False):
    '''Aligns for a set of pairs (pospairs or negpairs), and lets you adjust penalties for gap opening and 
    extension. Returns all the scores for given pairs. Also has option to calculate the shortest sequence 
    in each pair and normalize the scores by shortest sequence. '''
    all_scores = []
    all_align1 = []
    all_align2 = []
    for l in pairs_set: 
        a,b = l[0],l[1]
        seq1,seq2 = read_seq(a,b)
        score,align1,align2 = alignment(seq1,seq2,gap_open,gap_extend,subs_matrix)
        all_align1.append(align1)
        all_align2.append(align2)
        if norm==True:
            # normalize by shortest seq length
            shortest = np.min([len(seq1),len(seq2)])
            score = score/shortest
        all_scores.append(score)
    return all_scores,all_align1,all_align2
    
def alignment_score(align1,align2,gap_open,gap_extend,subs_matrix):
    '''Calculate score for objective function calculation based on the aligned 
    sequences, gap opening/extension penalties, and the subs matrix'''
    score = 0
    for i in range(len(align1)):
        if align1[i] == '-' or align2[i] == '-':
            if align1[i-1] == '-' or align2[i-1] == '-':
                score += gap_extend
            else:
                score += gap_open
        else:
            score += match_score(align1[i],align2[i],subs_matrix)
    return score

def obj_func(static_alignments,subs_matrix):
    '''Objective function is the sum of the TPRs when the FPRs are 0, 0.1, 0.2, and 0.3
    Inputs: static starting alignments using optimal gap penalties (which come from the sim annealing function), subs_matrix. Returns evaluation of objective function'''

    all_scores = []
    labels = [] # 0 is true negative, 1 is true positive 

    static_align1_pos,static_align2_pos,static_align1_neg,static_align2_neg = static_alignments[0],static_alignments[1],static_alignments[2],static_alignments[3]
    
    # now, calculate scores to be used in calculating obj. function
    # manually set these penalties because they were determined from Part 1 and won't change throughout this optimization process 
    gap_open = -7
    gap_extend = -3

    #get scores of all positive pairs:
    for ix in range(len(static_align1_pos)):
        align1,align2 = static_align1_pos[ix], static_align2_pos[ix]
        score = alignment_score(align1,align2,gap_open,gap_extend,subs_matrix)
        all_scores.append(score)
        labels.append(1)
    
    #get scores of all negative pairs:
    for ix in range(len(static_align1_neg)):
        align1,align2 = static_align1_neg[ix], static_align2_neg[ix]
        score = alignment_score(align1,align2,gap_open,gap_extend,subs_matrix)
        all_scores.append(score)
        labels.append(0)

    fpr,tpr,thresh = roc_curve(labels,all_scores,pos_label=1)
    roc_auc = auc(fpr,tpr)
    obj = 0
    obj += np.interp(0.0,fpr,tpr)
    obj += np.interp(0.1,fpr,tpr)
    obj += np.interp(0.2,fpr,tpr)
    obj += np.interp(0.3,fpr,tpr)
    return obj
    
def sim_annealing(static_alignments,subs_matrix,stepsize):
    '''Performs simulated annealing on static alignments of the positive and 
    negative pairs. Annealing schedule is determined by T,TP_min, and alpha (stepsize).
    Returns a list of all the values of the objective function at every iteration, 
    along with the optimized substitution matrix'''
    
    obj_values = [] # list of objective function values at every iteration
    
    T = 1.0
    T_min = 0.3
    alpha = 0.9997 # rate of changing T (stepsize)
    
    static_align1_pos,static_align2_pos,static_align1_neg,static_align2_neg = static_alignments[0],static_alignments[1],static_alignments[2],static_alignments[3]
    
    obj = obj_func(static_alignments,subs_matrix) # get initial objective function value
    
    # T keeps on decreasing, making it harder to accept new moves
    while T > T_min: 

        new_matrix = np.array(subs_matrix,dtype='float128') # float128 because otherwise,
        # it's integers

        # select a random pairwise score to increase or decrease by stepsize
        i = np.random.randint(0,23)
        j = np.random.randint(0,23)
        sign = np.random.randint(0,2) # randomize increase or decrease. 
        #if 0, dec score. if 1, inc score
        if sign == 0:
            new_matrix[i][j] = new_matrix[i][j] - (stepsize) 
        else: # sign == 1
            new_matrix[i][j] = new_matrix[i][j] + (stepsize) 
        
        new_matrix[j][i] = new_matrix[i][j] #symmetry
        newobj = obj_func(static_alignments,new_matrix) # calculate new objective function value
        obj_values.append([obj,newobj])
        
        # acceptance probability
        ap = np.sign(newobj-obj)*np.exp((newobj-obj)/T)
        #ap = np.sign(newobj-obj)*np.exp((newobj-obj)/T)
        x = np.random.rand()
        if ap > x and obj != newobj:
            subs_matrix = new_matrix
            obj = newobj
        else:
            pass
        T = T*alpha
    return obj_values,subs_matrix

def plot_optimized_matrices(static_alignments,pospairs,negpairs,orig_matrix,optimized_matrix,realign=False):
    '''Compares the original/input substitution matrix to the optimized matrix.'''
    '''There is an option to realign using the new matrix and show that ROC'''
    static_align1_pos,static_align2_pos,static_align1_neg,static_align2_neg = static_alignments[0],static_alignments[1],static_alignments[2],static_alignments[3]
    
    fig,ax = plt.subplots()
    matrices = [orig_matrix,optimized_matrix]
    for subs_matrix in matrices:
        all_scores = []
        labels = [] # 0 is true negative, 1 is true positive 
        
        # now, calculate scores to be used in calculating obj. function
        # manually set these penalties because they were determined from Part 1 and won't change throughout this optimization process 
        gap_open = -7
        gap_extend = -3

        #get scores of all positive pairs:
        for ix in range(len(static_align1_pos)):
            align1,align2 = static_align1_pos[ix], static_align2_pos[ix]
            score = alignment_score(align1,align2,gap_open,gap_extend,subs_matrix)
            all_scores.append(score)
            labels.append(1)
        
        #get scores of all negative pairs:
        for ix in range(len(static_align1_neg)):
            align1,align2 = static_align1_neg[ix], static_align2_neg[ix]
            score = alignment_score(align1,align2,gap_open,gap_extend,subs_matrix)
            all_scores.append(score)
            labels.append(0)

        fpr,tpr,thresh = roc_curve(labels,all_scores,pos_label=1)
        roc_auc = auc(fpr,tpr)
        print(roc_auc)
        ax.plot(fpr, tpr, lw=3, label=subs_matrix)
    plot_labels = ['MATIO', 'optimized from MATIO']
    if realign == True:
        true_pos,align1pos,align2pos = align_set(pospairs,-7,-3,optimized_matrix)
        true_neg,align1neg,align2neg = align_set(negpairs,-7,-3,optimized_matrix)
        all_scores = []
        labels = [] # 0 is true negative, 1 is true positive 
        #get scores of all positive pairs:
        for ix in range(len(true_pos)):
            align1,align2 = align1pos[ix], align2pos[ix]
            score = true_pos[ix]
            all_scores.append(score)
            labels.append(1)
        
        #get scores of all negative pairs:
        for ix in range(len(true_neg)):
            align1,align2 = align1neg[ix], align2neg[ix]
            score = true_neg[ix]
            all_scores.append(score)
            labels.append(0)

        fpr,tpr,thresh = roc_curve(labels,all_scores,pos_label=1)
        roc_auc = auc(fpr,tpr)
        print(roc_auc)
        ax.plot(fpr, tpr, lw=3)
        plot_labels = ['MATIO with static alignment','optimized_matrix with static alignment','realignment with optimized matrix']

    plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(plot_labels,loc='lower right')
    plt.title('Original vs. optimized substitution matrices')
    plt.show()

