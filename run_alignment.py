from HelperFunctions import *
import sys
import numpy as np, matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import roc_curve,auc
import seaborn as sns

#############################################################
########### SET UP MATRICES AND SEQUENCES ###################
#############################################################
#set up all the matrices
BLOSUM50 = read_matrix('BLOSUM50')
BLOSUM62 = read_matrix('BLOSUM62')
PAM100 = read_matrix('PAM100')
PAM250 = read_matrix('PAM250')
MATIO = read_matrix('MATIO')
# make list of lists containing pos and neg pairs
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

#############################################################
####### FINDING THRESHOLDS WHERE TRUE POS RATE IS 0.7 #######
################## part 1 question 1 ########################
#############################################################

# use dict to store information about penalties, true pos rates, false pos rates, and thresholds
# key = index
# value = [ gap opening penalty, gap extension penalty, threshold where TPR is closest to 0.7, TPR closest to 0.7, fpr at that threshold and gap penalty combination ] 

#dict = {}
#
#key = 0 # key for dictionary
#for opening in range(-1,-21,-1):
#    for extension in range(-1,-6,-1):
#        # for these combinations of penalty scores:
#        true_pos_scores = align_set(pospairs,opening,extension,BLOSUM50)[0]
#        
#        # stop when true_pos_rate dips below 0.7 and see which 
#        #threshold gives rate closest to 0.7
#        true_pos_rate = 1
#        for thresh in range(int(np.min(true_pos_scores)),int(np.max(true_pos_scores))):
#            true_count = 0
#            for score in true_pos_scores:
#                if score >= thresh:
#                    true_count += 1
#            true_pos_rate = true_count / (len(pospairs))
#            if true_pos_rate > 0.7:
#                pass
#            else:
#                break
#        # backtrack and try the previous threshold value to see if the true pos rate >0.7 
#        #is actually closer to 0.7 than the one <0.7
#        new_count = 0
#        for score in true_pos_scores:
#            if score >= thresh-1:
#                new_count +=1
#        new_true_pos_rate = new_count / len(pospairs)
#        # select threshold that's closest to 0.7:
#        if np.abs(new_true_pos_rate-0.7) < np.abs(true_pos_rate-0.7):
#            best_thresh = thresh-1
#            best_rate = new_true_pos_rate
#        else:
#            best_thresh = thresh
#            best_rate = true_pos_rate
#        # calculate FPR at these combinations of gap opening/extension penalties and TPR threshold
#        true_neg_scores = align_set(negpairs,opening,extension,BLOSUM50)[0]
#        false_pos = 0 # neg pairs that exceed threshold
#        for score in true_neg_scores:
#            if score >= best_thresh:
#                false_pos += 1
#        false_pos_rate = false_pos / len(negpairs)
#        dict[key] = []
#        dict[key].append(opening)
#        dict[key].append(extension)
#        dict[key].append(best_thresh)
#        dict[key].append(best_rate)
#        dict[key].append(false_pos_rate)
#        key += 1
#        print(key,opening,extension,best_thresh,best_rate,false_pos_rate)
            
## find gap opening and extension penalties where fpr is lowest
#fpr = []
#for key,value in dict.iteritems():
#    opening,extension,best_thresh,t_pr,f_pr = value[0],value[1],value[2],value[3],value[4]
#    fpr.append(f_pr)
#print(np.min(fpr))
## minimum is 0.22, which corresponds to opening penalty of -7 and extension penalty of -3

## compute scores for all pairs and compute roc curve
## use combination of gap opening penalty of -7, extension penalty of -3, and a threshold of 62, 
##which had the lowest fpr using BLOSUM50

#all_scores = []
#labels = [] # 0 is true negative, 1 is true positive 
#
#true_pos = align_set(pospairs,-7,-3,BLOSUM50)[0]
#true_neg = align_set(negpairs,-7,-3,BLOSUM50)[0]
#
#for i in true_pos:
#    all_scores.append(i)
#    labels.append(1)
#for i in true_neg: 
#    all_scores.append(i)
#    labels.append(0)
#fpr,tpr,thresh = roc_curve(labels,all_scores,pos_label=1)
#roc_auc = auc(fpr,tpr)
#
#plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
#plt.plot([0, 1], [0.7,0.7], color='red', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC for alignments using BLOSUM50')
#plt.annotate('(0.22,0.70)', xy=(0.22,0.7),xytext=(.1, .9),arrowprops=dict(facecolor='black',shrink=1,width=0.05,headwidth=4))
#plt.legend(loc="lower right")
#plt.show()

##################################################################
####### PERFORMANCE RATES FOR OTHER SUBSTITUION MATRICES #########
################## part 1 question 2 #############################
##################################################################
# use combination of gap opening penalty of -7, extension penalty of -3

#fig,ax = plt.subplots()
#for matrix in [BLOSUM50,BLOSUM62,PAM100,PAM250,MATIO]:
#    all_scores = [] # all scores meaning pos and neg
#    labels = [] # 0 is true negative, 1 is true positive 
#    
#    true_pos = align_set(pospairs,-7,-3,matrix)[0]
#    true_neg = align_set(negpairs,-7,-3,matrix)[0]
#    for i in true_pos:
#        all_scores.append(i)
#        labels.append(1)
#    for i in true_neg: 
#        all_scores.append(i)
#        labels.append(0)
#    fpr,tpr,thresh = roc_curve(labels,all_scores,pos_label=1)
#    roc_auc = auc(fpr,tpr)
#    print(roc_auc)
#    ax.plot(fpr, tpr, lw=3,label=matrix)
#
#plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
#plt.plot([0, 1], [0.7,0.7], color='red', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend(['BLOSUM50','BLOSUM62','PAM100','PAM250','MATIO'],loc="lower right")
#plt.title('Performance between different substitution matrices')
#plt.show()

##################################################################
####### PERFORMANCE RATES WITH NORMALIZED SCORES  ################
################## part 1 question 3 #############################
##################################################################
## use combination of gap opening penalty of -7, extension penalty of -3
#fig,ax = plt.subplots()
#for matrix in [BLOSUM50,BLOSUM62,PAM100,PAM250,MATIO]:
#    all_scores = [] # all scores meaning pos and neg
#    labels = [] # 0 is true negative, 1 is true positive 
#    
#    true_pos = align_set(pospairs,-7,-3,matrix,norm=True)[0]
#    true_neg = align_set(negpairs,-7,-3,matrix,norm=True)[0]
#   
#    for i in true_pos:
#        all_scores.append(i)
#        labels.append(1)
#    for i in true_neg: 
#        all_scores.append(i)
#        labels.append(0)
#    fpr,tpr,thresh = roc_curve(labels,all_scores,pos_label=1)
#    roc_auc = auc(fpr,tpr)
#    print(roc_auc)
#    ax.plot(fpr, tpr, lw=3,label=matrix)
#plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
#plt.plot([0, 1], [0.7,0.7], color='red', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend(['BLOSUM50','BLOSUM62','PAM100','PAM250','MATIO'],loc="lower right")
#plt.title('Performance rate with scores normalized to shortest length')
#plt.show()

##show plot for BLOSUM50 with and without normalization
#fig,ax = plt.subplots()
#for matrix in [BLOSUM50]:
#    all_scores = [] # all scores meaning pos and neg
#    labels = [] # 0 is true negative, 1 is true positive 
#    
#    true_pos = align_set(pospairs,-7,-3,matrix)[0]
#    true_neg = align_set(negpairs,-7,-3,matrix)[0]
#    for i in true_pos:
#        all_scores.append(i)
#        labels.append(1)
#    for i in true_neg: 
#        all_scores.append(i)
#        labels.append(0)
#    fpr,tpr,thresh = roc_curve(labels,all_scores,pos_label=1)
#    roc_auc = auc(fpr,tpr)
#    print(roc_auc)
#    ax.plot(fpr, tpr, lw=3,label=matrix)
#
#for matrix in [BLOSUM50]:
#    all_scores = [] # all scores meaning pos and neg
#    labels = [] # 0 is true negative, 1 is true positive 
#    
#    true_pos = align_set(pospairs,-7,-3,matrix,norm=True)[0]
#    true_neg = align_set(negpairs,-7,-3,matrix,norm=True)[0]
#    for i in true_pos:
#        all_scores.append(i)
#        labels.append(1)
#    for i in true_neg: 
#        all_scores.append(i)
#        labels.append(0)
#    fpr,tpr,thresh = roc_curve(labels,all_scores,pos_label=1)
#    roc_auc = auc(fpr,tpr)
#    print(roc_auc)
#    ax.plot(fpr, tpr, lw=3,label=matrix)
#
#plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
#plt.plot([0, 1], [0.7,0.7], color='red', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#
#plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
#plt.plot([0, 1], [0.7,0.7], color='red', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend(['raw scores','normalized scores'],loc="lower right")
#plt.title('Performance rate with normalized scores using BLOSUM50')
#plt.show()


##################################################################
########### OPTIMIZE SUBS MATRIX FOR THIS DATASET ################
################## part 2 question 1 #############################
##################################################################
## first, generate static alignments to be optimized and 
## use BLOSUM50 as starting point
#score_pos,static_align1_pos,static_align2_pos = align_set(pospairs,-7,-3,BLOSUM50)
#score_neg,static_align1_neg,static_align2_neg = align_set(negpairs,-7,-3,BLOSUM50)
#static = []
#static.append(static_align1_pos)
#static.append(static_align2_pos)
#static.append(static_align1_neg)
#static.append(static_align2_neg)

##perform simulated annealing on static alignments
##to optimize BLOSUM50
##try different step sizes!

#opt_results_5 = sim_annealing(static,BLOSUM50,5)
#np.save('optimization_5.npy',opt_results_5)
#opt_results_10 = sim_annealing(static,BLOSUM50,10)
#np.save('optimization_10.npy',opt_results_10)
#opt_results_25 = sim_annealing(static,BLOSUM50,25)
#np.save('optimization_25.npy',opt_results_25)
#opt_results_50 = sim_annealing(static,BLOSUM50,50)
#np.save('optimization_50.npy',opt_results_50)

## plot objective function values to see which stepsize is best
#fig,ax = plt.subplots()
#for stepsize in [5,10,25,50]:
#    results = np.load('optimization_%s.npy' % stepsize)
#    obj_vals,optimized_matrix = results[0],results[1]
#    y = []
#    x = np.arange(len(obj_vals)-200)
#    for i in range(len(obj_vals)-200):
#        y.append(obj_vals[i][0])
#    ax.plot(x,y,lw=3,label=stepsize)
#plt.xlabel('Iteration number')
#plt.ylabel('Sum of TPR at FPRs of 0, 0.1, 0.2, 0.3')
#plt.legend(['stepsize 5','stepsize 10','stepsize 25', 'stepsize 50'], loc = 'lower right')
#plt.title('Effect of step size on optimization')
#plt.show()
#
##################################################################
########### COMPARE OPTIMIZED MATRIX TO BLOSUM50  ################
################## part 2 question 2 #############################
##################################################################
## plot ROC for original vs. optimized matrix
#optimized_matrix = np.load('optimization_25.npy')[1]
#plot_optimized_matrices(static,pospairs,negpairs,BLOSUM50,optimized_matrix)

## plot same graph with ROC after realigning with optimized matrix
#plot_optimized_matrices(static,pospairs,negpairs,BLOSUM50,optimized_matrix,realign=True)

##################################################################
########### OPTIMIZE MATIO MATRIX WITH SIM ANNEALING  ############
################## part 2 question 3 #############################
##################################################################
## first, generate static alignments to be optimized (same as 
## part 2 question 1
## use MATIO as starting point

#score_pos,static_align1_pos,static_align2_pos = align_set(pospairs,-7,-3,BLOSUM50) # BLOSUM50 because question says to use same initial alignments
#score_neg,static_align1_neg,static_align2_neg = align_set(negpairs,-7,-3,BLOSUM50)
#static = []
#static.append(static_align1_pos)
#static.append(static_align2_pos)
#static.append(static_align1_neg)
#static.append(static_align2_neg)

#perform simulated annealing on static alignments
#to optimize MATIO, using same
#protocol (which is step size of 25)
#try different step sizes!

#opt_results_matio = sim_annealing(static,BLOSUM50,25)
#np.save('optimization_matio.npy',opt_results_matio)

# plot ROC for original MATIO vs. optimized matrix
optimized_matrix = np.load('optimization_matio.npy')[1]
#plot_optimized_matrices(static,pospairs,negpairs,MATIO,optimized_matrix)

# plot same graph with ROC after realigning with optimized matrix
plot_optimized_matrices(static,pospairs,negpairs,BLOSUM50,optimized_matrix,realign=True)

##################################################################
