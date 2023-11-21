# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['id']]=js['label']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=int(label)
    return predictions

def read_uncertain(filename):
    uncertainty={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,sco = line.split()
            print(sco)
            uncertainty[int(idx)] = float(sco)
    return uncertainty


def calculate_scores(answers,predictions, uncertainty):
    classified={}
    missclassified = {}
    Acc=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        elif answers[key] == predictions[key]:
            classified[int(key)] = uncertainty[key]
        else:
            missclassified[int(key)] = uncertainty[key]

        Acc.append(answers[key]==predictions[key])

    print(len(classified))
    print(len(missclassified))

    mean_classify = sum(classified.values())/ len(classified)
    max_classify = max(classified.values())
    min_classify = min(classified.values())

    mean_misclassify = sum(missclassified.values())/len(missclassified)
    max_misclassify = max(missclassified.values())
    min_misclassify = min(missclassified.values())

    print("min of classified scores '{}' and max of classified scores '{}'".format(min_classify,max_classify))
    print("min of misclassified scores '{}' and max of misclassified scores '{}'".format(min_misclassify,max_misclassify))
    print("mean of classified scores '{}' and mean of misclassified scores '{}'".format(mean_classify,mean_misclassify))

    # create a list of misclassified and classified data ids
    misclassified_ids = list(missclassified.keys())
    #print(misclassified_ids)#[classified[i] for i in range(len(classified)) if predicted_labels[i] != true_labels[i]]
    classified_ids = list(classified.keys())#[data_ids[i] for i in range(len(data_ids)) if predicted_labels[i] == true_labels[i]]
    print("work")
    # plot the distribution of uncertainty scores for misclassified and classified data
    plt.hist([uncertainty[key] for key in uncertainty.keys() if key in misclassified_ids], color='r', alpha=0.5, label='Misclassified')
    plt.hist([uncertainty[key] for key in uncertainty.keys() if key in classified_ids], color='b', alpha=0.5, label='Classified')
    plt.xlabel('Uncertainty Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Uncertainty Scores for Classified and Misclassified Data')
    plt.legend()
    plt.savefig('./scores/normalized.png') # save the plot as a PNG image
    plt.show()
    print("works")

    y_labels = list(answers.values())
    
    y_scores = list(uncertainty.values())
    #auc_Score = roc_auc_score(y_labels, y_scores, multi_class='ovr')
    #print(auc_Score)
    count = 0
    clasi= 0
    #write the classified and missclassified indexes with scores to text file. 
    fc= open("./scores/newclassified.txt", "w")
    for k in classified.keys():
        
        fc.write(" {}  {}\n".format(k, classified[k]))
        if(classified[k] <0.27):
            clasi = clasi+1
    fc.close()

    fm= open("./scores/misclassified.txt", "w")
    for k in missclassified.keys():
        fm.write(" {}  {}\n".format(k, missclassified[k]))
        if(missclassified[k] <0.27):
            count = count+1
    fm.close()
    print("count is {}", count)
    print("classi is {}",clasi)
    scores={}
    scores['Acc']=np.mean(Acc)
    return scores

def find_threshold(answers, predictions, uncertainty):
    data = []
    for key in answers:
        if key in predictions and key in uncertainty:
            prediction = predictions[key]
            truth = answers[key]
            score = uncertainty[key]
            data.append((key,score,prediction,truth))

    sorted_data = sorted(data, key=lambda x:x[1])


    confusion_matrices = []
    accuracies = []
    for threshold_index in range(len(sorted_data)):
        threshold = sorted_data[threshold_index][1]
        y_pred = [1 if score >= threshold else 0 for _,score, _, _ in sorted_data]
        y_true = [g for _,_, _, g in sorted_data]
        cm = confusion_matrix(y_true, y_pred)
        tp, tn, fp, fn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        confusion_matrices.append(cm)
        accuracies.append(accuracy)

    # Find the threshold with the highest accuracy
    best_threshold_index = np.argmax(accuracies)
    best_threshold = sorted_data[best_threshold_index][1]
    print(best_threshold)
"""
def find_threshold(answers, predictions, uncertainty):
    data=[]
    for key in answers:
        if key in predictions and key in uncertainty:
            prediction = predictions[key]
            truth = answers[key]
            score = uncertainty[key]
            data.append((key,prediction,truth,score))

    #sort data
    sorted_data = sorted(data, key=lambda x:x[3])

    #cumulative sum 
    classified = np.cumsum([1 if p==t else 0 for _,p,t,_ in sorted_data])
    missclassified = np.cumsum([1 if p!=t else 0 for _,p,t,_ in sorted_data])


    #compute propotions
    total_correct = sum([1 if p == g else 0 for _, p, g, _ in sorted_data])
    total_misclassified = sum([1 if p != g else 0 for _, p, g, _ in sorted_data])
    correctly_classified_prop = classified / total_correct
    misclassified_prop = missclassified / total_misclassified

    #find threshold uncertainty score
    sen = correctly_classified_prop
    spe = 1- missclassified
    auc = np.trapz(sen,spe)
    optimal_id = np.argmax(sen+spe)
    threshold = sorted_data[optimal_id][3]

    # Print the threshold uncertainty score
    print("Optimal threshold uncertainty score:", threshold)
"""

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument('--uncertainty', '-u', help="filename of the uncertainty scores in txt format")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    uncertainty = read_uncertain(args.uncertainty)
    #print(uncertainty)
    scores=calculate_scores(answers,predictions, uncertainty)
    find_threshold(answers, predictions, uncertainty)
    print(scores)

if __name__ == '__main__':
    main()