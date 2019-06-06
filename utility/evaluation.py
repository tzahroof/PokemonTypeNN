import pandas as pd
import csv
import os
import matplotlib.image as mpimg
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import color
import numpy as np
from skimage.util import dtype

def load_type_dict():
    data_path = os.path.join(os.getcwd(),"data")
    types_file = "types.csv"
    types_path = os.path.join(data_path,types_file)
    types_csv = pd.read_csv(types_path)
    types_csv = types_csv.to_dict(orient="records")

    type_dict={}

    for i in range(0, len(types_csv)) :
        type_dict[i] = {"label":types_csv[i]["identifier"], 
                        "color":types_csv[i]["color"]}
    return type_dict

def plot_evaluation(label,metric1_label, metric1, metric2_label, metric2, title_metric_label, 
                    title_metric, type_dict = load_type_dict()):
    
    #Create grid for plotting
    fig = plt.figure(figsize=(8.8,5))
    gs = gridspec.GridSpec(2,1, height_ratios = (1,9))
       
    #Plotting model-level metrics
    ax = plt.subplot(gs[0])     
    ax.axis("off")    
    ax.annotate(("{} Average {} Score = {:.0%}").format(label, title_metric_label, title_metric), 
                (0.5, 0.5), color='#000000', fontsize=18, ha='center', va='center')     
        
    #Ploting class-level metrics
    ax = plt.subplot(gs[1]) 
    ax.axis("off")    
    
    #In some cases, there are no records of some classes (usually 3:Flying) Here, we fill
    #up the missing classes with 'None' values.
    unique_labels = np.arange(start=0, stop=19) #number of types in Pokemon
    metrics = dict( (key, {metric1_label : None, metric2_label : None}) for key in range(0,18))
    for key, v_metric1, v_metric2 in zip(unique_labels, metric1, metric2):
        metrics[key][metric1_label] = v_metric1
        metrics[key][metric2_label] = v_metric2

    #Writing the headers of the class table
    ax.annotate(metric1_label, (0.27, 19/20), color='#000000', weight='bold', 
                fontsize=12, ha='center', va='center')     
    ax.annotate(metric2_label, (0.4, 19/20), color='#000000', weight='bold', 
                fontsize=12, ha='center', va='center')        
    ax.annotate(metric1_label, (0.77, 19/20), color='#000000', weight='bold', 
                fontsize=12, ha='center', va='center')        
    ax.annotate(metric2_label, (0.9, 19/20), color='#000000', weight='bold', 
                fontsize=12, ha='center', va='center')            
    
    #Writing the metrics for each class
    for i, (pkm_type, metric) in enumerate(metrics.items()):
        column = int(i/9)
        row = i % 9 + 1
        left = column*0.5
        top = 1.0-(row+1)*1/10
        type_box = patches.Rectangle(
            (left,top),
            0.2,
            1/9,
            fc = type_dict[pkm_type]["color"],
            ec = "#FFFFFF"            
        ) 
        ax.add_patch(type_box)
        ax.annotate(type_dict[pkm_type]["label"], (left+0.1, top+1/20), color='#FFFFFF', weight='bold', 
                    fontsize=12, ha='center', va='center')  
        #Precision
        if metric[metric1_label] is not None:
            ax.annotate("{:.0%}".format(metric[metric1_label]), (left+0.27, top+1/20), color='#000000', 
                        fontsize=14, ha='center', va='center')    
        #Recall
        if metric[metric2_label] is not None:
            ax.annotate("{:.0%}".format(metric[metric2_label]), (left+0.40, top+1/20), color='#000000', 
                        fontsize=14, ha='center', va='center')


def getTopXTruePos(output_vector, actual, top_number):
    if top_number > len(actual):
        raise ValueError("top_number is greater than the array size")
    elif top_number <= 0:
        raise ValueError("top_number is less than or equal to 0")
    elif output_vector.shape != actual.shape:
        raise ValueError("Input arrays are not of the same shape")
    #Get the indexes of highest classes
    #Highest element first
    #Only keep the top number
    predlist = np.argsort(-output_vector)[:top_number]

    #Bothlist is the correct classes identified (indices)
    actuallist = np.where(actual == 1)[0]
    bothlist = np.intersect1d(predlist, actuallist)

    true_pos = np.zeros(18)
    true_pos[bothlist] = 1
    return true_pos

def getUnderlyingDist(output_vector, actual, threshold):

    if(output_vector.shape != actual.shape):
        raise ValueError("Input arrays are not of the same shape")

    true_pos  = np.zeros(18)
    false_pos = np.zeros(18)
    false_neg = np.zeros(18)
    prediction= np.zeros(18)

    #Keep the first two numbers
    output_idces = np.argsort(-output_vector)[:2]

    #Keep the prediction classes where the class prediction >= 0.5
    predlist = output_idces[np.where(output_vector[output_idces] >= threshold)[0]]

    #If none are greater than 0.5, take the max prediction
    if(len(predlist) == 0):
        predlist = output_idces[0:1]

    #What we guessed
    prediction[predlist] = 1
    
    #indices which label the actual type
    actuallist = np.where(actual == 1)[0]

    #Correctly identified classes (union between prediction and actual list)
    bothlist = np.intersect1d(predlist, actuallist)

    true_pos[bothlist] = 1

    #False Positives (in prediction, but not in the union of the two)
    falseposlist = np.setdiff1d(predlist, bothlist)
    false_pos[falseposlist] = 1

    #False Negatives (in actual, but not in the union of the two)
    falseneglist = np.setdiff1d(actuallist, bothlist)
    false_neg[falseneglist] = 1


    return {"true_pos":true_pos,
            "false_pos":false_pos,
            "false_neg":false_neg,
            "prediction":prediction}


def getMetrics(true_pos, false_pos, false_neg, actual_dist):
    classes = ["normal","fighting","flying","poison","ground",
        "rock","bug","ghost","steel","fire",
        "water","grass","electric","psychic","ice","dragon","dark","fairy"]

    precision = (true_pos) / ( (true_pos) + (false_pos))
    precision = np.round(precision, decimals=2)
    recall    = (true_pos) / ( (true_pos) + (false_neg))
    recall    = np.round(recall, decimals=2)
    F1 = 2 * precision * recall/ (precision + recall)
    F1 = np.round(F1, decimals=2)
    accuracy  = (true_pos) / actual_dist
    accuracy  = np.round(accuracy, decimals=2)



    mean_Precision = np.sum(true_pos)/(np.sum(true_pos) + np.sum(false_pos))
    mean_Recall    = np.sum(true_pos)/(np.sum(true_pos) + np.sum(false_pos))
    mean_F1 = 2 * mean_Precision * mean_Recall / (mean_Precision + mean_Recall)

    ## UNCOMMENT FOR averaged mean stats based on classes (each class equally weighted, 
    ## regardless of number of examples)
    # nprec = len(classes)
    # nrec = len(classes)
    # nf1 = len(classes)

    # mean_Precision = 0
    # mean_Recall = 0
    # mean_F1 = 0

    # for t in range(0, len(classes)):
    #     if(not np.isnan(precision[t])):
    #         mean_Precision += precision[t]
    #     else:
    #         nprec -= 1
        
    #     if(not np.isnan(recall[t])):
    #         mean_Recall += recall[t]
    #     else:
    #         nrec -= 1
            
    #     if(not np.isnan(F1[t])):
    #         mean_F1 += F1[t]
    #     else:
    #         nf1 -= 1
            
    # mean_Precision /= nprec
    # mean_Recall    /= nrec
    # mean_F1        /= nf1

    mean_accuracy = np.sum(true_pos)/np.sum(actual_dist)

    metrics = {}

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["F1"] = F1
    metrics["mean_precision"] = mean_Precision
    metrics["mean_recall"] = mean_Recall
    metrics["mean_F1"] = mean_F1
    metrics["accuracy"] = accuracy
    metrics["mean_accuracy"] = mean_accuracy

    return metrics

def plot_sprite(sprite,type_1=1,type_2=None,pred=None,type_dict = load_type_dict(),save=None,save_path="./classification"):
    #Definindo as dimensões do Grid
    if pred:
        grid_rows = 2
        grid_cols = 2
        figsize = (8, 4.4)        
        width_ratios = (1, 1)
        sprite_grid = 0
        pred_grid = 1
        type_grid = 2
    else:
        grid_rows = 2
        grid_cols = 1
        figsize = (4, 4.4)
        width_ratios = (1,)
        sprite_grid = 0
        type_grid = 1
    
        
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_rows,grid_cols, height_ratios = (10,1), width_ratios = width_ratios)
    
    #Plotando o sprite do Pokemon
    ax_sprite = plt.subplot(gs[sprite_grid])
    #ax_sprite.imshow(color.hsv2rgb(sprite))
    ax_sprite.imshow(sprite)    
    
    #Plotando o tipo verdadeiro do pokemon
    ax_type = plt.subplot(gs[type_grid])
    plt.axis("off")
    
    type_box_01 = patches.Rectangle(
        (0,0),
        0.5 if type_2 else 1,  
        1,  
        fc = type_dict[type_1]["color"],
        ec = "#FFFFFF"
    )    
    ax_type.add_patch(type_box_01)
    ax_type.annotate(type_dict[type_1]["label"], (0.25 if type_2 else 0.5, 0.5), color='w', weight='bold', 
                fontsize=12, ha='center', va='center')    
    if type_2:
        type_box_02 = patches.Rectangle(
            (0.5,0),
            0.5,  
            1,  
            fc = type_dict[type_2]["color"],
            ec = "#FFFFFF"
        )            
        ax_type.add_patch(type_box_02)
        ax_type.annotate(type_dict[type_2]["label"], (0.75, 0.5), color='w', weight='bold', 
                    fontsize=12, ha='center', va='center')       

    
    #Plotando as previsões
    if pred:
        ax_pred = plt.subplot(gs[pred_grid])
        plt.axis("off")

        pred_list = list(pred.items())
        pred_list = sorted(pred_list, key = lambda x: x[1],reverse = True)    
        for idx, (pred_type, pred_prob) in enumerate(pred_list):
            pred_box = patches.Rectangle(
                (0,0.8-0.2*idx),
                0.5,
                0.2,
                fc = type_dict[pred_type]["color"],
                ec = "#FFFFFF"            
            )
            ax_pred.add_patch(pred_box)
            ax_pred.annotate(type_dict[pred_type]["label"], (0.25, 0.9-0.2*idx), color='#FFFFFF', weight='bold', 
                        fontsize=12, ha='center', va='center')       
            ax_pred.annotate("{:.0%}".format(pred_prob), (0.75, 0.9-0.2*idx), color='#000000', weight='bold', 
                        fontsize=16, ha='center', va='center')   


    ###
    # Saves into correct if the biggest guess is correct
    ###
    
    if save:
        correct_path = os.path.join(save_path,"correct")
        wrong_path = os.path.join(save_path,"wrong")
        if not os.path.exists(correct_path):
            os.makedirs(correct_path)
        if not os.path.exists(wrong_path):
            os.makedirs(wrong_path)      
        if type_1 == pred_list[0][0]:
            save_file = os.path.join(correct_path,save)
        elif type_2 == pred_list[0][0]: 
            save_file = os.path.join(correct_path,save)            
        else:
            save_file = os.path.join(wrong_path,save)
        fig.savefig(save_file) 