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


# def plot_evaluation(label,recall, precision, mean_f1, type_dict = load_type_dict()):
# #     y_pred = np.argmax(y_pred,axis=1)+1
# #     y_true = np.argmax(y_true,axis=1)+1    
# #     #Evaluate model metrics over input data
# #     recall = recall_score(y_true, y_pred, average=None)
# #     precision = precision_score(y_true, y_pred, average=None)
# #     accuracy = accuracy_score(y_true, y_pred)
    
#     #Create grid for plotting
#     fig = plt.figure(figsize=(8.8,5))
#     gs = gridspec.GridSpec(2,1, height_ratios = (1,9))
       
#     #Plotting model-level metrics
#     ax = plt.subplot(gs[0])     
#     ax.axis("off")    
#     ax.annotate("{} Average F1 Score = {:.0%}".format(label, mean_f1), (0.5, 0.5), color='#000000', 
#                 fontsize=18, ha='center', va='center')     
        
#     #Ploting class-level metrics
#     ax = plt.subplot(gs[1]) 
#     ax.axis("off")    
    
#     #In some cases, there are no records of some classes (usually 3:Flying) Here, we fill
#     #up the missing classes with 'None' values.
#     unique_labels = np.arange(start=0, stop=19) #number of types in Pokemon
#     metrics = dict( (key, {"recall" : None, "precision" : None}) for key in range(0,18))
#     for key, v_recall, v_precision in zip(unique_labels, recall, precision):
#         metrics[key]["recall"] = v_recall
#         metrics[key]["precision"] = v_precision

#     #Writing the headers of the class table
#     ax.annotate("Precision", (0.27, 19/20), color='#000000', weight='bold', 
#                 fontsize=12, ha='center', va='center')     
#     ax.annotate("Recall", (0.4, 19/20), color='#000000', weight='bold', 
#                 fontsize=12, ha='center', va='center')        
#     ax.annotate("Precision", (0.77, 19/20), color='#000000', weight='bold', 
#                 fontsize=12, ha='center', va='center')        
#     ax.annotate("Recall", (0.9, 19/20), color='#000000', weight='bold', 
#                 fontsize=12, ha='center', va='center')            
    
#     #Writing the metrics for each class
#     for i, (pkm_type, metric) in enumerate(metrics.items()):
#         column = int(i/9)
#         row = i % 9 + 1
#         left = column*0.5
#         top = 1.0-(row+1)*1/10
#         type_box = patches.Rectangle(
#             (left,top),
#             0.2,
#             1/9,
#             fc = type_dict[pkm_type]["color"],
#             ec = "#FFFFFF"            
#         ) 
#         ax.add_patch(type_box)
#         ax.annotate(type_dict[pkm_type]["label"], (left+0.1, top+1/20), color='#FFFFFF', weight='bold', 
#                     fontsize=12, ha='center', va='center')  
#         #Precision
#         if metric["precision"] is not None:
#             ax.annotate("{:.0%}".format(metric["precision"]), (left+0.27, top+1/20), color='#000000', 
#                         fontsize=14, ha='center', va='center')    
#         #Recall
#         if metric["recall"] is not None:
#             ax.annotate("{:.0%}".format(metric["recall"]), (left+0.40, top+1/20), color='#000000', 
#                         fontsize=14, ha='center', va='center') 


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

    nprec = len(classes)
    nrec = len(classes)
    nf1 = len(classes)

    mean_Precision = 0
    mean_Recall = 0
    mean_F1 = 0

    for t in range(0, len(classes)):
        if(not np.isnan(precision[t])):
            mean_Precision += precision[t]
        else:
            nprec -= 1
        
        if(not np.isnan(recall[t])):
            mean_Recall += recall[t]
        else:
            nrec -= 1
            
        if(not np.isnan(F1[t])):
            mean_F1 += F1[t]
        else:
            nf1 -= 1
            
    mean_Precision /= nprec
    mean_Recall    /= nrec
    mean_F1        /= nf1

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