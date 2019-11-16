import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
from io import StringIO
import time


def thresh_doc(df_result,true_label,pred_label,conf_label,doc_name):

    groups=df_result[pred_label].tolist()
    groups=list(set(groups))
    thresholds1=[]
    accuracies=[]
    real=[]
    for group in groups:
        thresholds2=[]
        dfx=df_result[df_result[pred_label]==group]
        acc=accuracy_score(dfx[pred_label],dfx[true_label])
        
        if acc<0.8:
            size=0.75
        elif 0.8<=acc<0.9:
            size=0.90
        else:
            size=1
            thresholds1.append(0)
            accuracies.append(acc)
            real.append(group)
            
            
        for i in np.linspace(0,1,10):
            
            dfz=dfx[dfx[conf_label]>=i]
            if len(dfz)>(size*len(dfx)):
                thresholds2.append((i,accuracy_score(dfz[pred_label],dfz[true_label])))
        if len(thresholds2)!=0:
            real.append(group)
            thresholds1.append(max(thresholds2, key=lambda k: k[1])[0])
            accuracies.append(max(thresholds2, key=lambda k: k[1])[1])
    thresh=pd.DataFrame(columns=[pred_label,'threshold','accuracy'])
    thresh[pred_label]=real
    thresh['threshold']=thresholds1
    thresh['accuracy']=accuracies
    thresh.to_excel(doc_name+'.xlsx')
    print('Threshold document saved')