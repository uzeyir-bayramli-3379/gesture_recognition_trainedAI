import pandas as pd
import numpy as np
datapd= pd.read_csv('gesture_data.csv')
arr=[]
labels=[]
T=30
overlap=10
for seq_id , group_df in datapd.groupby('seq_id'):
    for f in range(0,len(group_df),T-overlap):
        if f+T<=len(group_df):
            arr.append(group_df[f:f+T].drop(columns=['seq_id','frame','label']))
        else:
            arr.append(group_df[-T:].drop(columns=['seq_id','frame','label']))
        labels.append(int(group_df['label'].iloc[0]))

print(arr[:3])

arr=np.array(arr)
labels=np.array(labels)
print(arr.shape,labels.shape)
