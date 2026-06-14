import pandas as pd
import numpy as np
datapd= pd.read_csv('gesture_data.csv')
arr=[]
labels=[]
seq_ids=[]
seq10 = datapd['seq_id'] == 10
n10 = int(seq10.sum())
bounds10 = np.linspace(0, n10, 5).astype(int)
chunk_idx10 = np.zeros(n10, dtype=int)
for i in range(4):
    chunk_idx10[bounds10[i]:bounds10[i+1]] = i

seq11 = datapd['seq_id'] == 11
n11 = int(seq11.sum())
bounds11 = np.linspace(0, n11, 5).astype(int)
chunk_idx11 = np.zeros(n11, dtype=int)
for i in range(4):
    chunk_idx11[bounds11[i]:bounds11[i+1]] = i

datapd['split_id'] = datapd['seq_id'].astype(str)   # reset ONCE
datapd.loc[seq10, 'split_id'] = ['10_' + str(c) for c in chunk_idx10]
datapd.loc[seq11, 'split_id'] = ['11_' + str(c) for c in chunk_idx11]
T=30
overlap=10
for seq_id , group_df in datapd.groupby('split_id'):
    for f in range(0,len(group_df),T-overlap):
        if f+T<=len(group_df):
            arr.append(group_df[f:f+T].drop(columns=['seq_id','frame','label','split_id']))
        else:
            arr.append(group_df[-T:].drop(columns=['seq_id','frame','label','split_id']))
        labels.append(int(group_df['label'].iloc[0]))
        seq_ids.append((group_df['split_id'].iloc[0]))
#print(arr[-1])

arr=np.array(arr)
labels=np.array(labels)
seq_ids=np.array(seq_ids)
#print(arr.shape,labels.shape,seq_ids.shape)
#print(seq_ids)
