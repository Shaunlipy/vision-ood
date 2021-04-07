import pandas as pd
import numpy as np
np.random.seed(0)

def print_unique(df):
    print(np.unique(df[1], return_counts=True))

file_name = '/Volumes/CT500/Researches/Attention_OOD/data/isic/isic_train_0.txt'

df_train = pd.read_csv(file_name, header=None)
print_unique(df_train)

file_name = '/Volumes/CT500/Researches/Attention_OOD/data/isic/isic_val_0.txt'

df_val = pd.read_csv(file_name, header=None)
print_unique(df_val)

file_name = '/Volumes/CT500/Researches/Attention_OOD/data/isic/isic_unseen_0.txt'

df_unseen = pd.read_csv(file_name, header=None)
print_unique(df_unseen)

df = pd.concat([df_train, df_val, df_unseen])
print_unique(df)

file_keep = 1000
val = int(0.1 * file_keep)

df_res = None
df_dict = {}
for i in range(8):
    temp = df[df[1]==i]
    if len(temp) > file_keep:
        ids = np.random.choice(len(temp), file_keep, replace=False)
    else:
        ids = list(range(len(temp)))
    np.random.shuffle(ids)
    id_v = ids[:val]
    id_t = ids[val:]
    df_dict[f'{i}_t'] = temp.iloc[id_t]
    df_dict[f'{i}_v'] = temp.iloc[id_v]
    if df_res is None:
        df_res = temp.iloc[ids]
    else:
        df_res = pd.concat([df_res, temp.iloc[ids]])

print_unique(df_res)

import os
new_dir = '/Volumes/CT500/Researches/Attention_OOD/data/isic_new'
if not os.path.exists(new_dir):
    os.makedirs(new_dir, exist_ok=True)

for i in range(8):
    df_u = pd.concat([df_dict[f'{i}_t'], df_dict[f'{i}_v']])
    df_t = None
    df_v = None
    for j in range(8):
        if i == j:
            continue
        if df_t is None:
            df_t = df_dict[f'{j}_t']
            df_v = df_dict[f'{j}_v']
        else:
            df_t = pd.concat([df_t, df_dict[f'{j}_t']])
            df_v = pd.concat([df_v, df_dict[f'{j}_v']])
    df_u.to_csv(f'{new_dir}/isic_unseen_{i}.txt', index=False, header=False)
    df_t.to_csv(f'{new_dir}/isic_train_{i}.txt', index=False, header=False)
    df_v.to_csv(f'{new_dir}/isic_val_{i}.txt', index = False, header=False)
