from typing import List, Union

import pandas as pd
from sklearn.model_selection import KFold


def split_data(data: Union[List, pd.DataFrame], n_split: int):
    if ".csv" in data:
        data = pd.DataFrame(data)

    kfold = KFold(n_splits=n_split, random_state=42, shuffle=True)
    for i, (train_index, val_index) in enumerate(kfold.split(data)):
        data.loc[val_index, "fold"] = i
    return data
