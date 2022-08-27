import pandas as pd
import os
from main import data_list

mi = pd.MultiIndex.from_product([data_list, ["Accuracy", "AUC", "F1 Measure", "Mean CV on Train"]])
df = pd.DataFrame(columns=mi)

for i in os.listdir("results"):
    for j in data_list:
        if i.startswith(f"{j}_"):
            temp_df = pd.read_csv(f"results/{i}", header=0, index_col=0)
            df.loc[i[len(j) + 1:-4], (j, slice(None))] = temp_df.loc["value", :].to_numpy()

df.to_csv("compiled.csv", header=True, index=True)
