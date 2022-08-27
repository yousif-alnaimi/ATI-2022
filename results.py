import pandas as pd
import os

df = pd.DataFrame(columns=["Accuracy", "AUC", "F1 Measure", "Mean CV on Train"])

for i in os.listdir("results"):
    temp_df = pd.read_csv(f"results/{i}", header=0, index_col=0)
    df.loc[i[:-4], :] = temp_df.loc["value", :]

df.to_csv("results/compiled.csv", header=True, index=True)
