import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_csv("results/aggregated_features_folded_with_demographics.csv")
df = df[[col for col in df.columns 
        if ('Oddball' in col or 'Resting' in col or col == "Group")
]]

for name, g in df.groupby("Group"):
    print(f"\nGroup: {name}")

    desc = g.describe()

    cv = 100 * desc.loc["std"] / desc.loc["mean"]
    cv.name = "cv_percent"

    out = pd.concat([desc, cv.to_frame().T], axis=0)
    
    print(out.round(2))
    out.to_csv(f"results/PR01_summary_{name}.csv")
