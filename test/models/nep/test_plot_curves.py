import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/lightning_logs/version_4/metrics.csv")
print(df)
plt.plot([*range(len(df.loc[:, "grad/raw_norm"].tolist()))], 
         df.loc[:, "grad/raw_norm"].tolist(),
         alpha=0.7)
plt.plot([*range(len(df.loc[:, "grad/fixed_threshold"].tolist()))],
          df.loc[:, "grad/fixed_threshold"].tolist(),
          alpha=0.7)
plt.savefig("./curves.png")
