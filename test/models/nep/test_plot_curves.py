import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/lightning_logs/version_33/metrics.csv")
print(df)
#plt.plot([*range(len(df.loc[:, "grad/raw_norm_step"].tolist()))], 
#         df.loc[:, "grad/raw_norm_step"].tolist(),
#         alpha=0.7)
#plt.plot([*range(len(df.loc[:, "grad/clip_threshold_step"].tolist()))],
#          df.loc[:, "grad/clip_threshold_step"].tolist(),
#          alpha=0.7)
#plt.savefig("./curves.png")
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 可选：显示所有列（防止列被省略）
print(df.loc[df["train/f_rmse_epoch"].notna(), ["epoch", "train/e_rmse_epoch"]])