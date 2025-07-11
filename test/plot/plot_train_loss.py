import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/version_1/metrics.csv")

ys_lst = df.loc[:, "train_mse_epoch"].dropna(axis=0).to_numpy()
ys_lst = np.sqrt(ys_lst)
xs_lst = [*range(len(ys_lst))]
plt.plot(xs_lst, ys_lst)
plt.yscale("log")  # 设置 y 轴为对数坐标
plt.ylim(10**(-2), 1)
plt.xlim(0, 500)
plt.ylabel("RMSE (Energy + Force)", fontdict={"fontsize": 16})
plt.xlabel("Epoch", fontdict={"fontsize": 16})

plt.hlines(y=np.min(ys_lst), xmin=0, xmax=500,
           ls='--', colors="red")
plt.text(x=100, y=0.04, s="Minimum RMSE = {0:.3f}".format(np.min(ys_lst)))
plt.savefig("./epoch_loss.png")