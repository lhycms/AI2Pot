import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSSe_sro_bader/output/total.csv")

### 1. 绘制图像
### 0. 全局设置
# 0.1. 字体设置
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# 0.2. 刻度线朝内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))

plt.scatter(df["41_16_sro"], df["41_charge"], 
            s=300,
            alpha=0.7,
            marker="P",
            c="yellowgreen",
            label="$\mathbf{SRO^{1}_{Nb,S}}$")
plt.scatter(df["75_16_sro"], df["75_charge"],
            s=300,
            alpha=0.7,
            marker="X",
            c="lightpink",
            label="$\mathbf{SRO^{1}_{Re,S}}$")

# 1. Retouch the xlabel, ylabel
plt.xlabel("$\mathbf{SRO}$", fontsize=35, 
                fontweight="bold"
                )
plt.ylabel("Bader charge (e)", fontsize=35, 
        fontweight="bold"
        )

# 2. Retouch the ticks of x-axis/y-axis
plt.xticks(fontsize=30, 
        fontweight="bold"
        )
plt.yticks(fontsize=30, 
        fontweight="bold"
        )

# 4. 刻度线的粗细
plt.tick_params(
        width=2,        # 刻度线的粗细
        length=5,       # 刻度线的长短
        #labelsize=28   # 刻度线的字体大小
        )

# 5. 设置坐标轴的粗细
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5);###设置右边坐标轴的粗细

# 7. 添加水平线
#plt.axhline(y=-763.5, ls="--",c="limegreen", linewidth=3)#添加水平直线

legend_font = {"size" : 30, 
                "weight": "bold"
                }
plt.legend(loc=(0.68, 0.75),
        prop=legend_font,
        frameon=True,
        handletextpad=0.2,
        handlelength=0.5)

font_dict = {"fontsize": 30,
            "fontweight": "bold"
            }

plt.savefig("./test.png", dpi=800, bbox_inches="tight")
