import torch
import torch.nn as nn
import matplotlib.pyplot as plt


model = nn.Linear(12, 1)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                   gamma=0.98)
lr_start = 1e-4
e_wgts = []
e_wgt_start = 0.02
e_wgt_stop = 1

for i in range(100):
    optimizer.step()
    current_lr = optimizer.param_groups[0]["lr"]
    e_wgt = e_wgt_stop * (1 - current_lr / lr_start) + e_wgt_start * current_lr / lr_start
    e_wgts.append(e_wgt)
    schedular.step()


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

plt.figure(figsize=(12, 12))
plt.plot(e_wgts, lw=5)

ax = plt.gca()
#ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis='y')

# 1. Retouch the xlabel, ylabel
plt.xlabel("Epoch", fontsize=35, 
                fontweight="bold"
                )
plt.ylabel("Energy weight", fontsize=35, 
        fontweight="bold"
        )

# 2. Retouch the ticks of x-axis/y-axis
plt.xticks(fontsize=30, 
        fontweight="bold"
        )
plt.yticks(fontsize=30,
        fontweight="bold"
        )

# 3. y range
#plt.ylim([-7.075, -7.030])

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



plt.savefig("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/presentation/2024_11_26/e_weight.png",
            dpi=500, bbox_inches="tight")