`NNMTP`:
--------
$$\left\{ \begin{matrix} 
bilist \\
bnumneigh \\
bfirstneigh \\
brcs \\
btypes \\
bnghost \\
\end{matrix} \right\} \rightarrow 
DescriptorMTP \rightarrow 
bdescriptor \rightarrow
BatchNorm1D \rightarrow
\left\{ \begin{matrix} 
FittingNet\_type_0 \quad (type\_bias_0)\\
FittingNet\_type_1 \quad (type\_bias_1)\\
\vdots \\
FittingNet\_type_N \quad (type\_bias_N)
\end{matrix} \right\} \rightarrow
\begin{cases}
etot \\
fi   \\
v
\end{cases}$$


# 1. `RMSE` of `Energy`, `Force` and `Virial`
<font color="steelblue" size="5">

$$RMSE(E)^2 = \frac{1}{K}\sum_{k=1}^{K}{(\frac{E^{mtp}(cfg_k; \theta)}{N^{(k)}} - \frac{E^{qm}(cfg_k)}{N^{(k)}})^2}$$

$$RMSE(\vec{f})^2 = \frac{1}{K}\sum_{k=1}^K{\frac{1}{3N^{(k)}} \sum_{i=1}^{N_k}{|\vec{f}_i^{mtp}(cfg_k;\theta) - \vec{f}_i^{qm}(cfg_k)|^2}}$$

$$RMSE(W)^2 = \frac{1}{K}\sum_{k=1}^{K}\frac{1}{9}|W^{mtp}(cfg_k; \theta) - W^{qm}(cfg_k)|^2$$

</font>


# 2. Total train loss
<font color="steelblue" size="5">

$$L_E = \frac{w_e}{N_k} (E^{mtp}(cfg_k;\theta) - E^{qm}(cfg_k))^2$$

$$L_f = \frac{w_f}{3N_k} \sum_{i=1}^{N_k}(f^{mtp}_{i,\alpha}(cfg_k; \theta) - f^{qm}_{i, \alpha}(cfg_k))^2$$

$$L_W = \frac{w_W}{9N} \sum_{\alpha, \beta}{(W^{mtp}_{\alpha, \beta}(cfg_k; \theta) - W^{qm}_{\alpha, \beta}(cfg_k))^2}$$

</font>