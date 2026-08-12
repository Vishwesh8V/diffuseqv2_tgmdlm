# Inference Pipeline

**step1:** Take the best checkpoint (f\_`\theta`{=tex}\^\*)

**step2:** Take validation data of batch size (B)

\[ (x_b,y_b)\_{b=1}\^{B} \]

\[ Z_0
`\in `{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}L`\times `{=tex}d} \]

where

-   (L `\rightarrow`{=tex}) length of seq
-   (d `\rightarrow`{=tex}) dim

**step3:** get

\[
z_t^i=`\sqrt{\bar{\alpha}_t^i}`{=tex},z_0^i+`\sqrt{1-\bar{\alpha}_t^i}`{=tex},`\epsilon`{=tex}\^i
\]

and token-wise log SNR

\[
`\lambda`{=tex}\_t\^i=`\log`{=tex}`\frac{\bar{\alpha}_t^i}{1-\bar{\alpha}_t^i}`{=tex}
\]

for (t`\in`{=tex}{2000,`\ldots`{=tex},1})

**step4:** Predict clean latents

\[
`\hat{z}`{=tex}*{0,t}=f*`\theta`{=tex}(z_t,h_x,t,`\lambda`{=tex}\_t)`\in`{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}L`\times `{=tex}d}
\]

**step5:** Compute loss

\[
`\ell`{=tex}*{b,t,i}=`\frac{1}{d}`{=tex}`\left`{=tex}\|`\hat{z}`{=tex}*{0,b,t}^{,i}-z\_{0,b}^{,i}`\right`{=tex}\|\_2\^2
`\in`{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}T`\times `{=tex}L} \]

**step6:** Average across valid samples for each (i)

\[ E\_{t,i}=`\frac{\sum_{b=1}^{B}\ell_{b,t,i}}{|B|}`{=tex}
`\in`{=tex}`\mathbb{R}`{=tex}\^{T`\times `{=tex}L} \]

for position (i), loss curve

\[ C_i={(`\lambda`{=tex}*t\^i,E*{t,i})}\_{t=1}\^{T} \]

**step7:** Plot loss v/s logSNR

\[ x=`\lambda`{=tex}*t\^i,`\qquad `{=tex}y=E*{i,t} \]

During reverse denoising

\[
`\lambda`{=tex}`\uparrow`{=tex},`\qquad `{=tex}E\_{i,t}`\downarrow`{=tex}
\]

⇒ Fit a curve on it

**step8:** Calculate slope

\[ `\frac{dE_i}{d\lambda}`{=tex} \]

**step9:** Construct

\[ a_i(`\lambda`{=tex})=-e\^`\lambda`{=tex}`\frac{dE_i}{d\lambda}`{=tex}
\]

Compute node density

\[ r_i(`\lambda`{=tex})=`\sqrt{a_i(\lambda)}`{=tex} \]

**step10:** Integrate the density

\[
G_i(`\lambda`{=tex})=`\int`{=tex}*{`\lambda`{=tex}*{`\min`{=tex}}}\^{`\lambda`{=tex}}a_i(u),du
\]

Normalize

\[
`\bar`{=tex}{G}\_i(`\lambda`{=tex})=`\frac{G_i(\lambda)}{G_i(\lambda_{\max})}`{=tex}
\]

Then

\[ `\bar`{=tex}{G}*i(`\lambda`{=tex}*{`\min`{=tex}})=0,`\qquad`{=tex}
`\bar`{=tex}{G}*i(`\lambda`{=tex}*{`\max`{=tex}})=1 \]

**step11:** Select (K)-nodes

for (K)-retained nodes

\[ q_r=`\frac{r}{K-1}`{=tex},`\qquad `{=tex}r=0,`\ldots`{=tex},K-1 \]

Example: for (K=5)

\[ q=\[0,;0.25,;0.5,;0.75,;1\] \]

Invert Cumm-fund

\[ `\lambda`{=tex}\_{r,i}\^{\*}=`\bar`{=tex}{G}\_i\^{-1}(q_r) \]

This places nodes uniformly in cumm. monitor mass.

**step12:** Map selected logSNR values to dense time steps

for each selected (`\lambda`{=tex}\_{r,i}\^{\*})

\[ J\_{r,i} = `\arg`{=tex}`\min`{=tex}\_t `\left`{=tex}\|
`\lambda`{=tex}*t\^i-`\lambda`{=tex}*{r,i}\^{\*} `\right`{=tex}\| \]

Then construct

\[ J`\in`{=tex}`\mathbb{R}`{=tex}\^{K`\times `{=tex}L} \]
