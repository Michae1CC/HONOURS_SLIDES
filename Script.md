# Complementary Script

## Problem Significance

### Problem Setting and Motivation

- This project focuses on the problem of time series prediction.
- Given a data set of $n$ observations $\mathcal{D} = \left\{ \left( x_i , y_i \right) \right\}_{i=1}^{n}$, where each input $x_i \in \mathbb{R}_{>0}$ is a time value and $y_i \in \mathbb{R}$ is a output or experimental observation that acts a function of time, the goal of time series prediction is to try and best predict a value $y_{\star}$ at time $x_{\star}$.
- The idea of studying time series prediction came from a research group from the Gatton campus, lead by Andries Potgieter, analysing crop growth from previous seasons to forecast when certain phenological stages will take place in the current harvest.
- Originally, Potgieter's team surveyed a number of different parameteric models to carry out forecasting. However, the parameteric models we serverely limited in their ability to inform when key phenological stages would take place. After seeing the success of applying GPs to other remote sensing tasks, the team investigated the use of GPs in their own research to find that they could produce much higher resolution predictions from which they could infer a far richer phenological timeline.

## Gaussian Processes

### Introduction to Gaussian Processes

- A Gaussian Process (GP) is a collection of random variables with index set $I$, such that every finite subset of random variables has a joint Gaussian distribution and are completely characterized by a mean function $m : X \to \mathbb{R}$ and a kernel $k : X \times X \to \mathbb{R}$.

### Predictions

- The aim of GPs is to find a suitable mean function, $m$, for which we can then predict inputs from outside the observed values, $\bm{X_{\star}}$. This requires an understanding of the function $f$.
- Adopting the notation $\left( \bm{K}_{\bm{W} \bm{W}'} \right)_{i,j} \triangleq k \left( \bm{w}_i , \bm{w}_j' \right)$, when attempting to model our value function we usually do not have access to the value function itself but a noisy version thereof, $y = f(\bm{x}) + \varepsilon$ where $\varepsilon \sim \mathbb{N} (0, \sigma_n^2)$ meaning the prior on the noisy values becomes $\operatorname{cov} (\bm{y}) = \bm{K_{XX}} + \sigma_n^2 \bm{I}$.

### Unoptimized GPR

- _Present Algorithm and highlight slow parts_

## Nystrom

- One technique to speed up the computation of $\bm{K_{XX}}$ is to use a Nystrom approximation.
- The Nystrom method we seek a matrix $\bm{Q}\in \mathbb{R}^{n \times k}$ that satisfies ${|| \bm{A} - \bm{Q} \bm{Q}^{\ast} \bm{A} ||}_{F} \leq \varepsilon$, where $\bm{A} \in \mathbb{R}^{n \times n}$ is positive semi definite matrix, to form the rank$-k$ approximation _show approximation_.
- A matrix $\bm{Q}$ that satisfies the above conditions can be built using through a very popular column sampling technique.
- As the name suggests, the matrix $\bm{Q}$ essentially samples and rescales columns from $\bm{A}$ using a probability distribution across the columns of $\bm{A}$.
- When $\bm{Q}$ is constructed in this manner, it is usually referred to as a sketching matrix and denoted $\bm{S}$.

## RFF
- The other technique investigated to speed up the computation of $\bm{K_{XX}}$ is the Random Fourier Feature (RFF) approximation.
- The main idea is instead of using a kernel function to implicitly lift data into a higher dimensional feature space, an explicit feature map $\varphi : \mathbb{R}^d \to \mathbb{R}^D$ could be used to approximate $k$ as $k \left( \bm{x} , \bm{y} \right) = \langle \Phi (\bm{x}) , \Phi (\bm{y}) \rangle_{\mathbb{R}^N} \simeq \langle \varphi (\bm{x}) , \varphi (\bm{y}) \rangle_{\mathbb{R}^D}$ where $D$ is chosen so that $n \gg  D$. Once $\varphi (\bm{x}_i)$ has been computed for each $\bm{x}_i$, every entry of the Gram matrix can be swiftly approximated as $\bm{K}_{ij} = \bm{K}_{ji} \simeq \langle \varphi (\bm{x}_i) , \varphi (\bm{y}_j) \rangle_{\mathbb{R}^D}$.
- The RFF technique hinges on Bochners theorem which characterises positive definite functions (namely kernels) and states that any positive definite functions can be represented as _insert integral_ where $\mu_k$ is a positive finite measure on the frequencies of $\bm{\omega}$.
- Can then be approximated via the following Monte Carlo estimate ...

## Moving Forward

- We saw that the Nystrom technique is better at producing approximations of the Gram matrix, $\bm{K}_{\bm{XX}}$, with smaller {\it relative Frobenius errors} while the RFF technique is better at producing approximations with smaller {\it relative absolute errors}.
- The question reamins which is better for GP prediction, lower relative error or absolute error (or perhaps some combination of the two)? This will ultimately determined which of the two techniques are more useful in Machine Learning applications.
- Recall, the other bottle neck in the GPR algorithm was the Cholesky decomposition.
- This can be avoided by employing faster linear system solvers. The two solvers we are interested in are the {\it Conjugate Gradient} (CG) and {\it Minimum Residual} (MINRES) methods used for solving positive semi-definite and symmeteric indefinite systems respectively.
- While MINRES can be applied to a wider class of linear systems, in certain scenarios it may perform better than the more widely used CG method. It will be interesting to see which of the two produced better predictions in the context of GPs.
