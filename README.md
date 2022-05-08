# Complementary Script

## Problem Significance

### Problem Setting and Motivation

- This project focuses on the problem of time series prediction.
- Given a data set of $n$ observations $\mathcal{D}$ where each observations has a corressponding label the goal of time series prediction is to try and best predict a values of the phenomena we are modelling at unobserved times.
- The idea of studying time series prediction came from a research group from the Gatton campus, lead by Andries Potgieter, analysing crop growth from previous seasons to forecast when certain phenological stages will take place in the current harvest.
- Originally, Potgieter's team surveyed a number of different parameteric models to carry out forecasting. However, the parameteric models we serverely limited in their ability to inform when key phenological stages would take place. After seeing the success of applying GPs to other remote sensing tasks, the team investigated the use of GPs in their own research to find that they could produce much higher resolution predictions from which they could infer a far richer phenological timeline.

## Gaussian Processes

### Introduction to Gaussian Processes

- A Gaussian Process (GP) is a collection of random variables with index set $I$, such that every finite subset of random variables has a joint Gaussian distribution and are completely characterized by a mean function $m : X \to \mathbb{R}$ and a kernel $k : X \times X \to \mathbb{R}$.
- The aim of GPs is to find a suitable mean function, $m$, for which we can then predict inputs from outside the observed values, $\bm{X_{\star}}$. This requires an understanding of the function $f$.

### Predictions

- Using the assumption that our data can be modelled as a GP, we can model the relationship between the observed and unobserved values as a multivariate normal distribution.
- This distribution here represents the prior distribution, before we have made any observations which is why a mean of zero is used. We construct the covariance matrix by taking pair wise kernel evaluations between inputs of both the observed and unobserved values.
- As with most Machine Learning tasks, we don't usually have access to exact values of the underpinning function we are try to model (in this case f) but a noisy version thereof (is this case y). We try and account for this within our GP model by adding the variance of the labels from the data set to the diagonal elements to this top left hand sub matrix which corresponds to the covariances between feature vectors from the data set.
- Using some well known probability theory, we can compute a posterior to obtain expressions for both the mean and covariance of these novel points.

### Unoptimized GPR

- _Present Algorithm and highlight slow parts_

## Nystrom

- One technique to speed up the computation of $\bm{K_{XX}}$ is to use a Nystrom approximation.
- The Nystrom method we seek a matrix $\bm{Q}\in \mathbb{R}^{n \times k}$ that satisfies ${|| \bm{A} - \bm{Q} \bm{Q}^{\ast} \bm{A} ||}_{F} \leq \varepsilon$, where $\bm{A} \in \mathbb{R}^{n \times n}$ is positive semi definite matrix, to form the rank$-k$ approximation _show approximation_.
- A matrix $\bm{Q}$ that satisfies the above conditions can be built using through a very popular column sampling technique.
- As the name suggests, the matrix $\bm{Q}$ essentially samples and rescales $k << n$ columns from $\bm{A}$ using a probability distribution across the columns of $\bm{A}$.
- The reason why this column sampling technique works at a high level is, indicative from this first line, the Nystrom technique sort of projects the column space of $\bm{Q}$ onto $\bm{A}$ so that, if you've sampled you're columns right, the range of $\bm{Q}$ should resemble that of $\bm{A}$, so that when the column space of $\bm{Q}$ is projected onto $\bm{A}$ we should hopefully end up with a good rank $k$ approximation of $\bm{A}$.

## RFF
- The other technique investigated to speed up the computation of $\bm{K_{XX}}$ is the Random Fourier Feature (RFF) approximation.
- The main idea is instead of using a kernel function to implicitly lift data into a higher dimensional feature space, an explicit feature map $\varphi : \mathbb{R}^d \to \mathbb{R}^D$ could be used to approximate $k$ as $k \left( \bm{x} , \bm{y} \right) = \langle \Phi (\bm{x}) , \Phi (\bm{y}) \rangle_{\mathbb{R}^N} \simeq \langle \varphi (\bm{x}) , \varphi (\bm{y}) \rangle_{\mathbb{R}^D}$ where $D$ is chosen so that $n \gg  D$. Once $\varphi (\bm{x}_i)$ has been computed for each $\bm{x}_i$, every entry of the Gram matrix can be swiftly approximated as $\bm{K}_{ij} = \bm{K}_{ji} \simeq \langle \varphi (\bm{x}_i) , \varphi (\bm{y}_j) \rangle_{\mathbb{R}^D}$.
- The RFF technique hinges on Bochners theorem which characterises positive definite functions (namely kernels) and states that any positive definite functions can be represented as _insert integral_ where $\mu_k$ is a positive finite measure on the frequencies of $\bm{\omega}$.
- Can then be approximated via the following Monte Carlo estimate ...

## Krylov Subspace Methods

- When looking at Krylov Subspace Methods, or more generally speaking any sort of linear system solving, the name of the game is to find an $x$ that satifies this expression up here $A x^{\star} = b$.
- From algebra we know that the inverse of one minus an element within any field can be written as the sum of increasing powers. Coupling this with the Cayley Hamilton theorem it is fairly straight forward to show that an exact solution is in the linear span of various powers of $A$ applied to the initial residual, which is called the Krylov subspace, plus an initial guess, $x_0$.
- Of course these constraints are not enough to specify a unique solution on each iteration. Imposing different constraints on the linear system gives rise to the many Krylov subspace methods that we all know and love.
- The particular Krylov subspace methods used in this thesis where the conjugate gradient (CG for short) and minimum residual methods (MINRES for short)
- The CG method imposes additional constraints so that the energy norm is minimized against the $k^{th}$ residual where the energy norm is taken with respect to $A^{-1}$
- MINRES, on the other hand, imposes additional constraints so that Euclidean distance of $A x_k - b$ is minimized.

## Nystrom and RFF

- In general Rami and Recht's RFF method usually delivered the best predictions with smaller time budgets were the rls generally caught up and overtook it when a larger time budget was allowed.
- The data columns method, overall, yields the poorest predictions errors among the selected methods.
- The RFF method would probably be the best method to use in practice for two main reasons. The first reason is, if one was to select an inexact method for kernel matrix production, it likely means that they do not have a large time budget to begin with so that RFF's ability to produce the best approximation within shorter timeframes makes it highly desirable to use.

## Krylov Comparison

- From our findings, the MINRES method performs just as well and, occasionally, even better than CG. This becomes very apparent from the prediction errors produced from the abalone and 3DSN datasets
- In particular, the MINRES method seems to give better results even of the quadratic dataset. The quadratic dataset is an artificial dataset constructed using a Gaussian process with a quadratic mean function $x^2 + y^2$ and some amount of variance. The purpose of this dataset was to check if applied method was prone to overfitting. The fact that, overall, MINRES gave better error indicates that it is also more robust against overfitting compared to CG.

## Moving Forward

- We saw that the Nystrom technique is better at producing approximations of the Gram matrix, $\bm{K}_{\bm{XX}}$, with smaller {\it relative Frobenius errors} while the RFF technique is better at producing approximations with smaller {\it relative absolute errors}.
- The question reamins which is better for GP prediction, lower relative error or absolute error (or perhaps some combination of the two)? This will ultimately determined which of the two techniques are more useful in Machine Learning applications.
- Recall, the other bottle neck in the GPR algorithm was the Cholesky decomposition.
- The Cholesky decomposition is used twice to compute expressions in the mean and covariance that involve $\left[ \bm{K_{XX}} + \sigma_n^2 \mathbb{I}_{n \times n} \right]^{-1}$. When used in conjunction with some sort of triangular solve will produce exact solutions. However using a decomposition Cholesky would seem like a rather extravagant choice to solve these linear systems for two reasons. To start this requires a hefty $O (n^3)$ run time, making it the dominate time sink by a long mile. Moreover, when paired with an inexact Gram matrix construct method, it seems counter intutive to then use an exact linear solver straight after.
- Instead, inexact iterative linear solvers can be used in place of Cholesky. The two solvers we are interested in are the {\it Conjugate Gradient} (CG) and {\it Minimum Residual} (MINRES) methods used for solving positive semi-definite and symmeteric indefinite systems respectively.
- While MINRES can be applied to a wider class of linear systems, in certain scenarios it may perform better than the more widely used CG method. It will be interesting to see which of the two produced better predictions in the context of GPs.
