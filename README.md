# Complementary Script

## Problem Significance

### Problem Setting and Motivation

- This project focuses on the problem of time series prediction.
- Given a data set of $n$ observations $\mathcal{D}$ the goal of time series prediction is to try and best predict a values of the phenomena we are modelling at unobserved times.
- The idea of studying time series prediction came from a research group from the Gatton campus, lead by Andries Potgieter, analysing crop growth from previous seasons to forecast when certain phenological stages will take place in the current harvest.
- Originally, Potgieter's team surveyed a number of different parameteric models to forecast crop growth. However, the parameteric models we serverely limited in their ability to inform when key phenological stages would take place. After seeing the success of applying GPs to other remote sensing tasks, the team investigated the use of GPs in their own research to find that they could produce much higher resolution predictions from which they could infer a far richer phenological timeline.

- The focus of my thesis was to study time series prediction, in particular gaussian processes. The idea to study this model came from one of the research groups down the Gatton on campus led by Dr Potgeiter who is analysing crop growth from previous seasons to forecast when certain phenological stages will take place in the current harvest.
- Originally, Potgieter's team surveyed a number of different parameteric models to forecast crop growth. However, the parameteric models we serverely limited in their ability to inform when key phenological stages would take place. After seeing the success of applying GPs to other remote sensing tasks, the team investigated the use of GPs in their own research to find that they could produce much higher resolution predictions from which they could infer a far richer phenological timeline.

## Gaussian Processes

### Introduction to Gaussian Processes

- A Gaussian Process (GP) is a collection of random variables with index set $I$, such that every finite subset of random variables has a joint Gaussian distribution and are completely characterized by a mean function $m : X \to \mathbb{R}$ and a kernel $k : X \times X \to \mathbb{R}$.
- The aim of GPs is to find a suitable mean function, $m$, for which we can then predict inputs from outside the observed values, $\bm{X_{\star}}$. This requires an understanding of the function $f$.

- Okay, so what exactly are Gaussian Process? Well, just very briefly, Gaussian Process it's a collection of random variables, with some sort of index that when you take any finite subset of these random variables it forms a joint gaussian distribution. They can be completely characterized by a mean function $m : X \to \mathbb{R}$ and a kernel $k : X \times X \to \mathbb{R}$. The mean function is what we're going to be using to make predictions, and is what is going to be used to model the function underpinning our phenomina and this covariance or kernel function you can really think about as just providing some notion of similarity between points.

### Predictions

- Using the assumption that our data can be modelled as a GP, we can model the relationship between the observed and unobserved values as a multivariate normal distribution.
- This distribution here represents the prior distribution, before we have made any observations which is why a mean of zero is used. We construct the covariance matrix by taking pair wise kernel evaluations between inputs of both the observed and unobserved values.
- As with most Machine Learning tasks, we don't usually have access to exact values of the underpinning function we are try to model (in this case f) but a noisy version thereof (is this case y). We try and account for this within our GP model by adding the variance of the labels from the data set to the diagonal elements to this top left hand sub matrix which corresponds to the covariances between feature vectors from the data set.
- Using some well known probability theory, we can compute a posterior to obtain expressions for both the mean and covariance of these novel points.

- Okay so how do we actually go about making predictions? Well under the assumption that our data can be modeled as Gaussian Process, the points that we'd like to predict at and our data should form a joint gaussian distribution.
- In the absence of any observations we don't know how phenomena will behave, so we just usually give it a mean of zero.
- However, we can still construct our covariance matrix by just taking pairwise kernel evaluations between novel points and points from our dataset.
- But, as with any machine learning task, we usually don't have access to exact values of underpinning function we're trying to model.
- In this case this would be F, but some some noisy version thereof it in this case it's each of these $\bm{y}$ from our data and we try to  take this into account by adding the label variance to these diagonal elements to the top left hand sub matrix which corresponds to the covariance between feature vectors from our data set.

### Unoptimized GPR

- _Present Algorithm and highlight slow parts_

## Nystrom

- One technique to speed up the computation of $\bm{K_{XX}}$ is to use a Nystrom approximation.
- The Nystrom method we seek a matrix $\bm{Q}\in \mathbb{R}^{n \times k}$ that satisfies ${|| \bm{A} - \bm{Q} \bm{Q}^{\ast} \bm{A} ||}_{F} \leq \varepsilon$, where $\bm{A} \in \mathbb{R}^{n \times n}$ is positive semi definite matrix, to form the rank$-k$ approximation _show approximation_.
- A matrix $\bm{Q}$ that satisfies the above conditions can be built using through a very popular column sampling technique.
- As the name suggests, the matrix $\bm{Q}$ essentially samples and rescales $k << n$ columns from $\bm{A}$ using a probability distribution across the columns of $\bm{A}$.
- The reason why this column sampling technique works at a high level is, indicative from this first line, we project the column space of $\bm{Q}$ onto $\bm{A}$ so that, if you've sampled you're columns right, the range of $\bm{Q}$ should resemble that of $\bm{A}$, so that when you do this projection we should hopefully end up with a good rank $k$ approximation of $\bm{A}$.

## RFF

- The other technique investigated to speed up the computation of $\bm{K_{XX}}$ is the Random Fourier Feature (RFF) approximation.
- The main idea is instead of using a kernel function to implicitly lift data into a higher dimensional feature space, an explicit feature map $\varphi : \mathbb{R}^d \to \mathbb{R}^D$ could be used to approximate $k$ as $k \left( \bm{x} , \bm{y} \right) = \langle \Phi (\bm{x}) , \Phi (\bm{y}) \rangle_{\mathbb{R}^N} \simeq \langle \varphi (\bm{x}) , \varphi (\bm{y}) \rangle_{\mathbb{R}^D}$ where $D$ is chosen so that $n \gg  D$. Once $\varphi (\bm{x}_i)$ has been computed for each $\bm{x}_i$, every entry of the Gram matrix can be swiftly approximated as $\bm{K}_{ij} = \bm{K}_{ji} \simeq \langle \varphi (\bm{x}_i) , \varphi (\bm{y}_j) \rangle_{\mathbb{R}^D}$.
- The RFF technique hinges on Bochners theorem which characterises positive definite functions (namely kernels) and states that any positive definite functions can be represented as _insert integral_ where $\mu_k$ is a positive finite measure on the frequencies of $\bm{\omega}$.
- Can then be approximated via the following Monte Carlo estimate ...

## Krylov Subspace Methods

- Finally we used Krylov subspace techniques to replace the Cholesky decompositions from the Gaussian process algorithm.
- So when we're solving linear system we want to find a vector $\bm{x}^{\ast}$ that satisfies this system of equations $\bm{A} \bm{x^{\star}} = \bm{b}$
- From the cayley hamilton theorem, we know that $\bm{x}^{\ast}$ belongs to the Krylov subspace of order $n$ plus your initial guess, where the Krylov subspace of order $k$ is just this thing here, $\bm{A}$ to some power ranging between $0$ and $k-1$ applied to your initial residual.
- The rationale behind these methods is that you slowly build up the order of your Krylov subspace with each iterate belonging to the next subspace and where additional constraints are employed to guide each estimate toward an exact solution.
- Different assumptions on $\bm{A}$ and constraints leads to different flavours of Krylov subspace methods.
- The particular Krylov subspace methods used in this thesis where the conjugate gradient (CG for short) and minimum residual methods (MINRES for short)
- The CG method imposes additional constraints so that the energy norm is minimized against the $k^{th}$ iterate and assumes $\bm{A}$ is PSD
- MINRES, on the other hand, imposes additional constraints so that Euclidean distance of $A x_k - b$ is minimized.

## Nystrom

- Okay so now looking at some results for kernel matrix approximation from our Nystrom techniques we found that some of the more sophisticated sampling distributions provided far better results for kernel matrices with non-uniform spectrum.
- So here we can see that some of the more intelligent sampling techniques such as as the red ridge leverage score sampler are much more performant over the black uniform sampler at least for the 3D spatial network dataset whose kernel spectrum was highly non-uniform.
- As we move into datasets with slightly more uniform kernel spectrums, the errors of these sampling distributions bunch up and the sophisticated samplers start losing their edge
- And in the most extreme cases, when the spectrum is almost completely uniform, all the methods perform more or less the same.

## RFF

- Surprisingly, when looking at the various RFF methods, the sophisticated techniques used to construct the transformation did not provide much of an advantage over basic methods, despite their accalimed theoretical bounds from literature.
- So this is what the errors looked like when different RFF techniques across every dataset

## Nystrom and RFF

- The Nystrom family is superior in lowering the Frobenius error in the approximations it produces, while the RFF methods provide approximations with smaller infinity errors
- This makes sense since much of the theory in the Nystrom methods was aimed toward lowering Frobenius errors, while RFF methods are more focused on lowering infinity errors.

- Well so far we've just been looking at how well RFF and Nystrom can construct a kernel matrix but the question reamins, which is better for GP prediction, since that's what we really care about at the end of the day.
- Generally speaking, the RFF method delivered the best predictions with smaller time budgets. For this reason, we think RFF method would is probably best to use in practice since if you're using an inexact method for kernel matrix production, it most likely means you don't have a great deal of time to work with which so that producing the best approximation within short timeframe is a very nice quality

## Krylov Comparison

- From our findings, the MINRES method performs just as well and, occasionally, even better than CG for regression tasks, even when paired with RFF.
- This may come as a surprise to some as we can provided better theoretical bounds in terms of error and convergence for CG. So why does MINRES work better here?
- So here the mean square error is used to determine the quality of prediction for regression tasks. The true mean square error is computed as follows.
- Why is this important, well, each iterate of the MINRES methods minimizes the empirical version of the MSE up to a noise value of sigma
- So the success of MINRES comes from this ability to more directly lower the Euclidean distance between predictions and outputs

## Moving Forward

- From what my supervisor and I could find, this is probably one of the most extensive experimental analysis conducted to determine which of methods delivered the best results, empirically speaking.
- Some of the findings discovered in this thesis have, to our knowledge, not been reported elsewhere. As a result, we intend to publish these findings given their obvious appeal to the wider scientific computing community.
- In terms of research, many of the datasets used in this thesis were far too small to observe the asymptotic benefits of using approximation methods. It would be interesting to determine how well these techniques scale on very large datasets. In particular, our discoveries are likely to benefit the agricultural sector as more data is collected over the forthcoming years to perform crop analysis on.
- Another, direction future research could be taken is the application of the approximation techniques applied to multi-output or multi-task Gaussian process models. In many machine learning scenarios we may want to predict multiple outputs using the same set of inputs. As an example, remote sensing researchers attempt to predict the intensity of multiple bands of light reflecting off farm land to give an indication of crop growth. Different versions of the Gaussian processes algorithm exist to predict multiple outputs simultaneously; however, they also suffer from the same bottlenecks as single output Gaussian processes. It would be intriging to learn whether the approximation techniques studied in this thesis could potentially improve the prediction time of multi-task Gaussian Processes.

- The question reamins which is better for GP prediction, lower relative error or absolute error (or perhaps some combination of the two)? This will ultimately determined which of the two techniques are more useful in Machine Learning applications.
- Recall, the other bottle neck in the GPR algorithm was the Cholesky decomposition.
- The Cholesky decomposition is used twice to compute expressions in the mean and covariance that involve $\left[ \bm{K_{XX}} + \sigma_n^2 \mathbb{I}_{n \times n} \right]^{-1}$. When used in conjunction with some sort of triangular solve will produce exact solutions. However using a decomposition Cholesky would seem like a rather extravagant choice to solve these linear systems for two reasons. To start this requires a hefty $O (n^3)$ run time, making it the dominate time sink by a long mile. Moreover, when paired with an inexact Gram matrix construct method, it seems counter intutive to then use an exact linear solver straight after.
- Instead, inexact iterative linear solvers can be used in place of Cholesky. The two solvers we are interested in are the {\it Conjugate Gradient} (CG) and {\it Minimum Residual} (MINRES) methods used for solving positive semi-definite and symmeteric indefinite systems respectively.
- While MINRES can be applied to a wider class of linear systems, in certain scenarios it may perform better than the more widely used CG method. It will be interesting to see which of the two produced better predictions in the context of GPs.
