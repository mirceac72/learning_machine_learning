# Ridge regression

**Topic:** linear regression.

Regression algorithms construct an approximation of a function $f:\mathbb{R}^p \to \mathbb{R}$ from a set of observed pairs (features, response). Linear regression assumes that the function is a linear combination of the $p$ features plus a constant, and estimates its parameters by minimising the sum of squared approximation errors, for which reason it is also called least-squares regression. Ridge regression is least-squares regression with a penalty on the size of the coefficient vector, whose strength is set by a hyperparameter $\lambda > 0$. For every such $\lambda$ the estimator exists and is unique, even with more features than observations, and it is less sensitive than least squares to correlated features and to noise in the response. The derivation shows what the penalty does to each direction of the feature space, why the estimator must be applied to centred and scaled features, how $\lambda$ is chosen from the data, and where the method cannot help.

**Lesson plan.**

- *The problem and a running example.* The linear model and its noise assumptions, why the intercept is fitted by centring rather than penalised, and a five-observation example carried through every derivation.
- *Least squares and its instability.* The normal equations, the variance of the estimator, and why correlated features make individual coefficients poorly determined.
- *The ridge estimator.* The penalised objective, its unique solution for every $\lambda > 0$, why the features must be standardised, and invariance to rotation but not to scale.
- *The SVD view.* Ridge as a shrinkage of each principal direction by a factor set by its variance; effective degrees of freedom; the scale of $\lambda$.
- *Bias and variance.* The mean squared error as a function of $\lambda$, and the proof that some $\lambda > 0$ always improves on least squares.
- *Other routes to the same estimator.* Constrained least squares, a Gaussian prior, augmented data, the dual form, and weight decay and early stopping in gradient descent.
- *Choosing $\lambda$.* K-fold cross-validation, the leave-one-out score in closed form, and generalised cross-validation.
- *Practice.* Computing the fit, the workflow from raw data to prediction, software conventions, the situations in which ridge does not help, and a NumPy implementation with the procedure in pseudocode.

---

## Preliminaries

This section collects the results from linear algebra, calculus and probability that the lesson uses.

**Vectors and matrices.** Vectors are columns of real numbers. In regression, the features of one observation form a vector, and the feature vectors of $n$ observations, stacked as rows, form an $n \times p$ matrix. For $v \in \mathbb{R}^p$, $\|v\|^2 = v^\top v = \sum_j v_j^2$ is the squared Euclidean norm. For matrices, $(AB)^\top = B^\top A^\top$, and a matrix $M$ is *symmetric* when $M^\top = M$. $I$ denotes the identity matrix and $\mathbf{1}$ the vector of ones; their sizes follow from context. The *trace* $\mathrm{tr}(M) = \sum_i M_{ii}$ satisfies $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ whenever both products are defined. A square matrix $Q$ is *orthogonal* when $Q^\top Q = Q Q^\top = I$; multiplication by $Q$ preserves norms, since $\|Qv\|^2 = v^\top Q^\top Q v = \|v\|^2$.

**Gradients of linear and quadratic functions.** For a function $f : \mathbb{R}^p \to \mathbb{R}$, the *gradient* $\nabla f(\beta) \in \mathbb{R}^p$ is the vector of partial derivatives, with $k$-th entry $\partial f/\partial\beta_k$. At a point, the gradient is the direction in which the function increases fastest, and it is zero at a minimum. The first fact drives iterative minimisation, the second gives minimisers in closed form. The ridge objective is built from linear and quadratic functions of $\beta$.

*Linear.* Let $f(\beta) = a^\top\beta = \sum_j a_j\beta_j$ for a fixed $a \in \mathbb{R}^p$. Only the term $a_k\beta_k$ depends on $\beta_k$, so $\partial f/\partial\beta_k = a_k$ and

$$\nabla f(\beta) = a$$

*Quadratic.* Let $f(\beta) = \beta^\top M\beta = \sum_{j}\sum_{l}\beta_j M_{jl}\beta_l$ for a fixed symmetric $p \times p$ matrix $M$. The terms that contain $\beta_k$ are of three kinds: the diagonal term $M_{kk}\beta_k^2$; the terms $\beta_k M_{kl}\beta_l$ with $j = k$ and $l \neq k$; and the terms $\beta_j M_{jk}\beta_k$ with $l = k$ and $j \neq k$. Since $M_{jk} = M_{kj}$, the last two groups are equal, and together they sum to $2\beta_k\sum_{l \neq k}M_{kl}\beta_l$. Differentiating with respect to $\beta_k$,

$$\frac{\partial f}{\partial\beta_k} = 2M_{kk}\beta_k + 2\sum_{l \neq k}M_{kl}\beta_l = 2\sum_{l}M_{kl}\beta_l = 2\,(M\beta)_k$$

and collecting the $p$ entries into a vector,

$$\nabla f(\beta) = 2M\beta$$

*The two parts of the objective function.* Ridge regression finds the parameters that minimise an objective function with two aims: to keep the fitted values close to the observed values while penalising large parameter vectors. For fixed $y \in \mathbb{R}^n$ and $X \in \mathbb{R}^{n \times p}$, where $n$ is the number of observations and $p$ the number of features, expanding the square gives

$$\|y - X\beta\|^2 = (y - X\beta)^\top(y - X\beta) = y^\top y - 2\,(X^\top y)^\top\beta + \beta^\top(X^\top X)\beta$$

As a function of $\beta$, the right-hand side is a constant, a linear function with $a = X^\top y$, and a quadratic with the symmetric matrix $M = X^\top X$. Applying the two rules term by term,

$$\nabla_\beta \|y - X\beta\|^2 = -2X^\top y + 2X^\top X\beta = -2X^\top(y - X\beta)$$

The penalty on the parameter vector $\|\beta\|^2 = \beta^\top I\beta$ is the quadratic case with $M = I$, so $\nabla_\beta\|\beta\|^2 = 2\beta$.

**Positive definite matrices.** A symmetric matrix $M$ is *positive semidefinite* (PSD) when $v^\top M v \ge 0$ for every $v$, and *positive definite* (PD) when $v^\top M v > 0$ for every $v \neq 0$. A PD matrix is invertible: $Mv = 0$ would give $v^\top M v = 0$, so $v = 0$. For any matrix $X$, $X^\top X$ is PSD because $v^\top X^\top X v = \|Xv\|^2 \ge 0$, and it is PD exactly when $Xv = 0$ has no nonzero solution, that is, when the columns of $X$ are linearly independent.

**Minimising a quadratic.** Let $f(\beta) = c - 2b^\top \beta + \beta^\top M \beta$ with $M$ symmetric PSD. Expanding directly, for any $\beta$ and $h$,

$$f(\beta + h) = f(\beta) + h^\top(2M\beta - 2b) + h^\top M h = f(\beta) + h^\top \nabla f(\beta) + h^\top M h$$

If $\nabla f(\beta) = 0$ then $f(\beta + h) - f(\beta) = h^\top M h \ge 0$ for every $h$, so a stationary point is a global minimiser; if $M$ is PD the difference is strictly positive for $h \neq 0$, so the minimiser is unique. All objective functions involved in ridge regression have this form.

**Random vectors.** For a random vector $\varepsilon \in \mathbb{R}^n$, $E[\varepsilon]$ is the vector of expectations and $\mathrm{Cov}(\varepsilon) = E\big[(\varepsilon - E\varepsilon)(\varepsilon - E\varepsilon)^\top\big]$ the matrix of covariances, with $\mathrm{Var}(\varepsilon_i)$ on the diagonal. Expectation is linear, $E[A\varepsilon + c] = A\,E[\varepsilon] + c$ for fixed $A$ and $c$, and covariance transforms as $\mathrm{Cov}(A\varepsilon + c) = A\,\mathrm{Cov}(\varepsilon)\,A^\top$, which follows by substituting the definition. For a random vector $Z$ with mean $\mu$, $E\|Z - a\|^2 = \mathrm{tr}\,\mathrm{Cov}(Z) + \|\mu - a\|^2$ for any fixed $a$: write $Z - a = (Z - \mu) + (\mu - a)$, expand the square, and use that the cross term has expectation zero and $E\|Z - \mu\|^2 = \sum_i \mathrm{Var}(Z_i)$.

**The normal distribution.** A vector $Z \in \mathbb{R}^p$ has independent $N(0, \tau^2)$ components when its density is $\prod_j (2\pi\tau^2)^{-1/2} e^{-z_j^2/2\tau^2} \propto e^{-\|z\|^2/2\tau^2}$.

---

## Part I — Theory

### 1. Problem statement

#### 1.1 The model

Regression is the following task. Observations $(x_i, y_i)$, $i = 1, \dots, n$, are given, with $x_i \in \mathbb{R}^p$ a vector of $p$ features and $y_i \in \mathbb{R}$ a response. The model is linear in the features:

$$y_i = \beta_0 + x_i^\top \beta + \varepsilon_i, \qquad i = 1, \dots, n$$

where $\beta_0 \in \mathbb{R}$ is the intercept, $\beta \in \mathbb{R}^p$ the coefficient vector, and $\varepsilon_i$ a noise term. Stacking rows gives $y = \beta_0\mathbf{1} + X\beta + \varepsilon$, where $X \in \mathbb{R}^{n \times p}$ has $x_i^\top$ as its $i$-th row.

**Assumptions on the noise.** $E[\varepsilon] = 0$ and $\mathrm{Cov}(\varepsilon) = \sigma^2 I$: the noise has mean zero, constant variance $\sigma^2$, and is uncorrelated across observations. $X$ is treated as fixed: all expectations are conditional on the observed features. No distributional form is assumed for $\varepsilon$ unless stated.

**Goals.** Two are distinguished, because a method can serve one and fail the other: *prediction*, producing $\hat{y}$ for a new $x$ that is close to the new $y$; and *estimation*, producing $\hat\beta$ close to $\beta$. The first is measured by prediction error on data not used for fitting; the second by $E\|\hat\beta - \beta\|^2$, which is computable only in theory since $\beta$ is unknown.

**Types.**

| Symbol | Mathematical type | Representation in code | Remarks |
|---|---|---|---|
| $X$ | $n \times p$ real matrix | `ndarray (n, p)` | one row per observation; columns are features |
| $y$ | vector in $\mathbb{R}^n$ | `ndarray (n,)` | |
| $\beta_0, \beta$ | scalar and vector in $\mathbb{R}^p$ | `float`, `ndarray (p,)` | unknown; to be estimated |
| $\lambda$ | non-negative scalar | `float` | the penalty strength; not dimensionless |
| $\hat\beta_\lambda$ | vector in $\mathbb{R}^p$ | `ndarray (p,)` | the estimate at penalty $\lambda$ |

#### 1.2 The intercept

The intercept is treated differently from the coefficients. Adding a constant $c$ to every response should shift every fitted value by $c$ and leave the slopes alone; a penalty that shrinks $\beta_0$ toward zero pulls the fit toward $y = 0$ instead, so the estimate would depend on the arbitrary origin of $y$. Least squares and ridge therefore both minimise an objective of the form

$$\|y - \beta_0\mathbf{1} - X\beta\|^2 + P(\beta)$$

with a penalty $P$ that does not involve $\beta_0$ ($P = 0$ for least squares). Differentiating with respect to $\beta_0$ and setting to zero gives $\mathbf{1}^\top(y - \beta_0\mathbf{1} - X\beta) = 0$, that is,

$$\hat\beta_0 = \bar{y} - \bar{x}^\top \beta$$

where $\bar{y}$ is the mean response and $\bar{x} \in \mathbb{R}^p$ the vector of column means of $X$. Substituting back, $y - \hat\beta_0\mathbf{1} - X\beta = (y - \bar{y}\mathbf{1}) - (X - \mathbf{1}\bar{x}^\top)\beta$, so the objective becomes

$$\|y_c - X_c\beta\|^2 + P(\beta)$$

with $y_c$ and $X_c$ the centred response and features. The procedure is therefore: centre $y$ and every column of $X$; minimise over $\beta$ alone; recover $\hat\beta_0 = \bar{y} - \bar{x}^\top\hat\beta$. Every formula in Part I is written for centred $X$ and $y$ with the intercept omitted. For prediction at a new point $x$, $\hat{y} = \hat\beta_0 + x^\top\hat\beta = \bar{y} + (x - \bar{x})^\top\hat\beta$, with $\bar{x}$ and $\bar{y}$ the training means.

---

### 2. A running example

The following five observations of two features are used throughout the lesson.

| $i$ | $x_{i1}$ | $x_{i2}$ | $y_i$ |
|---|---|---|---|
| 1 | 1 | 3 | 7 |
| 2 | 2 | 4 | 8 |
| 3 | 3 | 6 | 10 |
| 4 | 4 | 5 | 11 |
| 5 | 5 | 7 | 14 |

The column means are $\bar{x}_1 = 3$, $\bar{x}_2 = 5$, $\bar{y} = 10$. Subtracting each column's mean (§1.2) gives the centred data used in every formula below:

$$X = \begin{pmatrix} -2 & -2 \\ -1 & -1 \\ 0 & 1 \\ 1 & 0 \\ 2 & 2 \end{pmatrix}, \qquad y = \begin{pmatrix} -3 \\ -2 \\ 0 \\ 1 \\ 4 \end{pmatrix}, \qquad X^\top X = \begin{pmatrix} 10 & 9 \\ 9 & 10 \end{pmatrix}, \qquad X^\top y = \begin{pmatrix} 17 \\ 16 \end{pmatrix}$$

The two features are strongly correlated: their correlation is $9/\sqrt{10 \cdot 10} = 0.9$.

---

### 3. Ordinary least squares

#### 3.1 The estimator

Ordinary least squares (OLS) chooses $\beta$ to minimise the residual sum of squares

$$\mathrm{RSS}(\beta) = \|y - X\beta\|^2$$

By the gradient rule of the Preliminaries, $\nabla \mathrm{RSS} = -2X^\top(y - X\beta)$, and setting it to zero gives the **normal equations**

$$X^\top X \hat\beta = X^\top y$$

RSS is a quadratic with matrix $X^\top X$, which is PSD, so any solution of the normal equations is a global minimiser. When the columns of $X$ are linearly independent, $X^\top X$ is PD and the minimiser is unique:

$$\hat\beta_{\mathrm{OLS}} = (X^\top X)^{-1} X^\top y$$

When the columns are linearly dependent — including whenever $p > n$, since more than $n$ vectors in $\mathbb{R}^n$ are linearly dependent — the normal equations have infinitely many solutions.

For the running example, $(X^\top X)^{-1} = \frac{1}{19}\begin{pmatrix} 10 & -9 \\ -9 & 10 \end{pmatrix}$ and

$$\hat\beta_{\mathrm{OLS}} = \frac{1}{19}\begin{pmatrix} 170 - 144 \\ -153 + 160 \end{pmatrix} = \begin{pmatrix} 26/19 \\ 7/19 \end{pmatrix} = \begin{pmatrix} 1.368 \\ 0.368 \end{pmatrix}$$

with $\mathrm{RSS} = 0.842$.

#### 3.2 Mean and variance

Substituting $y = X\beta + \varepsilon$ (the intercept is absent because the data are centred, §1.2):

$$\hat\beta_{\mathrm{OLS}} = (X^\top X)^{-1}X^\top(X\beta + \varepsilon) = \beta + (X^\top X)^{-1}X^\top \varepsilon$$

Taking expectations, $E[\hat\beta_{\mathrm{OLS}}] = \beta$: the estimator is unbiased. Applying the covariance rule with $A = (X^\top X)^{-1}X^\top$ and $\mathrm{Cov}(\varepsilon) = \sigma^2 I$,

$$\mathrm{Cov}(\hat\beta_{\mathrm{OLS}}) = \sigma^2 (X^\top X)^{-1} X^\top X (X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}$$

For the example, $\mathrm{Var}(\hat\beta_{\mathrm{OLS},1}) = \mathrm{Var}(\hat\beta_{\mathrm{OLS},2}) = 10\sigma^2/19 = 0.526\,\sigma^2$, and $\mathrm{Cov}(\hat\beta_{\mathrm{OLS},1}, \hat\beta_{\mathrm{OLS},2}) = -9\sigma^2/19 = -0.474\,\sigma^2$. The total is $E\|\hat\beta_{\mathrm{OLS}} - \beta\|^2 = \mathrm{tr}\,\mathrm{Cov} = 20\sigma^2/19 = 1.053\,\sigma^2$.

#### 3.3 The instability

The large negative covariance says the two coefficient errors move in opposite directions: the *sum* $\hat\beta_{\mathrm{OLS},1} + \hat\beta_{\mathrm{OLS},2}$ is well determined while the *difference* is not. Directly, $\mathrm{Var}(\hat\beta_{\mathrm{OLS},1} + \hat\beta_{\mathrm{OLS},2}) = (10 + 10 - 18)\sigma^2/19 = 2\sigma^2/19$, while $\mathrm{Var}(\hat\beta_{\mathrm{OLS},1} - \hat\beta_{\mathrm{OLS},2}) = (10 + 10 + 18)\sigma^2/19 = 2\sigma^2$: the difference is nineteen times more variable than the sum.

The same appears without any probability. Change the third response from $10$ to $11$, so that centred $y$ becomes $(-3.2, -2.2, 0.8, 0.8, 3.8)$ — an edit of one unit in one of five observations. Then $X^\top y = (17, 17)$ and

$$\hat\beta_{\mathrm{OLS}} = \frac{1}{19}\begin{pmatrix} 170 - 153 \\ -153 + 170 \end{pmatrix} = \begin{pmatrix} 0.895 \\ 0.895 \end{pmatrix}$$

The first coefficient fell by $0.47$ and the second rose by $0.53$; the sum moved from $1.737$ to $1.789$. The mechanism is that the data contain almost no information about the direction $(1, -1)$ — the two columns nearly coincide, so the fit cannot tell how much of the common effect to assign to each — and least squares, having no reason to prefer any assignment, follows the noise.

---

### 4. The ridge estimator

#### 4.1 The objective and its solution

Ridge regression modifies the objective function to be minimised by adding to the residual sum of squares a penalty proportional to the squared norm of the coefficient vector:

$$L_\lambda(\beta) = \|y - X\beta\|^2 + \lambda \|\beta\|^2, \qquad \lambda \ge 0$$

This change expresses a preference for solutions with a smaller Euclidean norm $\|\beta\|^2$. Its gradient, from the Preliminaries, is $-2X^\top(y - X\beta) + 2\lambda\beta$. Setting it to zero:

$$(X^\top X + \lambda I)\,\hat\beta_\lambda = X^\top y$$

For $\lambda > 0$ the matrix $X^\top X + \lambda I$ is PD, since $v^\top(X^\top X + \lambda I)v = \|Xv\|^2 + \lambda\|v\|^2 > 0$ for $v \neq 0$. It is therefore invertible, and the stationary point is the unique global minimiser:

$$\boxed{\;\hat\beta_\lambda = (X^\top X + \lambda I)^{-1} X^\top y\;}$$

This holds for every $X$ — correlated columns, identical columns, or $p > n$. At $\lambda = 0$ the equations reduce to the normal equations and $\hat\beta_{\lambda=0} = \hat\beta_{\mathrm{OLS}}$ whenever the latter exists.

#### 4.2 The example

With $\lambda = 1$, $X^\top X + I = \begin{pmatrix} 11 & 9 \\ 9 & 11 \end{pmatrix}$, whose determinant is $121 - 81 = 40$, so

$$\hat\beta_{\lambda=1} = \frac{1}{40}\begin{pmatrix} 11 & -9 \\ -9 & 11 \end{pmatrix}\begin{pmatrix} 17 \\ 16 \end{pmatrix} = \frac{1}{40}\begin{pmatrix} 43 \\ 23 \end{pmatrix} = \begin{pmatrix} 1.075 \\ 0.575 \end{pmatrix}$$

The fitted values are $X\hat\beta_1 = (-3.3, -1.65, 0.575, 1.075, 3.3)$, the residuals $(0.3, -0.35, -0.575, -0.075, 0.7)$, and $\mathrm{RSS} = 1.039$, larger than the OLS value $0.842$ because OLS minimises RSS and ridge does not.

Under the same one-unit edit to $y_3$ as in §3.3, $X^\top y = (17, 17)$ and $\hat\beta_{\lambda=1} = \frac{1}{40}(187 - 153,\; -153 + 187) = (0.85, 0.85)$. The first coefficient moved by $0.225$ instead of $0.47$: the same perturbation, half the response.

The full path as $\lambda$ grows:

| $\lambda$ | $\hat\beta_{\lambda,1}$ | $\hat\beta_{\lambda,2}$ | $\lVert\hat\beta_\lambda\rVert^2$ | RSS |
|---|---|---|---|---|
| 0 | 1.368 | 0.368 | 2.008 | 0.842 |
| 0.5 | 1.179 | 0.513 | 1.654 | 0.917 |
| 1 | 1.075 | 0.575 | 1.486 | 1.039 |
| 2 | 0.952 | 0.619 | 1.290 | 1.324 |
| 5 | 0.771 | 0.604 | 0.959 | 2.433 |
| 10 | 0.614 | 0.524 | 0.652 | 4.663 |
| 100 | 0.144 | 0.134 | 0.039 | 21.570 |

Two regularities are visible. The norm of the estimate falls, which is expected because a greater $\lambda$ favours smaller norms, and the RSS rises monotonically in $\lambda$, also expected because $\lambda$ reduces the space of the eligible $\beta$. The two coefficients approach each other before they approach zero: the penalty first removes the part of the estimate that the data determine poorly, the difference, and only later the part they determine well, the sum.

---

### 5. The scale of the features

#### 5.1 Why the features are standardised

Replace column $j$ of $X$ by $c\,x_j$ for some $c > 0$, as happens when a feature is re-expressed in different units. The model $y = X\beta + \varepsilon$ is unchanged if $\beta_j$ is replaced by $\beta_j/c$, and OLS respects this: its estimate of the $j$-th coefficient is divided by $c$ and every fitted value is the same. Ridge does not follow the same proportion. The penalty term for coefficient $j$ becomes $\lambda(\beta_j/c)^2 = (\lambda/c^2)\beta_j^2$: scaling the feature up by $c$ is equivalent to reducing the penalty on its coefficient by $c^2$. A feature measured in millimetres ($c=1000$) is penalised a million times less than the same feature in metres.

For the example, multiply the second column by $10$ and fit with $\lambda = 1$. In the original units the estimate becomes $(0.899, 0.790)$ instead of $(1.075, 0.575)$, and the prediction at centred $x = (1, 1)$ changes from $1.650$ to $1.689$. The data are identical; only the labelling of the axis changed.

Since the penalty compares coefficients on a common scale, the features must be placed on one. The standard choice divides each centred column by its standard deviation $s_j = \sqrt{\frac{1}{n}\sum_i (x_{ij} - \bar{x}_j)^2}$, so that every column has unit variance and every coefficient is measured per standard deviation of its feature. Coefficients on the original scale are recovered as $\hat\beta_j / s_j$. The means and standard deviations are statistics of the training data and are part of the fitted model: a new observation is transformed with the training values, never with its own.

The running example has $s_1 = s_2 = \sqrt{2}$, so standardising it would divide both columns by the same number and change only the unit in which $\lambda$ is measured: minimising $\|y - X\gamma/\sqrt{2}\|^2 + \lambda\|\gamma\|^2$ over $\gamma = \sqrt{2}\beta$ is the same as minimising $\|y - X\beta\|^2 + 2\lambda\|\beta\|^2$. The unscaled centred columns are kept for the rest of the lesson so that the integer matrix $X^\top X$ survives.

#### 5.2 Rotational invariance

Standardisation is necessary because ridge is not invariant to rescaling a column. It is invariant to rotating the feature space. Let $Q$ be a $p \times p$ orthogonal matrix and replace $X$ by $XQ$, so that the new features are linear combinations of the old. Then $Q^\top X^\top X Q + \lambda I = Q^\top(X^\top X + \lambda I)Q$, whose inverse is $Q^\top(X^\top X + \lambda I)^{-1}Q$, and

$$\hat\beta'_\lambda = Q^\top(X^\top X + \lambda I)^{-1} Q\,Q^\top X^\top y = Q^\top \hat\beta_\lambda$$

The estimate rotates with the features and the fitted values $XQ\hat\beta'_\lambda = X\hat\beta_\lambda$ do not change. The penalty $\|\beta\|^2$ is a length, and lengths are preserved by rotation and destroyed by stretching one axis. This distinguishes ridge from penalties such as $\sum_j |\beta_j|$ (the lasso), which single out the coordinate axes and are not rotation-invariant. The set where that penalty is constant is a diamond with corners on the axes, points at which a coefficient is exactly zero, and the contours of the RSS can meet it at a corner; the set where $\|\beta\|^2$ is constant is a sphere with no corners, so ridge moves coefficients toward zero and never onto it.

---

### 6. The singular value decomposition view

#### 6.1 Two facts from linear algebra

**Spectral theorem.** A real symmetric $p \times p$ matrix $M$ can be written $M = V D V^\top$ with $V$ orthogonal and $D = \mathrm{diag}(\mu_1, \dots, \mu_p)$ real. The columns $v_j$ of $V$ are eigenvectors, $Mv_j = \mu_j v_j$, and they form an orthonormal basis. If $M$ is PSD then every $\mu_j = v_j^\top M v_j \ge 0$. The theorem is stated without proof.

**Singular value decomposition.** Applying the spectral theorem to $X^\top X$ gives $X^\top X = V D^2 V^\top$ with $D^2 = \mathrm{diag}(d_1^2, \dots, d_p^2)$, $d_1 \ge \dots \ge d_p \ge 0$. Define $u_j = Xv_j / d_j$ for each $d_j > 0$; these are orthonormal, since $u_j^\top u_k = v_j^\top X^\top X v_k / (d_j d_k) = d_k^2\,v_j^\top v_k/(d_j d_k)$, which is $1$ for $j = k$ and $0$ otherwise. When $d_j = 0$, $\|Xv_j\|^2 = d_j^2 = 0$, so $Xv_j = 0$; put $u_j = 0$ for those $j$. In every case $Xv_j = d_j u_j$. Collecting the $u_j$ as the columns of $U \in \mathbb{R}^{n \times p}$ and writing $D = \mathrm{diag}(d_1, \dots, d_p)$, the $p$ relations read $XV = UD$, and multiplying on the right by $V^\top$ gives

$$X = U D V^\top, \qquad V^\top V = V V^\top = I$$

This is the *singular value decomposition* (SVD) of $X$; the $d_j$ are its singular values. The nonzero columns of $U$ are orthonormal, so $U^\top U = I$ when every $d_j > 0$; a zero column enters every formula below multiplied by $d_j$ or by a factor that vanishes with it. Geometrically, the $v_j$ are the *principal directions* of the feature cloud and $d_j^2 = \|Xv_j\|^2$ is $n$ times the variance of the data along $v_j$.

For the example, $X^\top X = \begin{pmatrix} 10 & 9 \\ 9 & 10 \end{pmatrix}$ has eigenvalues $19$ and $1$, with eigenvectors

$$v_1 = \tfrac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}, \qquad v_2 = \tfrac{1}{\sqrt{2}}\begin{pmatrix} -1 \\ 1 \end{pmatrix}$$

so $d_1^2 = 19$, $d_2^2 = 1$. The sum direction carries nineteen times the variance of the difference direction, which is the geometry behind §3.3.

#### 6.2 Ridge in SVD coordinates

Substituting $X = UDV^\top$ into the ridge solution, with $X^\top X + \lambda I = V(D^2 + \lambda I)V^\top$ and $X^\top y = V D U^\top y$:

$$\hat\beta_\lambda = V (D^2 + \lambda I)^{-1} D\, U^\top y = \sum_{j=1}^{p} \frac{d_j}{d_j^2 + \lambda}\,(u_j^\top y)\, v_j$$

The OLS estimate is the same expression at $\lambda = 0$: $\hat\beta_{\mathrm{OLS}} = \sum_j (u_j^\top y / d_j)\, v_j$. Writing $\alpha = V^\top\beta$ for the coordinates of any coefficient vector in the basis of principal directions, the two estimates are related coordinate by coordinate:

$$\boxed{\;\hat\alpha_{\lambda, j} = s_j(\lambda)\,\hat\alpha_{\mathrm{OLS}, j}, \qquad s_j(\lambda) = \frac{d_j^2}{d_j^2 + \lambda} \in (0, 1]\;}$$

The relation holds for every $j$ with $d_j > 0$; a coordinate with $d_j = 0$ has no OLS value and is $0$ under ridge for every $\lambda > 0$. Ridge is OLS followed by a *shrinkage* of each principal coordinate by the factor $s_j(\lambda)$. The factor depends on the direction: where the data have large variance ($d_j^2 \gg \lambda$) it is near $1$ and the OLS coordinate is kept; where they have little ($d_j^2 \ll \lambda$) it is near $0$ and the coordinate is discarded. The penalty acts on exactly the directions in which §3.3 found OLS to be unstable, and it acts on them in proportion to how unstable they are.

For the example at $\lambda = 1$, $s_1 = 19/20 = 0.95$ and $s_2 = 1/2 = 0.5$. The OLS coordinates are $\hat\alpha_{\mathrm{OLS}} = V^\top\hat\beta_{\mathrm{OLS}} = \big((26 + 7)/(19\sqrt{2}),\; (-26 + 7)/(19\sqrt{2})\big) = (1.228, -0.707)$; shrinking gives $(1.167, -0.354)$; and mapping back,

$$\hat\beta_{\lambda=1} = 1.167\,v_1 - 0.354\,v_2 = \begin{pmatrix} 0.825 + 0.250 \\ 0.825 - 0.250 \end{pmatrix} = \begin{pmatrix} 1.075 \\ 0.575 \end{pmatrix}$$

which is the value found in §4.2. The sum direction lost $5\%$ of its OLS value; the difference direction lost half.

#### 6.3 Consequences

**The norm decreases and the RSS increases in $\lambda$.** Since $V$ is orthogonal, $\|\hat\beta_\lambda\|^2 = \|\hat\alpha_\lambda\|^2 = \sum_j s_j(\lambda)^2 \hat\alpha_{\mathrm{OLS},j}^2$, and every $s_j$ is strictly decreasing in $\lambda$. For the RSS, decompose $y$ along the $u_j$ and the part $y_\perp$ orthogonal to all of them: the fitted vector is $X\hat\beta_\lambda = \sum_j s_j (u_j^\top y) u_j$, so the residual is $\sum_j (1 - s_j)(u_j^\top y) u_j + y_\perp$ and

$$\mathrm{RSS}(\lambda) = \sum_j (1 - s_j(\lambda))^2 (u_j^\top y)^2 + \|y_\perp\|^2$$

which increases with $\lambda$. Both regularities of the table in §4.2 are therefore general.

**The two limits.** As $\lambda \to 0$, $s_j \to 1$ for every $d_j > 0$ and $\hat\beta_\lambda \to \hat\beta_{\mathrm{OLS}}$. If some $d_j = 0$ the coefficient $d_j/(d_j^2 + \lambda)$ is $0$ for every $\lambda > 0$, so the limit is the least-squares solution with no component along the null directions: the minimum-norm least-squares solution. As $\lambda \to \infty$, every $s_j \to 0$ and $\hat\beta_\lambda \to 0$; the fit collapses to the intercept $\bar{y}$.

**The hat matrix and effective degrees of freedom.** The fitted values are a linear function of $y$, $\hat{y} = H_\lambda y$, with

$$H_\lambda = X(X^\top X + \lambda I)^{-1}X^\top = U\,\mathrm{diag}(s_1, \dots, s_p)\,U^\top$$

For OLS, $H_0 = UU^\top$ is the orthogonal projection onto the column space of $X$ and its trace, the number of fitted parameters, is $p$. For ridge the trace is

$$\mathrm{df}(\lambda) = \mathrm{tr}(H_\lambda) = \sum_{j=1}^{p} \frac{d_j^2}{d_j^2 + \lambda}$$

a number between $0$ and $p$ called the *effective degrees of freedom*: the count of parameters the fit is behaving as though it had. When the intercept is fitted by centring, it adds one unpenalised degree of freedom, and $H_\lambda$ gains the term $\mathbf{1}\mathbf{1}^\top/n$. For the example, $\mathrm{df}(1) = 0.95 + 0.5 = 1.45$, plus one for the intercept.

**The scale of $\lambda$.** The shrinkage factor compares $\lambda$ with $d_j^2$, and $\sum_j d_j^2 = \mathrm{tr}(X^\top X) = \sum_j \|x_j\|^2$, the total sum of squares of the columns. For standardised columns this is $np$. So $\lambda$ has the units of a column sum of squares, grows in proportion to $n$ for a fixed problem, and takes effect on direction $j$ around $\lambda \approx d_j^2$. A search over $\lambda$ should cover, on a logarithmic grid, the range from well below $d_p^2$ to well above $d_1^2$.

---

### 7. Bias, variance, and when ridge wins

#### 7.1 Mean and covariance of the ridge estimate

Substituting $y = X\beta + \varepsilon$ into the ridge solution,

$$\hat\beta_\lambda = (X^\top X + \lambda I)^{-1}X^\top X\,\beta + (X^\top X + \lambda I)^{-1}X^\top\varepsilon$$

Taking expectations, $E[\hat\beta_\lambda] = (X^\top X + \lambda I)^{-1}X^\top X\,\beta = V\,\mathrm{diag}(s_j)\,V^\top\beta$: in principal coordinates, $E[\hat\alpha_{\lambda,j}] = s_j\alpha_j$. The estimate is biased, by

$$E[\hat\alpha_{\lambda,j}] - \alpha_j = -(1 - s_j)\alpha_j = -\frac{\lambda}{d_j^2 + \lambda}\,\alpha_j$$

The bias is toward zero and largest in the low-variance directions. Applying the covariance rule to the noise term,

$$\mathrm{Cov}(\hat\beta_\lambda) = \sigma^2 (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1} = \sigma^2\, V\,\mathrm{diag}\!\left(\frac{d_j^2}{(d_j^2 + \lambda)^2}\right) V^\top$$

so $\mathrm{Var}(\hat\alpha_{\lambda,j}) = \sigma^2 d_j^2/(d_j^2 + \lambda)^2 = s_j^2\cdot\sigma^2/d_j^2$: the OLS variance in that direction, multiplied by the shrinkage factor squared. For the example at $\lambda = 1$: OLS variances along $v_1, v_2$ are $\sigma^2/19$ and $\sigma^2$, and ridge reduces them to $19\sigma^2/400 = 0.048\sigma^2$ and $\sigma^2/4$. The trace falls from $1.053\sigma^2$ to $0.298\sigma^2$.

#### 7.2 The mean squared error

Since $\|\hat\beta_\lambda - \beta\| = \|\hat\alpha_\lambda - \alpha\|$, the identity $E\|Z - a\|^2 = \mathrm{tr}\,\mathrm{Cov}(Z) + \|EZ - a\|^2$ from the Preliminaries gives

$$\boxed{\;\mathrm{MSE}(\lambda) = E\|\hat\beta_\lambda - \beta\|^2 = \sum_{j=1}^{p}\left[\frac{\sigma^2 d_j^2}{(d_j^2 + \lambda)^2} + \frac{\lambda^2\alpha_j^2}{(d_j^2 + \lambda)^2}\right]\;}$$

The first term in each bracket is variance, decreasing in $\lambda$; the second is squared bias, increasing from $0$ at $\lambda = 0$ toward $\alpha_j^2$ as $\lambda \to \infty$. At $\lambda = 0$ the expression is the OLS value $\sigma^2\sum_j 1/d_j^2$, which is dominated by the smallest singular value; that is the precise form of the instability of §3.3.

#### 7.3 A positive $\lambda$ always helps

Differentiating each bracket with respect to $\lambda$ (the quotient rule, with $(d_j^2 + \lambda)^{-2}$ having derivative $-2(d_j^2 + \lambda)^{-3}$):

$$\frac{d\,\mathrm{MSE}}{d\lambda} = \sum_{j=1}^{p}\frac{-2\sigma^2 d_j^2 + 2\lambda\alpha_j^2(d_j^2 + \lambda) - 2\lambda^2\alpha_j^2}{(d_j^2 + \lambda)^3} = 2\sum_{j=1}^{p}\frac{d_j^2\,(\lambda\alpha_j^2 - \sigma^2)}{(d_j^2 + \lambda)^3}$$

At $\lambda = 0$ this equals $-2\sigma^2\sum_j 1/d_j^4 < 0$. The MSE is therefore strictly decreasing at the origin, so there is an interval $(0, \lambda^*)$ on which every $\lambda$ gives a smaller mean squared error than OLS. This holds for every $\beta$, every $\sigma^2 > 0$ and every $X$. The derivative also shows that every term is negative while $\lambda < \sigma^2/\max_j\alpha_j^2$, so the MSE decreases at least up to that point. The result is due to Hoerl and Kennard (1970).

What the theorem does not give is the value of $\lambda^*$, which depends on the unknown $\alpha$ and $\sigma^2$. That is why $\lambda$ is chosen from the data, by cross-validation, rather than from the formula.

For the example, suppose the truth were $\beta = (1, 0.5)$ and $\sigma^2 = 1$, so that $\alpha = V^\top\beta = (1.061, -0.354)$. Then

| $\lambda$ | variance | bias$^2$ | MSE |
|---|---|---|---|
| 0 | 1.053 | 0 | 1.053 |
| 1 | 0.298 | 0.034 | 0.332 |
| 2 | 0.154 | 0.066 | 0.220 |
| 5 | 0.061 | 0.136 | 0.196 |
| 10 | 0.031 | 0.237 | 0.268 |
| 20 | 0.015 | 0.409 | 0.424 |

The minimum lies between $2$ and $5$ (at $\lambda \approx 3.7$ on a fine grid), at under a fifth of the OLS error. Almost all the gain is variance removed from the direction $v_2$, at the price of a bias that grows quadratically and eventually dominates.

#### 7.4 Estimation error and prediction error

Prediction error at the training features is $E\|X\hat\beta_\lambda - X\beta\|^2$, and since $X(\hat\beta_\lambda - \beta) = UD(\hat\alpha_\lambda - \alpha)$ with $U$ orthonormal, it is the MSE with each bracket weighted by $d_j^2$:

$$E\|X\hat\beta_\lambda - X\beta\|^2 = \sum_{j=1}^{p} d_j^2\left[\frac{\sigma^2 d_j^2}{(d_j^2 + \lambda)^2} + \frac{\lambda^2\alpha_j^2}{(d_j^2 + \lambda)^2}\right]$$

The weight $d_j^2$ suppresses the low-variance directions: an error along $v_2$ in the example is multiplied by $1$ where an error along $v_1$ is multiplied by $19$. This is why OLS can predict acceptably at points resembling the training data while its coefficients are wildly wrong, and why the same coefficients fail at points that lie off the principal directions of the training cloud. Cross-validation measures prediction error, so it selects $\lambda$ for prediction on the training distribution; a $\lambda$ that is good for estimating $\beta$, or for predicting far from the data, is generally larger.

---

### 8. Three other routes to the same estimator

Each of the following arrives at the ridge solution from a different premise and adds an interpretation of $\lambda$.

#### 8.1 Constrained least squares

Consider minimising $\|y - X\beta\|^2$ subject to $\|\beta\|^2 \le t$. Let $\hat\beta_\lambda$ be the ridge estimate at some $\lambda > 0$ and let $\beta$ be any vector with $\|\beta\|^2 \le \|\hat\beta_\lambda\|^2$. Since $\hat\beta_\lambda$ minimises $L_\lambda$,

$$\|y - X\beta\|^2 + \lambda\|\beta\|^2 \ge \|y - X\hat\beta_\lambda\|^2 + \lambda\|\hat\beta_\lambda\|^2 \quad\Longrightarrow\quad \|y - X\beta\|^2 \ge \|y - X\hat\beta_\lambda\|^2 + \lambda\big(\|\hat\beta_\lambda\|^2 - \|\beta\|^2\big) \ge \|y - X\hat\beta_\lambda\|^2$$

So $\hat\beta_\lambda$ solves the constrained problem with $t = \|\hat\beta_\lambda\|^2$. Because $\|\hat\beta_\lambda\|^2$ decreases continuously from $\|\hat\beta_{\mathrm{OLS}}\|^2$ to $0$ as $\lambda$ runs from $0$ to $\infty$ (§6.3), every budget $t$ below $\|\hat\beta_{\mathrm{OLS}}\|^2$ corresponds to exactly one $\lambda$; for $t$ at or above it the constraint is inactive and the solution is OLS itself. When OLS is undefined, the minimum-norm solution of §6.3 takes its place, and the decrease is strict as long as that solution is nonzero. Ridge is least squares inside a ball around the origin; $\lambda$ is the price per unit of squared norm.

#### 8.2 The Bayesian view

Suppose $\varepsilon$ has independent $N(0, \sigma^2)$ components and, before seeing the data, $\beta$ is regarded as random with independent $N(0, \tau^2)$ components. The joint density of $y$ and $\beta$ is proportional to

$$\exp\!\left(-\frac{\|y - X\beta\|^2}{2\sigma^2}\right)\exp\!\left(-\frac{\|\beta\|^2}{2\tau^2}\right)$$

and, for the observed $y$, the value of $\beta$ that maximises it (the *maximum a posteriori* estimate) minimises

$$\frac{\|y - X\beta\|^2}{2\sigma^2} + \frac{\|\beta\|^2}{2\tau^2} \;\propto\; \|y - X\beta\|^2 + \frac{\sigma^2}{\tau^2}\|\beta\|^2$$

which is ridge with $\lambda = \sigma^2/\tau^2$: the ratio of noise variance to prior coefficient variance. A large $\lambda$ encodes the belief that coefficients are small relative to the noise; $\lambda \to 0$ is the belief that any coefficient size is equally plausible. Since the exponent is a quadratic in $\beta$, the posterior is itself normal and its mean coincides with its maximum, so the ridge estimate is also the posterior mean. The prior is symmetric across coordinates, which is the probabilistic form of the requirement in §5.1 that the features share a scale.

#### 8.3 Data augmentation

Append $p$ artificial rows to the data:

$$\tilde{X} = \begin{pmatrix} X \\ \sqrt{\lambda}\,I \end{pmatrix} \in \mathbb{R}^{(n + p) \times p}, \qquad \tilde{y} = \begin{pmatrix} y \\ 0 \end{pmatrix}$$

Then $\|\tilde{y} - \tilde{X}\beta\|^2 = \|y - X\beta\|^2 + \lambda\|\beta\|^2$, so ridge is OLS on the augmented data, and $\tilde{X}^\top\tilde{X} = X^\top X + \lambda I$ is the ridge matrix. Each artificial row is an observation saying "$\beta_j = 0$", with weight $\lambda$. The rows are linearly independent whatever $X$ is, which is another way of seeing why the solution always exists. Any least-squares routine computes a ridge fit when handed $\tilde{X}$ and $\tilde{y}$.

---

### 9. The dual form

The ridge solution can be written using an $n \times n$ matrix instead of a $p \times p$ one. Start from the identity $X^\top(XX^\top + \lambda I) = X^\top X X^\top + \lambda X^\top = (X^\top X + \lambda I)X^\top$. Both bracketed matrices are PD for $\lambda > 0$; multiplying on the left by $(X^\top X + \lambda I)^{-1}$ and on the right by $(XX^\top + \lambda I)^{-1}$ gives

$$(X^\top X + \lambda I)^{-1}X^\top = X^\top(XX^\top + \lambda I)^{-1}$$

and therefore

$$\hat\beta_\lambda = X^\top a, \qquad a = (K + \lambda I)^{-1}y, \qquad K = XX^\top$$

$K$ is the *Gram matrix*, $K_{ik} = x_i^\top x_k$, the matrix of inner products between observations. Two consequences follow.

**The estimate is a combination of the observations.** $\hat\beta_\lambda = \sum_i a_i x_i$ lies in the span of the rows of $X$, whatever $p$ is. The prediction at a new point is $x^\top\hat\beta_\lambda = \sum_i a_i\,x^\top x_i$, which involves the features only through inner products. Replacing the inner product by another function of two points, a *kernel*, gives kernel ridge regression.

**The cost moves from $p$ to $n$.** The primal form solves a $p \times p$ system; forming $X^\top X$ costs of the order of $np^2$ operations and the solve of the order of $p^3$. The dual solves an $n \times n$ system at cost of the order of $n^2 p + n^3$. When $p \gg n$, the dual is the one to use.

For the example the dual returns $(1.075, 0.575)$ at $\lambda = 1$.

---

### 10. Gradient descent and weight decay

When $X$ is too large to factorise, ridge is fitted iteratively. The gradient of $\tfrac{1}{2}L_\lambda$ is $X^\top(X\beta - y) + \lambda\beta$, and gradient descent with step size $\eta$ updates

$$\beta \leftarrow \beta - \eta\big(X^\top(X\beta - y) + \lambda\beta\big) = (1 - \eta\lambda)\,\beta - \eta\,X^\top(X\beta - y)$$

The penalty appears as the factor $(1 - \eta\lambda)$ multiplying the current coefficients before each data step: at every iteration the weights *decay* toward zero by a fixed fraction. This is the *weight decay* of neural-network training. Under plain gradient descent, and for any differentiable loss, decaying the weights by the fraction $\eta w$ per step is the same as adding $\tfrac{w}{2}\|\beta\|^2$ to the loss; for the objective $\tfrac{1}{2}L_\lambda$ used here, $w = \lambda$.

**Convergence.** In principal coordinates the update decouples: $\alpha_j \leftarrow (1 - \eta(d_j^2 + \lambda))\alpha_j + \eta d_j u_j^\top y$, a scalar recursion that converges when $|1 - \eta(d_j^2 + \lambda)| < 1$ for every $j$, that is, when $\eta < 2/(d_1^2 + \lambda)$. The slowest direction is the one with the smallest $d_j^2 + \lambda$, so the number of iterations scales with the *condition number* $(d_1^2 + \lambda)/(d_p^2 + \lambda)$. The penalty improves it: for the example, $19/1$ at $\lambda = 0$ becomes $20/2$ at $\lambda = 1$.

**Early stopping.** Run gradient descent on the *unpenalised* loss from $\beta = 0$. The recursion is $\alpha_j^{(t+1)} = (1 - \eta d_j^2)\alpha_j^{(t)} + \eta d_j u_j^\top y$, and unrolling it from $\alpha_j^{(0)} = 0$ gives

$$\alpha_j^{(t)} = \big(1 - (1 - \eta d_j^2)^t\big)\,\frac{u_j^\top y}{d_j} = \big(1 - (1 - \eta d_j^2)^t\big)\,\hat\alpha_{\mathrm{OLS}, j}$$

For $\eta < 1/d_1^2$, after $t$ steps each OLS coordinate is multiplied by a factor in $(0, 1)$ that is close to $1$ for large $d_j$ and close to $0$ for small $d_j$: the same qualitative shrinkage as ridge, with the number of iterations $t$ in the role of $1/\lambda$. Stopping gradient descent early is a regulariser for the same reason ridge is.

---

## Part II — Practice

### 11. Choosing $\lambda$

#### 11.1 What is being minimised

The training RSS increases monotonically in $\lambda$ (§6.3), so it cannot choose $\lambda$: it always votes for $\lambda = 0$. The quantity to minimise is the expected squared error on observations *not used for fitting*. Cross-validation estimates it by holding data out.

#### 11.2 K-fold cross-validation

Split the $n$ observations at random into $K$ folds of near-equal size, with $K = 5$ or $10$ the usual choices. For each candidate $\lambda$ on a logarithmic grid spanning the range identified in §6.3, and for each fold $k$: standardise using the means and standard deviations of the other $K - 1$ folds, fit on those folds, predict the held-out fold, and record the squared errors. The cross-validation score $\mathrm{CV}(\lambda)$ is the mean squared error over all $n$ held-out predictions. Choose the $\lambda$ that minimises it, then refit on all $n$ observations at that $\lambda$.

The standardisation must be recomputed inside each fold. Standardising once on all $n$ observations lets the held-out fold's values influence the scale applied to the training folds, which is a leak; it is a small one for ridge but the habit protects against larger leaks elsewhere in a pipeline.

The curve $\mathrm{CV}(\lambda)$ is typically flat near its minimum, and the minimum's location is itself noisy. A common convention, the *one-standard-error rule*, takes the largest $\lambda$ whose CV score is within one standard error (the standard deviation of the $K$ per-fold scores divided by $\sqrt{K}$) of the minimum, preferring the simpler model among those the data cannot distinguish.

#### 11.3 Leave-one-out cross-validation (LOOCV) in closed form

If we consider $K = n$, then every observation is its own fold. The model is fitted without observation $i$, asked to predict $y_i$, and the squared error is averaged over all $n$ observations. This *leave-one-out* score uses every observation as a test case.

Taken literally, leave-one-out requires $n$ fits for each candidate $\lambda$: with a grid of $50$ values and a few hundred observations, tens of thousands of fits. Fortunately, for ridge it requires one fit per candidate $\lambda$, because the prediction the model *would* have made for observation $i$ without seeing it can be recovered from the fit that did see it.

Fix a candidate $\lambda$ and an observation $i$. Let $\hat\beta_{(-i)}$ be the ridge fit without observation $i$, and $\hat{y}_{(-i), i} = x_i^\top\hat\beta_{(-i)}$ its prediction of the held-out response. Now form a modified response vector $\tilde{y}$ equal to $y$ except that $\tilde{y}_i = \hat{y}_{(-i), i}$, and consider the ridge objective on all $n$ observations with $\tilde{y}$ in place of $y$:

$$\sum_{k \neq i}(y_k - x_k^\top\beta)^2 + (\hat{y}_{(-i),i} - x_i^\top\beta)^2 + \lambda\|\beta\|^2$$

The first and third terms together are minimised by $\hat\beta_{(-i)}$, by definition. The middle term is non-negative and equals $0$ at $\beta = \hat\beta_{(-i)}$. So $\hat\beta_{(-i)}$ minimises the whole objective: the full-data ridge fit to $\tilde{y}$ *is* the leave-one-out fit. Since fitted values are linear in the response, $\hat{y} = H_\lambda y$,

$$\hat{y}_{(-i),i} = (H_\lambda\tilde{y})_i = \sum_{k} h_{ik}\tilde{y}_k = \hat{y}_i - h_{ii}y_i + h_{ii}\hat{y}_{(-i),i}$$

where $\hat{y}_i = (H_\lambda y)_i$ is the ordinary fitted value and $h_{ii}$ the $i$-th diagonal entry of $H_\lambda$. Solving for $\hat{y}_{(-i),i}$ and subtracting from $y_i$:

$$\boxed{\;y_i - \hat{y}_{(-i),i} = \frac{y_i - \hat{y}_i}{1 - h_{ii}}, \qquad \mathrm{LOOCV}(\lambda) = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2\;}$$

The leave-one-out residual is the ordinary residual inflated by $1/(1 - h_{ii})$. The *leverage* $h_{ii}$ measures how much observation $i$ pulls its own fitted value; for $\lambda > 0$ it is strictly below $1$. With the intercept included, $h_{ii} = 1/n + \sum_j s_j u_{ij}^2$ (§6.3). The columns of $U$ are orthogonal to $\mathbf{1}$ because $X$ is centred, so the nonzero columns of $[\mathbf{1}/\sqrt{n},\, U]$ are orthonormal; a set of orthonormal columns extends to an orthogonal matrix, whose rows have unit norm, so $1/n + \sum_j u_{ij}^2 \le 1$. Since every $s_j < 1$ when $\lambda > 0$, it follows that $h_{ii} < 1$. With the intercept term included in $H_\lambda$, the formula equals the leave-one-out score of a procedure that recentres the data after removing each observation but keeps the standard deviations used to scale each column, computed from all $n$ training rows; recomputing them from the remaining $n - 1$ rows as well changes the score by an amount of order $1/n$, negligible for moderate $n$.

**Selecting $\lambda$ among all the candidates.** With the SVD $X = UDV^\top$ of §6, the fitted values and the leverages for any $\lambda$ are $\hat{y} = \sum_j s_j(\lambda)\,(u_j^\top y)\,u_j$ and $h_{ii} = 1/n + \sum_j s_j(\lambda)\,u_{ij}^2$ with $s_j(\lambda) = d_j^2/(d_j^2 + \lambda)$, so $\mathrm{LOOCV}(\lambda)$ is evaluated on the whole grid from one factorisation. The selected value is

$$\boxed{\;\hat\lambda = \arg\min_{\lambda \in \Lambda}\;\frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i(\lambda)}{1 - h_{ii}(\lambda)}\right)^2\;}$$

where $\Lambda$ is the logarithmic grid of §6.3. The model is then refitted on all $n$ observations at $\hat\lambda$.

#### 11.4 Generalised cross-validation

Replacing each $h_{ii}$ by the average leverage $\mathrm{tr}(H_\lambda)/n = \mathrm{df}(\lambda)/n$ gives

$$\mathrm{GCV}(\lambda) = \frac{1}{n}\,\frac{\mathrm{RSS}(\lambda)}{\big(1 - \mathrm{df}(\lambda)/n\big)^2}$$

which needs only the residuals and the shrinkage factors, both available from one SVD for every $\lambda$ at negligible cost. GCV is invariant to rotations of the observations (it depends on $H_\lambda$ only through its trace) where LOOCV is not, and the two agree when the leverages are roughly equal. Golub, Heath and Wahba (1979) introduced GCV for ridge; Craven and Wahba (1978) is the companion treatment for smoothing.

#### 11.5 The example

For the five observations, with the intercept included in $H_\lambda$:

| $\lambda$ | df | RSS | LOOCV | GCV |
|---|---|---|---|---|
| 0 | 3.000 | 0.842 | 1.452 | 1.053 |
| 0.25 | 2.787 | 0.867 | 1.118 | 0.885 |
| 0.5 | 2.641 | 0.917 | 1.036 | 0.823 |
| 1 | 2.450 | 1.039 | 1.019 | 0.799 |
| 1.5 | 2.327 | 1.176 | 1.066 | 0.823 |
| 2 | 2.238 | 1.324 | 1.141 | 0.868 |
| 5 | 1.958 | 2.433 | 1.817 | 1.315 |
| 10 | 1.746 | 4.663 | 2.980 | 2.202 |

On a fine grid LOOCV is minimised at $\lambda \approx 0.81$ and GCV at $\lambda \approx 0.92$.

---

### 12. Computing the fit

**Do not invert.** Forming $(X^\top X + \lambda I)^{-1}$ explicitly and multiplying by $X^\top y$ costs about three times a Cholesky solve and is less accurate; solve the linear system instead.

**Cholesky.** For $n \ge p$ and a single $\lambda$: form $X^\top X + \lambda I$ (order $np^2$ operations), factorise it as $LL^\top$ with $L$ lower triangular (order $p^3/3$), and solve two triangular systems. This is the fastest route for one $\lambda$ and moderate $p$.

**SVD.** For a path of $\lambda$ values, compute $X = UDV^\top$ once (order $np^2$). Every fit is then $\hat\beta_\lambda = V\big(\tfrac{d_j}{d_j^2 + \lambda}\,u_j^\top y\big)_j$, at cost of order $p^2$ per $\lambda$ after the vector $U^\top y$ is formed, and $\mathrm{df}(\lambda)$, GCV and the leverages come from the same factors. A full cross-validation curve costs little more than one fit.

**QR on the augmented system.** Solving the least-squares problem of §8.3 by a QR factorisation of $\tilde{X}$ never forms $X^\top X$, whose condition number is the square of that of $X$; it is the most accurate route when $X$ is badly conditioned and $\lambda$ is small.

**Dual.** When $p \gg n$, solve the $n \times n$ system of §9.

**Conditioning.** The matrix solved has eigenvalues $d_j^2 + \lambda$ and condition number $(d_1^2 + \lambda)/(d_p^2 + \lambda)$, which the penalty drives toward $1$. Numerical difficulties in ridge appear only when $\lambda$ is tiny relative to $d_1^2$; in that regime the fit is nearly OLS and inherits its problems.

---

### 13. Workflow and software conventions

#### 13.1 The procedure

1. Separate a test set and do not touch it until the end.
2. Encode categorical features as indicator columns. Ridge does not require dropping one level per category: the penalty resolves the collinearity of a full set of indicators, and keeping all levels treats them symmetrically.
3. Choose the $\lambda$ grid on a logarithmic scale from below the smallest $d_j^2$ to above the largest. With standardised columns and an objective divided by $n$, each $d_j^2/n$ is of order $1$, and a grid from $10^{-4}$ to $10^{4}$ covers the range.
4. Run $K$-fold cross-validation or GCV with centring and scaling recomputed inside each fold. Pipelines that bundle the standardiser with the model do this automatically.
5. Refit on the full training set at the chosen $\lambda$, with its own standardisation statistics.
6. Evaluate once on the test set. A $\lambda$ chosen by looking at the test error is not an honest estimate of anything.
7. Store the means, standard deviations, $\hat\beta_0$ and $\hat\beta_\lambda$ together: the model is all four.

#### 13.2 Reading coefficients

A standardised coefficient is the change in $\hat{y}$ per one standard deviation of its feature, with the others held fixed. Ridge shrinks the estimate along the principal directions (§7.1), most strongly the low-variance ones, so the norm of the estimate is understated; an individual coefficient can move either way, as $\hat\beta_2$ does in the table of §4.2. Among a group of correlated features, ridge spreads the shared effect across the group (the two coefficients approaching each other in §4.2), and the split is not identifiable from the data. The usual OLS standard errors do not apply, because the estimator is biased; uncertainty in ridge coefficients is assessed by refitting on bootstrap resamples.

#### 13.3 Software conventions

Implementations agree on the estimator and disagree on the scale of $\lambda$. Three questions settle any conversion: is the RSS divided by $n$; does the penalty carry a factor $\tfrac{1}{2}$; does the software standardise $X$ itself.

| Software | Objective minimised | Standardises $X$ | Relation to $\lambda$ here |
|---|---|---|---|
| scikit-learn `Ridge(alpha)` | $\lVert y - X\beta\rVert^2 + \alpha\lVert\beta\rVert^2$ | no | $\alpha = \lambda$ |
| glmnet `alpha = 0`, penalty `lambda` | $\frac{1}{2n}\lVert y - X\beta\rVert^2 + \frac{\text{lambda}}{2}\lVert\beta\rVert^2$ | yes, by default | $\text{lambda} = \lambda/n$, on standardised columns |
| gradient descent with `weight_decay` $w$, mean loss $\frac{1}{2n}\lVert y - X\beta\rVert^2$ | $\frac{1}{2n}\lVert y - X\beta\rVert^2 + \frac{w}{2}\lVert\beta\rVert^2$ | no | $w = \lambda/n$ |

The glmnet objective multiplied by $2n$ is $\|y - X\beta\|^2 + n\,\text{lambda}\,\|\beta\|^2$, hence the conversion. For a fixed problem, the normalised conventions make $\lambda$ independent of $n$, which is convenient when comparing across datasets; the unnormalised convention is the one used in this lesson. scikit-learn and glmnet fit the intercept without penalty; a gradient-descent implementation decays every parameter it is given, bias included, unless the bias is excluded explicitly, which the frameworks do not do by default. Adaptive optimisers such as Adam apply weight decay in more than one way — added to the gradient, or applied directly to the weights — and the two are not equivalent outside the plain gradient-descent case; Loshchilov and Hutter (2019) is the reference.

---

### 14. Failure modes

- **The truth is sparse.** If only a few of many features matter, ridge shrinks all coefficients and zeroes none (§5.2): the irrelevant features keep small nonzero coefficients that add variance, and the relevant ones are shrunk to pay for it. A penalty on $\sum_j|\beta_j|$ (the lasso) or a combination of the two (the elastic net) is the tool for that case.
- **Features on different scales.** Without standardisation the penalty is a different multiple of $\lambda$ for each coefficient (§5.1) and the fit depends on the units chosen. The derivations hold for any $X$, but a single $\lambda$ is a single penalty strength only when the columns share a scale.
- **A penalised intercept.** Shrinking $\beta_0$ makes the fit depend on the origin of $y$ (§1.2). Centre the response, or use software that leaves the intercept free.
- **$\lambda$ chosen with the test set, or standardisation computed on all data before splitting.** Both let held-out observations shape the fit; the reported error is then an underestimate whose size is unknown.
- **$\lambda$ transplanted from another problem or another library.** The scale of $\lambda$ depends on $n$, on the column scales and on the software convention (§6.3, §13.3). Always re-select.
- **Extrapolation off the principal directions.** Cross-validation selects $\lambda$ for points resembling the training data, where errors in low-variance directions are cheap (§7.4). At points far along such directions those errors are exposed. Neither ridge nor OLS is trustworthy there, but ridge's shrunken coefficients extrapolate less violently.
- **A nonlinear relationship.** Ridge shrinks a linear fit; it does not bend it. Curvature must be supplied through the features — polynomial, spline or interaction terms — and ridge is then the natural way to control the many resulting coefficients.
- **Outliers and heavy-tailed noise.** The squared loss is unchanged by the penalty, so an aberrant observation moves the fit through the same mechanism as in OLS; the penalty lowers its leverage but does not bound its influence. Robust losses are a separate remedy.
- **Reading individual coefficients under collinearity.** The data do not identify how a shared effect divides among correlated features (§3.3); ridge resolves the ambiguity by spreading it evenly, which is a modelling convention rather than a finding.

---

### 15. Implementation notes

A plain array library (NumPy) is all that is needed. The whole of ridge, including the cross-validation curve, is one SVD and a few array operations:

```python
import numpy as np


def ridge_path(X, y, lambdas):
    """Ridge fits for every value of the penalty from one SVD of the features.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix with centred columns, normally also scaled to unit variance.
    y : ndarray of shape (n,)
        Centred response.
    lambdas : sequence of float
        Penalty values, each non-negative.

    Returns
    -------
    dict of ndarray
        ``"beta"``, shape (len(lambdas), p): coefficients at each penalty.
        ``"df"``, shape (len(lambdas),): effective degrees of freedom, counting the
        unpenalised intercept fitted by centring as one.
        ``"rss"``: residual sum of squares.
        ``"loocv"``: leave-one-out mean squared error in closed form, with the
        intercept included in the leverages.
        ``"gcv"``: generalised cross-validation score.
    """
    n = len(y)
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    Uy = U.T @ y
    out = {"beta": [], "df": [], "rss": [], "loocv": [], "gcv": []}
    for lam in lambdas:
        s = d**2 / (d**2 + lam)                       # shrinkage factors
        beta = Vt.T @ (d / (d**2 + lam) * Uy)         # zero where d == 0, for lam > 0
        resid = y - U @ (s * Uy)
        lev = 1.0 / n + (U**2) @ s                    # leverages h_ii, intercept included
        df = 1.0 + s.sum()
        out["beta"].append(beta)
        out["df"].append(df)
        out["rss"].append(resid @ resid)
        out["loocv"].append(np.mean((resid / (1.0 - lev)) ** 2))
        out["gcv"].append(np.mean(resid**2) / (1.0 - df / n) ** 2)
    return {k: np.array(v) for k, v in out.items()}


def fit_ridge(X_raw, y_raw, lam):
    """Standardise with the training statistics, fit ridge, and return a predictor.

    Parameters
    ----------
    X_raw : ndarray of shape (n, p)
        Training features on their original scale.
    y_raw : ndarray of shape (n,)
        Training response on its original scale.
    lam : float
        Penalty applied to the standardised features; non-negative.

    Returns
    -------
    callable
        Maps new rows of shape (m, p) on the original scale to predictions of
        shape (m,), applying the training means and standard deviations before
        the coefficients.
    """
    mu, sd = X_raw.mean(axis=0), X_raw.std(axis=0)
    ybar = y_raw.mean()
    path = ridge_path((X_raw - mu) / sd, y_raw - ybar, [lam])
    beta = path["beta"][0]
    return lambda X_new: ybar + ((X_new - mu) / sd) @ beta
```

Run on the running example with the centred columns unscaled (`X` and `y` of §2 and `lambdas = [0, 0.25, 0.5, 1, 1.5, 2, 5, 10]`), `ridge_path` reproduces the table of §11.5. `fit_ridge` standardises, which for this example divides both columns by $\sqrt{2}$, so `fit_ridge(X_raw, y_raw, lam)` equals the lesson's fit at $\lambda = 2\,$`lam` (§5.1). A library implementation (`sklearn.linear_model.RidgeCV`, which uses the same leave-one-out formula) serves as the answer key. 

---

### 16. Ridge regression in a nutshell

```text
RIDGE_REGRESSION(X_raw, y_raw, K = 10):
    // X_raw: n x p features, y_raw: n responses. Returns a predictor.

    // STEP 1: Candidate penalties: logarithmic grid spanning the squared singular values d_j^2
    //         of the standardised full-data X
    grid = logspace(log10(min_j d_j^2) - 1, log10(max_j d_j^2) + 1, 50)

    // STEP 2: Cross-validate, standardising inside each fold
    split rows at random into K folds
    for lam in grid:
        for k in 1..K:
            mu, sd, ybar = column means, column std devs, response mean of the K-1 training folds
            X_tr = (X_raw[train] - mu) / sd;  y_tr = y_raw[train] - ybar
            beta = solve (X_tr' X_tr + lam I) beta = X_tr' y_tr      // or one SVD for all lam
            errors[lam, k] = mean over held-out rows of (y - ybar - ((X - mu)/sd) beta)^2
        CV[lam] = mean_k errors[lam, k];  SE[lam] = std_k errors[lam, k] / sqrt(K)

    // STEP 3: Select
    lam_min = argmin CV
    lam_1se = largest lam with CV[lam] <= CV[lam_min] + SE[lam_min]   // optional, simpler model

    // STEP 4: Refit on all data at the chosen lam, with its own standardisation
    mu, sd, ybar = statistics of all n rows
    beta = solve (((X_raw - mu)/sd)' ((X_raw - mu)/sd) + lam I) beta = ((X_raw - mu)/sd)' (y_raw - ybar)
    return x -> ybar + ((x - mu)/sd) . beta                          // the model is (mu, sd, ybar, beta)
```

---

## References

**Cited in the text**

- Hoerl, A. E. and Kennard, R. W. (1970). *Ridge regression: biased estimation for nonorthogonal problems.* Technometrics 12(1), 55–67. The estimator, the eigen-analysis of its bias and variance, and the existence theorem of §7.3. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Tikhonov, A. N. (1963). *Solution of incorrectly formulated problems and the regularization method.* Soviet Mathematics Doklady 4, 1035–1038. The same estimator, arrived at independently as a method for ill-posed inverse problems; ridge is also called Tikhonov regularisation.
- Golub, G. H., Heath, M. and Wahba, G. (1979). *Generalized cross-validation as a method for choosing a good ridge parameter.* Technometrics 21(2), 215–223. GCV as used in §11.4, with its rotation-invariance argument. [doi:10.1080/00401706.1979.10489751](https://doi.org/10.1080/00401706.1979.10489751)
- Craven, P. and Wahba, G. (1978). *Smoothing noisy data with spline functions.* Numerische Mathematik 31, 377–403. GCV for smoothing splines, developed alongside the ridge case. [doi:10.1007/BF01404567](https://doi.org/10.1007/BF01404567)
- Allen, D. M. (1974). *The relationship between variable selection and data augmentation and a method for prediction.* Technometrics 16(1), 125–127. The leave-one-out residual formula of §11.3, under the name PRESS. [doi:10.1080/00401706.1974.10489157](https://doi.org/10.1080/00401706.1974.10489157)
- Stone, M. (1974). *Cross-validatory choice and assessment of statistical predictions.* Journal of the Royal Statistical Society B 36(2), 111–147. Cross-validation as a general principle for choosing tuning parameters. [doi:10.1111/j.2517-6161.1974.tb00994.x](https://doi.org/10.1111/j.2517-6161.1974.tb00994.x)
- Saunders, C., Gammerman, A. and Vovk, V. (1998). *Ridge regression learning algorithm in dual variables.* Proceedings of the 15th International Conference on Machine Learning, 515–521. The dual form of §9 and its extension to kernels.
- Loshchilov, I. and Hutter, F. (2019). *Decoupled weight decay regularization.* International Conference on Learning Representations. The distinction between $L_2$ penalty and weight decay under adaptive optimisers, mentioned in §13.3. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

**Background, not cited**

- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning.* 2nd ed., Springer. Section 3.4 covers ridge, the SVD view and the comparison with the lasso; the full text is [free online](https://hastie.su.domains/ElemStatLearn/).
- Friedman, J., Hastie, T. and Tibshirani, R. (2010). *Regularization paths for generalized linear models via coordinate descent.* Journal of Statistical Software 33(1). The glmnet package and the objective convention recorded in §13.3. [doi:10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- Björck, Å. (1996). *Numerical Methods for Least Squares Problems.* SIAM. The numerical linear algebra behind §12: Cholesky, QR, SVD and their conditioning. [doi:10.1137/1.9781611971484](https://doi.org/10.1137/1.9781611971484)
- Pedregosa, F. et al. (2011). *Scikit-learn: machine learning in Python.* Journal of Machine Learning Research 12, 2825–2830. The library whose `Ridge` and `RidgeCV` conventions are recorded in §13.3. [paper](https://jmlr.org/papers/v12/pedregosa11a.html)

---

## Licence

The prose of this lesson is licensed under [Creative Commons Attribution 4.0 International](LICENSE)
(CC BY 4.0): copy, adapt, translate and redistribute it, including commercially, provided you give
credit and indicate what you changed. The code samples it contains are licensed under Apache-2.0,
as is all code in [this repository](../../../LICENSE).
