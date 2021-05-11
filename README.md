## Optimization algorithms

### Gradient Descent

Gradient descent algorithm with backtracking line search.

<img src="pics/demo/gradient_descent.jpg" style="zoom:30%;" />
<img src="pics/demo/backtracking.jpg" style="zoom: 20%;" />

Example function:
$$f(x) = \sum_{i=1}^m[exp(a_i^Tx)+exp(-a_i^Tx)]$$

Code: gradient_descent.py

Grid search for parameters: $\alpha, \beta$ in $\{0.03, 0.1, 0.3, 0.5\}$

Take steps:

	[[70. 66. 28. 35.]
	 [70. 66. 28. 27.]
	 [74. 68. 27. 27.]
	 [82. 73. 29. 31.]]
So we choose $\alpha=\beta=0.3$ in the following experiments.

Plot figures:

* $\|\nabla f(\mathbf x)\|_2$

  <img src="pics/hw8/gd-gradient.png" style="zoom: 33%;" />

* step length

  <img src="pics/hw8/gd-step_l.png" style="zoom:33%;" />

* $f-p*$

  <img src="pics/hw8/gd-f-p.png" style="zoom:33%;" />

### Steepest Descent

Steepest descent algorithm in $l_\infty$-norm.

$\Delta\mathbf{x}_{nsd}=\arg\min_\mathbf{v}\{\nabla f(\mathbf{x})^T\mathbf{v}|\|\mathbf{v}\|_\infty=1\}$

Set $\Delta{\mathbf{x}_{i}} = -sign(\nabla f(\mathbf{x})_i)$.

<img src="pics/demo/steepest_descent.jpg" style="zoom: 20%;" />

Example function:
$$f(x) = \sum_{i=1}^m[exp(a_i^Tx)+exp(-a_i^Tx)]$$

Code: steepest_descent.py

```python
grad = grad_func(x)
dir = np.zeros(grad.shape)
i = np.argmax(np.abs(grad))
dir[i] = -grad[i]
```

Grid search for parameters: $\alpha, \beta$ in $\{0.03, 0.1, 0.3, 0.5\}$

Take steps:

	[[207. 404.  80.  74.]
	 [207. 404.  82.  53.]
	 [228. 502. 159.  49.]
	 [237. 719. 161. 149.]]

So we choose $\alpha=0.1,\beta=0.5$ in the following experiments.

Plot figures:

* $\|\nabla f(\mathbf x)\|_2$

  <img src="pics/hw8/st-gradient.png" style="zoom: 33%;" />

* step length

  <img src="pics/hw8/st-step_l.png" style="zoom:33%;" />

* $f-p*$

  <img src="pics/hw8/st-f-p.png" style="zoom:33%;" />

### Newton Methods

#### Damped Newton Method

<img src="pics/demo/damped_newton.jpg" alt="Damped Newtion" style="zoom:30%;" />

Example function:
$$f(\mathbf{x}) = 100(x_2-x_1^2)^2+(1-x_1)^2$$

Code: newton.py

Grid search for parameters: $\alpha, \beta$ in $\{0.03, 0.1, 0.3, 0.5\}$

Take steps:

	[[223.  79.  25.  24.]
	 [227.  81.  25.  25.]
	 [243.  89.  27.  25.]
	 [295. 103.  44.  26.]]

So we choose $\alpha=0.03,\beta=0.5$ in the following experiments.

Plot figure:

* $log(f(\mathbf{x}^{(k)})-p^*)$

  <img src="pics/hw8/dm-log(f-p).png" style="zoom:33%;" />

From the figure, we can see that damped newton method converges fast when it is close to the optimal solution.

Damped Newton time: 0.0020020008087158203s

#### Gauss-Newton Method

<img src="pics/demo/gauss_newton.jpg" style="zoom:30%;" />

```python
dir = -solve_linear(np.dot(J(x).T, J(x)), np.dot(J(x).T, r(x)))
```

Gauss-Newton time: 0.0s

The algorithm converges in 3 steps.

<img src="pics/hw8/gn-log(f-p).png" style="zoom:33%;" />


### Quasi-Newton Methods

#### Davidon-Fletcher-Powell (DFP)

Algorithm:

<img src="pics/demo/quasi-dfp.jpg" style="zoom:30%;" />

Example function:

$f(\mathbf{x}) = x_1^4/4 + x_2^2/2 - x_1x_2+x_1-x_2$

Code: dfp.py

The initial point $(0,0)^T$ converges to $(-1, 0)^T$ while $(1.5, 1)^T$ converges to $(1, 2)^T$. Because the function is not convex, and there exists different local optimum.


#### Broyden-Fletcher-Goldfarb-Shanno (BFGS)

Algorithm:

<img src="pics/demo/bfgs.jpg" style="zoom:30%;" />

Example function:

$f(\mathbf{x}) = (3-x_1)^2+7(x_2-x_1^2)^2+9(x_3-x_1-x_2^2)^2$

Code: bfgs.py


#### Limited-Memory BFGS

Algorithm:

<img src="pics/demo/lbfgs.jpg" style="zoom:30%;" />

Example function: Rosenbrock function.

Code: lbfgs.py

Note: Wolfe conditions in line search is not implemented. May be bugs in the code.

<img src="pics/hw9/lbfgs-1.png" style="zoom:40%;" />
<img src="pics/hw9/lbfgs-5.png" style="zoom:40%;" />
<img src="pics/hw9/lbfgs-10.png" style="zoom:40%;" />
<img src="pics/hw9/lbfgs-30.png" style="zoom:40%;" />


### Majorization Minimization

Algorithm:

<img src="pics/demo/majorization.jpg" style="zoom:30%;" />

Code: major.py

Example function: $\frac{1}{2}\|\mathbf{Ax}-\mathbf{b}\|^2 + \lambda \|\mathbf{x}\|_1$.

Majorant function1: Lipschitz gradient majorant function of the first term.

Majorant function2:  $\min_{\mathbf{d}>\mathbf{0}}\frac{1}{2}(\mathbf{x}^T\mathbf{D}\mathbf{x}+\mathbf{1}^T\mathbf{D}^{-1}\mathbf{1})$ for $\lambda \|\mathbf{x}\|_1$.

<img src="pics/hw9/major1.png" style="zoom:40%;" />
<img src="pics/hw9/major2.png" style="zoom:40%;" />