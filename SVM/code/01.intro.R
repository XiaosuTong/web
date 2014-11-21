# Support Vector Machine #

## Defination ##

### In Short ###

#### What is SVM ####
A support vector machine constructs a **hyperplane or set of hyperplanes in a high- or infinite-dimensional space**, 
which can be used for classification, regression, or other tasks.

#### Why is SVM ####

Original classification problem may be stated in a finite dimensional space, it often happens that 
the sets to discriminate are not linearly separable in that space. For this reason, it was proposed 
that the original finite-dimensional space be mapped into a much higher-dimensional space, presumably
making the separation easier in that space.

The mappings used by SVM schemes are designed to ensure that dot products may be computed easily in 
terms of the variables in the original space, by defining them in terms of a kernel function $k(x,y)$ 
selected to suit the problem.

#### What is optimal ####

A good separation is achieved by the hyperplane that has the largest distance to the nearest training 
data point of any class (so-called functional margin), since in general the larger the margin the lower 
the generalization error of the classifier.

### Motivation ###

In the case of support vector machines, a data point is viewed as a p-dimensional vector (a list of 
p numbers), and we want to know whether we can separate such points with a (p − 1)-dimensional 
hyperplane. This is called a linear classifier. 

There are many hyperplanes that might classify the data. One reasonable choice as the best hyperplane 
is the one that represents the largest separation, or margin, between the two classes. So we choose 
the hyperplane so that the distance from it to the nearest data point on each side is maximized. If 
such a hyperplane exists, it is known as the maximum-margin hyperplane and the linear classifier it 
defines is known as a maximum margin classifier; or equivalently, the perception of optimal stability.

Any hyperplane can be written as the set of points $\mathbf{x}$ satisfying

$\mathbf{w}\cdot\mathbf{x} - b = 0$

## Details ##

### More about SVM ###

An SVM model is a representation of the examples as points in space, mapped so that the examples of
the separate categories are divided by a clear gap that is as wide as possible. New examples are 
then mapped into that same space and predicted to belong to a category based on which side of the 
gap they fall on.

It often happens that the sets to discriminate are not linearly separable in that space. For this 
reason, it was proposed that the original finite-dimensional space be mapped into a much 
higher-dimensional space, presumably making the separation easier in that space.

The hyperplanes in the higher-dimensional space are defined as the set of points whose dot product
with a vector in that space is constant.

### Distance from Point to A Plane ###

First of all, we have to recall that one plane in Euclidean Space can be expressed as:
$$
\omega^Tx + b = 0, \;\;\;\; x \in R^d
$$
Here $\omega$ is the **normal** of the hyperplane. For any point $x_i$ that is on this hyperplane, 
it should have $\omega^Tx_i + b = 0$.

Then a common question is how to calculate the distance from a point outside of plane to this hyperplane.
Suppose the point is $x_0 \in R^d$, and we also can randomly find another point $x_1 \in R^d$ that is on the 
hyperplane. Then we have a vector $x_0 - x_1$, and the distance between $x_0$ and the plane is nothing but
the norm of the projection of vector $x_0 - x_1$ to the **normal** of the plane ($\omega$). And we know that
the projection is just the inner product of two vectors.
$$
Dist = \|Proj_{\omega}(x_0 - x_1)\| = \frac{\|\omega^T(x_0 - x_1)\|}{\|\omega\|} = \frac{\|\omega^Tx_0 + b\|}{\|\omega\|}
$$


### Functional and Geometric Margins ###

The geometric margin is just a scaled version of the functional margin.

You can think the functional margin, just as a testing function that will tell you whether a particular 
point is properly classified or not. And the geometric margin is functional margin scaled by $\|\omega\|$

the result would be positive for properly classified points and negative otherwise. If you scale 
that by $\|\omega\|$ then you will have the geometric margin. 
 
Why does the geometric margin exists?

Well to maximize the margin you need more that just the sign, you need to have a notion of magnitude,
the functional margin would give you a number but without a reference you can't tell if the point is 
actually far away or close to the decision plane. The geometric margin is telling you not only if 
the point is properly classified or not, but the magnitude of that distance in term of units of 
$\|\omega\|$ 

### Primal Optimization ###

First of all, consider a problem of the following form:

$$
\begin{aligned}
\min_{\omega} & f(\omega) \\
s.t. & h_i(\omega) = 0, \;\;\; i = 1, \cdots, l 
\end{aligned}
$$

It is obvious that the method of Lagrange multipliers can be used to solve it. In this method, we
define the **Lagrangian** to be
$$
\mathcal{L}(\omega, \beta) = f(\omega) + \sum_{i=1}^{l}\beta_ih_i(\omega)
$$
Here the $\beta$'s are called the **Lagrangian multipliers**. We would then find and set $\mathcal{L}$'s
partial derivatives to zero:
$$
\frac{\partial\mathcal{L}}{\partial\omega_i} = 0; \;\;\; \frac{\partial\mathcal{L}}{\partial\beta_i} = 0
$$
and solve for $\omega$ and $\beta$, since the second derivative will give us $h_i(\omega) = 0$

But sometimes, the constraint are not just $h_i(\omega) = 0$. We may have inequality and equality constraints.
So the problem will become as following, which we call the **primal** optimization problem:
$$
\begin{aligned}
\min_{\omega} & f(\omega) \\
s.t. & g_i(\omega) \le 0, \;\;\; i = 1, \cdots, k \\
& h_i(\omega) = 0, \;\;\; i = 1, \cdots, l 
\end{aligned}
$$
To solve it, we start by defining the **generalized Lagrangian**
$$
\mathcal{L}(\omega, \beta) = f(\omega) + \sum_{i=1}^{k}\alpha_ig_i(\omega) + \sum_{i=1}^{l}\beta_ih_i(\omega)
$$
Here, $\alpha_i$ and $\beta_i$ are Lagrange multipliers. And for the multipliers associated with inequality,
we should have further constraint: $\alpha_i \ge 0$. Consider the quantity:
$$
\theta_{\mathcal{P}}(\omega) = \max_{\alpha, \beta: \alpha_i \ge 0} \mathcal{L}(\omega, \alpha, \beta)
$$
Here, the "$\mathcal{P}$" subscript stands for "primal". For this quantity, we fix the $\omega$, and vary 
$\alpha$ and $\beta$. If the given $omega$ violates any primal constraints for some $i$, the we will have 
$$
\theta_{\mathcal{P}}(\omega) = f(\omega) + \max_{\alpha} \sum_{i=1}^{k}\alpha_ig_i(\omega) + \max_{\beta} 
\sum_{i=1}^{l}\beta_ih_i(\omega) =  \infty
$$

### Dual Optimization ###

### Kernel ###

Kernel methods owe their name to the use of kernel functions, which enable them to operate in a 
high-dimensional, implicit feature space without ever computing the coordinates of the data in that 
space, but rather by simply computing the inner products between the images of all pairs of data in the
feature space. This operation is often computationally cheaper than the explicit computation of the 
coordinates. This approach is called the "kernel trick". 

## Non-linearly Separable ##

### "Slack" Variables ###

Previously, we introduced how to separate points by linearly separable hyperplane. But it is possible
that the large margin solution is better even though one constraint is violated.

So we permit to have functional margin less than 1 for some cases, such that 
$$y_i(\omega^Tx_i + b) \ge 1 - \xi_i$$
$$\xi_i \ge 0$$ 
This $\xi_i$ are called slack variables. For the support vectors, the functional 
margin is 
$$\gamma_i = y_i(\omega^Tx_i + b) = 1,$$ 
which means $\xi_i = 0$, and distance from support vectors to classification hyperplane is $\frac{1}{\|\omega\|}$.
Originally, within either class, distance from any point to the classification hyperplane is larger or equal to 
the $\frac{1}{\|\omega\|}$. But now since we permit have some cases that functional margin is less than 1, we will
have some cases that the distance from those points to the classification hyperplane are less than 1 or maybe 
even larger than 1 but on the other side. So in other words, there are two different situations for slack variable.

#### Margin violation ####

For this type of situation, we have **$0 < \xi_i \le 1$**. This means the distance between this type of points
and the hyperplane $\omega^Tx + b$ is:
$$ 0 \le \frac{1 - \xi_i}{\|\omega\|} < \frac{1}{\|\omega\|}$$
The above distance greater or equal to zero means this point is still on the correct side of classification 
hyperplane. On the other hand, it should smaller than $\frac{1}{\|\omega\|}$, which is the distance from 
support vector.

In summary, margin violation means we will permit a correct classification point but with smaller margin than
support vectors. One thing to keep in mind that the distance we are talking about here do not have a direction.
Which means we are talking about the distance for a point from a given class to the hyperplane. Later on we 
will see a negative distance, that means the distance is from the other side.

#### Misclassified ####

For this type of situation, we have **$\xi_i > 1$**. This means the distance between this type of points and the
hyperplane $\omega^Tx + b$ is:
$$
\frac{1 - \xi_i}{\|\omega\|} < - \frac{1}{\|\omega\|}
$$
The magnitude of this distance is actually greater than $\frac{1}{\|\omega\|}$, which is because the distance
is larger than the support vectors. However the negative sign with the distance means this point is misclassified
on the other side.

### Regularization ###

Since we permit margin violation or misclassified points in our sample, we should add penalty on those points.
One thing should be noticed that if the $\xi_i$ is sufficiently large, then constraint for every point can be
satisfied.

$$ 
\begin{aligned}
& \min_{\gamma, \omega, b} \frac{1}{2}||\omega||^2 + C\sum_{i=1}^{m} \xi_i \\
s.t. & y_i(\omega^Tx_i + b) \ge 1 - \xi_i \\
& \xi_i \ge 0
\end{aligned}
$$

C is the regularization parameter, 
- If C is small, then all constraints are easily to be ignored. It means we can have relatively large $\xi_i$
and to have a large margin with many cases of margin violation and misclassification.
- If C is large, then all constraints are hard to be ignored. It means we can have small $\xi_i$ and to have a
narrow margin in order to avoid margin violation or misclassification.
- If C is $\infty$, then all constraints are enforced to be satisfied, which called hard margin. In this extreme
case, we do not have soft margin solution, and which goes back to original SVM.

Then the optimization problem can be reformed using **Lagrange multipliers** method. Let us define 
the **Lagrangian** to be:

$$
\begin{aligned}
\mathcal{L}(\omega, b, \xi, \alpha, r) = \frac{1}{2}\omega^T\omega + C\sum_{i=1}^m\xi_i - 
\sum_{i=1}^m\alpha_i[y_i(x^Tx + b) - 1 + \xi_i] - \sum_{i=1}^mr_i\xi_i
\end{aligned}
$$

Here $\alpha_i$ and $r_i$ are our Lagrange multipliers.
All optimal solutions must satisfy Karush-Kuhn-Tucker (KKT) conditions:

$$
\begin{aligned}
\frac{\partial}{\partial \omega}\mathcal{L} = 0 \\
\frac{\partial}{\partial b}\mathcal{L} = \sum_{i=1}^m \alpha_i y_i = 0 \\
\frac{\partial}{\partial \xi_i}\mathcal{L} = C - \alpha_i - r_i = 0 & \;\;\; i = 1, 2, \cdots, m \\
\alpha_i[y_i(x^Tx + b) - 1 + \xi_i] = 0 & \;\;\; i = 1, 2, \cdots, m \\
r_i\xi_i = 0 & \;\;\; i = 1, 2, \cdots, m \\
\alpha_i \ge 0 & \;\;\; i = 1, 2, \cdots, m \\
r_i \ge 0 & \;\;\; i = 1, 2, \cdots, m \\
-\xi_i \le 0 & \;\;\; i = 1, 2, \cdots, m \\
-[y_i(x^Tx + b) - 1 + \xi_i] \le 0 & \;\;\; i = 1, 2, \cdots, m
\end{aligned}
$$

First we look at the third condition. It can be rewrote as $\alpha_i = C - r_i$, recall that $r_i \ge 0$.
So we can get the condition for $\alpha_i$ as:
$$0 \le \alpha_i \le C$$

The optimality conditions are both necessary and sufficient. If we have
$C$, $\xi$, b, $\alpha$, and r satisfying the above conditions, we know that they represent optimal 
solutions to the primal and dual problems.

Also among all conditions, the KKT dual-complementarity conditions are:
$$
\begin{aligned}
\alpha_i[y_i(x^Tx + b) - 1 + \xi_i] = 0 & \;\;\; i = 1, 2, \cdots, m \\
r_i\xi_i = 0 & \;\;\; i = 1, 2, \cdots, m \\
\end{aligned}
$$
Then, we can have 
- If $\alpha_i = 0$, then $y_i(x^Tx + b) - 1 + \xi_i > 0$, according to the third condition of KKT
$r_i = C \Rightarrow \xi_i = 0 \Rightarrow y_i(x^Tx + b) > 1$
- If $\alpha_i = C$, then according to the third condition of KKT, $r_i = 0 \Rightarrow \xi_i >0$, 
$y_i(x^Tx + b) - 1 + \xi_i = 0 \Rightarrow y_i(x^Tx + b) < 1$
- If $0 < \alpha_i < C$, then $y_i(x^Tx + b) - 1 + \xi_i = 0$, and $r_i \ne 0 \Rightarrow \xi_i = 0$.
So $y_i(x^Tx + b) = 1$
## Algorithm ##
