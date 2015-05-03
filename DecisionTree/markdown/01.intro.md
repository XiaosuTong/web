# Decision Tree #

## Defination ##

### In Short ###

#### What is DT ####

A decision tree is a flowchart-like structure in which each internal node represents a "test" on 
an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome 
of the test and each leaf node represents a class label (decision taken after computing all 
attributes). The paths from root to leaf represents classification rules.

Tree models where the target variable can take a finite set of values are called classification 
trees. In these tree structures, leaves represent class labels and branches represent conjunctions 
of features that lead to those class labels. Decision trees where the target variable can take 
continuous values (typically real numbers) are called regression trees.

#### Why is DT ####

### Motivation ###

## Details ##

### More about DT ###

A decision tree is a simple representation for classifying examples. Decision tree learning is one 
of the most successful techniques for supervised classification learning. Assume that all of the 
features have finite discrete domains, and there is a single target feature called the 
classification ($Y$). Each element of the domain of the classification is called a class. A decision tree 
or a classification tree is a tree in which each internal (non-leaf) node is labeled with an input 
feature ($X_i$). 

A tree can be "learned" by splitting the source set into subsets based on an attribute value test. 
This process is repeated on each derived subset in a recursive manner called recursive partitioning. 
The recursion is completed when the subset at a node has all the same value of the target variable, 
or when splitting no longer adds value to the predictions. This process of top-down induction of 
decision trees (TDIDT) is an example of a greedy algorithm, and it is by far the most common strategy 
for learning decision trees from data.

Here is a very simple pseudocode:

$$
\begin{aligned}
 & \textbf{GROWTREE(S)} \\
 & \textbf{if} ( y = 0 \;\; \text{for} \;\; \text{all} (x,y) \in S) \\
 & \;\;\;\; \textbf{return} \;\; \text{new} \;\; \text{leaf(0)} \\
 & \textbf{else} \;\; \textbf{if} ( y = 1 \;\; \text{for} \;\; \text{all} (x,y) \in S) \\
 & \;\;\;\; \textbf{return} \;\; \text{new} \;\; \text{leaf(1)} \\
 & \textbf{else} \\
 & \;\;\;\; \text{choose best attribute} \;\; x_j \\
 & \;\;\;\; S_0 = \text{all} (x,y) \in S \text{with} x_j = 0 \\
 & \;\;\;\; S_1 = \text{all} (x,y) \in S \text{with} x_j = 1 \\
 & \;\;\;\; \textbf{return} \text{new} \;\; \text{node}(x_j, \textbf{GROWTREE}(S_0), \textbf{GROWTREE}(S_1))
\end{aligned}
$$

### Types ###

#### Types of tree ####

Decision trees used in data mining are of two main types:

- Classification tree analysis is when the predicted outcome is the class to which the data belongs.
- Regression tree analysis is when the predicted outcome can be considered a real number (e.g. the 
price of a house, or a patient’s length of stay in a hospital).

The term Classification And Regression Tree (CART) analysis is an umbrella term used to refer to both 
of the above procedures. Trees used for regression and trees used for classification have some 
similarities - but also some differences, such as the procedure used to determine where to split.

Decision tree learning is the construction of a decision tree from class-labeled training tuples. A 
decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a test on 
an attribute, each branch represents the outcome of a test, and each leaf (or terminal) node holds 
a class label. The topmost node in a tree is the root node.

#### Types of algorithms ####

There are many specific decision-tree algorithms. Notable ones include:

- ID3 (Iterative Dichotomiser 3)
- C4.5 (successor of ID3)
- CART (Classification And Regression Tree)
- CHAID (CHi-squared Automatic Interaction Detector). Performs multi-level splits when computing 
classification trees.
- MARS: extends decision trees to handle numerical data better.
- Conditional Inference Trees. Statistics-based approach that uses non-parametric tests as splitting 
criteria, corrected for multiple testing to avoid overfitting. This approach results in unbiased 
predictor selection and does not require pruning.

#### Types of decision nodes ####

Depending upon the decision node of a tree, decision trees for continuous attributes can be divided
into three categories:

- **Univariate decision trees**
At each node, they take the decision on one attribute. They create orthogonal splits (orthogonal to 
the attributes). They are computationally efficient, however, if the decision boundaries are not
orthogonal to a attribute’s axis, they are generally not very accurate. C4.5 and CART are examples 
of these kinds of trees.

- **Linear Multivariate decision trees**
It overcome the representational limitation of univariate decision trees. Linear multivariate 
decision trees allow the node to test a linear combination of the numeric attributes. This test can 
be presented as:
$$
\sum_{i=1}^{m} c_ix_i \le \theta
$$
where $x_i$ are the numeric attributes, $c_i$ are the corresponding real valued coefficients, and 
$\theta$ is a numeric constant. These trees are also called oblique decision trees as they uses 
oblique (non-axis-parallel) hyperplanes to partition the data.

- **Omnivariate decision trees**
At each node, they can take decisions on the nonlinear combination of attributes. They have better
representational power than univariate decision trees and linear univariate decision trees.

### Pick node ###

This session can also be phrased as how to choose the attribute based on which the data will be
splited. It is not hard to notice how we choose the attributes or nodes totally effects the performance
of the tree. First of all, let us have a look at several possible methods to calculate the impurity
of a set with several levels value, which is the case for our target variable $Y$.

#### Regression error ####

If we are working with the regression tree that our target $Y$ is a continuous variable. The impurity
of one set of examples would be just the MSE:
$$
\text{Impurity}(D) = \frac{1}{N} \sum_{i=1}^N (y_i - \bar y)^2
$$
Here everytime we got a split of dataset, we use the mean of $Y$ in that subset to represent the 
value of $y$ in that subset.

After we calculated the error of each subsets, we will try to learn branching criteria:
$$
b(X) = \arg \min \sum_{c=1}^m \frac{\|D_c\|}{N}impurity(D_c)
$$

#### Variance reduction ####


#### Classification error ####

The most naive way to choose the attributes can be:
$$
\begin{aligned}
 & \textbf{CHOOSEBESTATTRIBUTE(S)} \\
 & \text{choose j to minimize} J_j \text{computed as follows}: \\
 & \;\;\;\;\;\; S_0 = \text{all} \;\; (x,y) \in S \;\; \text{with} x_j = 0 \\
 & \;\;\;\;\;\; S_1 = \text{all} \;\; (x,y) \in S \;\; \text{with} x_j = 1 \\
 & \;\;\;\;\;\; y_0 = \text{the most common value of y in} \; S_0 \\
 & \;\;\;\;\;\; y_1 = \text{the most common value of y in} \; S_1 \\
 & \;\;\;\;\;\; J_0 = \text{number of example (x,y)} \;\; \in S_0 \;\; \text{with} y \ne y_0 \\
 & \;\;\;\;\;\; J_1 = \text{number of example (x,y)} \;\; \in S_1 \;\; \text{with} y \ne y_1 \\
 & \;\;\;\;\;\; J_j = J_0 + J_1 (\text{total number of misclassification})
\end{aligned}
$$
In this method, we just simply predict all examples in one subset to be same as the majority in 
that subset. Then calculate the misclassification count in two children. We can do a greedy search 
to find which attribute can give us the lost misclassification rate.

Unfortunately, this measure does not always work well, because it does not detect where we are
making "progress" toward a good tree.

Suppose we have examples that belongs to two classes based on $y$ value. 20 in one class, 10 in another.
We then split the data based on maybe $x_1$. Now we have 12/8 in subset one, and 8/2 in subset two. 
Let's assume that next step we split the two subsets based on $x_2$ and $x_3$ respectively, and we got
12/0, 0/8 for the first subset, and 8/0, 0/2 for the second subset. So we found that from original
dataset to the first splitting, the error rate did not change which are all 10, which means we did 
not make any progress. But the fact is we are.

After we calculated the error of each subsets, we will try to learn branching criteria:
$$
b(X) = \arg \min \sum_{c=1}^m \frac{\|D_c\|}{N}impurity(D_c)
$$

#### Information gain ####

Another method to choose the best attribute is based on **Information Theory**.

For a given random variable $V$, discrete for example, the **surprise** of each value of $V$ is defined
to be:
$$
S(V=v) = - \log P(V=v)
$$
Then the **Entropy** of random variable $V$ would be:
$$
H(V) = \sum_{v} - P(V=v) \log P(V=v) = E_{V}(- \log P(V))
$$
So the Entropy of random variable is nothing but the expectation value of the surprise function.

What we will use to help us to choose the attribute is the **Mutual Information**. Assume we have
two random variable $Y$ and $X$. The mutual information between $Y$ and $X$ is the amount of information
we learn about $Y$ by knowing the value of $X$ (it is symmetric). It is computed as follows:
$$
\begin{aligned}
I(Y;X) & = H(Y) - \sum_{x}P(X=x)H(Y|X=x) \\
& = H(Y) - \sum_{x}P(X=x) \sum_{y} - P(Y=y|X=x) \log P(Y=y|X=x) \\
& = H(Y) - \sum_{x}\sum_{y} P(x,y) \log P(Y=y|X=x) \\
& = H(Y) - E_{X,Y} (\log P(Y=y|X=x))
\end{aligned}
$$

Here is one problem, if attribute has many values or levels, Gain will select it because it can perfect
split our data. One approach is using **GainRatio** instead:
$$
\begin{aligned}
& GainRatio(S,A) = \frac{I(S;A)}{SplitInformation(S,A)} \\
& SplitInformation(S,A) = - \sum_{i=1}^c \frac{|S_i|}{|S|}\log\frac{|S_i|}{|S|}
\end{aligned}
$$
where $S_i$ is subset of $S$ for which $A$ has value $v_i$. This SplitInformation is nothing but the
sum of Entropy of target variable $Y$ over all subsets based on $X$ splitting variable. Finally, 
we just learn branching criteria by minimize the GainRatio.

#### Gini impurity ####

Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly 
labeled if it were randomly labeled according to the distribution of labels in the subset. Gini 
impurity can be computed by summing the probability of each item being chosen times the probability 
of a mistake in categorizing that item. It reaches its minimum (zero) when all cases in the node 
fall into a single target category.

To compute Gini impurity for a set of items, suppose i $\in$ $\{1, 2, \cdots, m\}$, which is all
possible levels of $Y$. And let $f_i$ be the fraction of items labeled with value
$i$ in the set.
$$
\begin{aligned}
I_G(f) & = \sum_{i=1}^m f_i(1 - f_i) = 1 - \sum_{i=1}^m f_i^2 \\
& = 1 - \sum_{i=1}^m \left( \frac{\sum_{n=1}^N\|y_n = m\|}{N} \right)^2
\end{aligned}
$$
After we calculated the Gini impurity of each subsets, we will try to learn branching criteria:
$$
b(X) = \arg \min \sum_{c=1}^m \frac{\|D_c\|}{N}impurity(D_c)
$$ 

### Missing Value ###

What if some examples are misiing values of A? There are several options can be considered, they 
all based on imputing the missing value:

- If node n tests A, assign most common value of A among other examples at node n.
- Assign most common value of A among other examples with same target value.
- Assign probability $p_i$ to each possible value $v_i$ of $A$. The example that has $A$ missing
will be splited to m examples if $A$ has m different levels. Now those examples will have 
probability as the value of $Y$ instead of 1 or 0 as before. Classify new examples in same fashion.

### Overfitting ###

Let us revisit the overfitting defination again here. Consider error of hypothesis $h$ over

- training data: $error_T(h)$
- entire distribution $D$ of data: $error_D(h)$

Hypothesis $h \in H$ **overfits** training data if there is an alternative hypothesis $h' \in H$
such that
$$
error_T(h) < error_T(h')
$$
and
$$
error_D(h) > error_D(h')
$$
So it is saying that our hypothesis or fitted model based on training data is not doing a good job 
on validation data or in general. The model is too complex without any generalization. In the 
decision tree case, the complexity of a tree is defined by the number of levels of the tree. The
more levels we have in the tree, the more complex the tree is. More levels means we choose more
variables to split the data. If we have a lot of levels, we will end up with very few examples in
each leaf which is not the generalization case.

How can we avoid overfitting in decision tree?

- Stop growing when data splitting is not statistically significant
- Grow a full tree, then post-prune
- Pre-prune the tree before grow the node

Regardless of whether the correct tree size is found by stopping early or by post-pruning, a key 
question is what criterion is to be used to determine the correct final tree size. Approaches include:

- Use a separate set of examples, distinct from the training examples, validation dataset to evaluate
the utility of post-pruning nodes from the tree.
- Use all the available data for training, but apply a statistical test to estimate whether expanding
or pruning a particular node is likely to produce an improvement beyond the training set. For example
a chi-square test to estimate whether further expanding a node is likely to improve performance over
the entire instance distribution, or only on the current sample of training data.
- Add complexity penalty to performance measure. Use an explicit measure of the complexity for
encoding the training examples and the decision tree.

#### Reduced error pruning ####

In summary, post-pruning is bettern than pre-pruning.

## Random Forests ##

### Blending ###

Suppose we have $h_1, \cdots, h_T$ prediction for the same event which using different hypothesis or
different model. Be more specifically $h_i$ can be the classification result ($-1$ or $1$) from different
models.

#### Type ####

- Select the most trust-worthy hypothesis, which is the most basic method: model selection
based on evaluation dataset, cross validation.
  
- mix the predictions from all the hypothesis uniformly

- mix the predictions from all the hypothesis non-uniformly

- combine the predictions conditionally

#### Selection ####

$$ H(x) = h_{t_*}(x) \;\;\; s.t. \;\;\; t_* = \arg\min_{t} MSE(eval) $$

#### Uniform blending ####

For known $h_t$ each with weight 1.
$$
H(x) = \sum_{t=1}^T 1 \cdot h_t(x)
$$
For regression, uniform blending can be:
$$
H(x) = \frac{1}{T} \sum_{t=1}^T h_t(x)
$$
very different $h_t$, average could be more accurate than individual. Diversity is better.

**Theory**
Suppose the true value of prediction for given $x$ is $f(x)$, we can examine the mean square error as follow:
$$
\begin{aligned}
\frac{1}{T} \sum_{t=1}^T( h_t(x) - f(x))^2 & = \frac{1}{T} \sum_{t=1}^T (h_t^2(x) - 2h_t(x)f(x) + f^2(x)) \\
& = \frac{1}{T} \sum_{t=1}^T h_t^2(x) - 2f(x)H(x) + f^2(x) \\
& = \frac{1}{T} \sum_{t=1}^T h_t^2(x) - H^2(x) + ( H(x) - f(x))^2 \\
& = \frac{1}{T} \sum_{t=1}^T h_t^2(x) - 2H^2(x) + H^2(x) + (H(x) - f(x))^2 \\
& = \frac{1}{T} \sum_{t=1}^T ( h_t^2(x) - 2H(x)h_t(x) + H^2(x)) + (H(x) - f(x))^2 \\
& = \frac{1}{T} \sum_{t=1}^T (h_t(x) - H(x))^2 + (H(x) - f(x))^2
\end{aligned}
$$
For any $x$ we can still get the same formula as above only replacing to be the expectation:
$$
\frac{1}{T} \sum_{t=1}^T E_x(h_t - f)^2 = \frac{1}{T} \sum_{t=1}^T E_x(h_t - G)^2 + E_x(H - f)^2 \\
\frac{1}{T} \sum_{t=1}^T E_x(h_t - f)^2 \ge E_x(H - f)^2
$$
This is nothing but a bias-variance decomposition of mean square error. By doing blending, we are actually
trying to reduce the variance 

#### Linear blending ####

The difference between linear blending and uniform blending is that we assign different weights to the
known $h_t(x)$.
$$
H(x) = \sum_{t=1}^T \alpha_t \cdot h_t(x)
$$
There is a restriction which is $\alpha_t \ge 0$. But this restriction can be removed if we are facing
a classification. If $\alpha_t < 0$ then we just set $\alpha_th_t(x) = |\alpha_t|\cdot(-h_t(x))$. So we
just flip the classification result from $h_t(x)$.

Without the positive restriction of $\alpha_t$, the original problem of linear blending can be solved as
$$
\alpha = \arg\min \frac{1}{N} \sum_{n=1}^N \left( y_n - \sum_{t=1}^T \alpha_th_t(x_n) \right)^2
$$  
This is just a linear regression of $y_n$ to the $x_n$ after transformation. If we assume the $x_n$ is
in dimension d, then we just transformed $x_n$ into $h(x_n)$ within dimension T.

### Bagging ###

Bagging sometimes also is called as Bootstrap Aggregation. In the blending session, we assumed we have
already knew all the $h_t$, then we aggregated all those known $h_t$ together.

The diversity of $h_t$ is important, it will give us better result with larger diversity.
- diversity can be different models
- diversity can be different tuning parameters
- diversity can be the different starting point in the algorithm
- diversity can be the data randomness

Data randomness can be achieved by bootstrapping: $D_t$ re-sample N examples from the original dataset $D$
uniformly with replacement.

Then we apply the model $h$ to the $D_t$ to get the estimate $h_t$, and blending can be used to average
the $h_t$.

### Boosting ###

Boosting refers to a general and provably effefctive method of producing a very accurate prediction
rule by combining rough and moderately inaccurate hypothesis.

#### Background ####

Rough and moderately inaccurate hypothesis is a very vague defination with respect to the machine 
learning hypothesis. If we want to define it with more accuracy, we need some basic idea about PAC 
learning model. Boosting actually has its roots in a theoretical framework for studying machine 
learning called the “PAC” learning model, due to Valiant. Kearns and Valiant were the first to 
pose the question of whether a “weak” learning algorithm which performs just slightly better than 
random guessing in the PAC model can be “boosted” into an arbitrarily accurate “strong” learning 
algorithm.

Schapire came up with the first provable polynomial-time boosting algorithm in 1989. Paper showed 
that any weak learning algorithm can be efficiently transformed or boosted into a strong learning 
algorithm. A year later, Freund presented the "boost-by-majority" algorithm that is considerably more
efficient than Schapire's developed a much more efficient boosting algorithm which, although optimal 
in a certain sense, nevertheless suffered from certain practical drawbacks.

Both algorithms work by calling a given weak learning algorithm **WeakLearn** multiple times, each time 
presenting it with a different distribution over the domain $X$, and finally combining all of the 
generated hypotheses into a single hypothesis. The intuitive idea is to alter the distribution over 
the domain $X$ in a way that increases the probability of the **harder** parts of the space, thus 
forcing the weak learner to generate new hypotheses that make less mistakes on these parts.

However, an important, practical deficiency of the boost-by-majority algorithm is the requirement 
that the bias of the weak learning algorithm **WeakLearn** be known ahead of time. Not only is this 
worst-case bias usually unknown in practice, but the bias that can be achieved by **WeakLearn**
will typically vary considerably from one distribution to the next. Unfortunately, the boost-by-majority 
algorithm cannot take advantage of hypotheses computed by **WeakLearn** with error significantly 
smaller than the presumed worst-case bias of $\frac{1}{2} - \gamma$.

#### AdaBoost ####

**AdaBoost** is a new boosting algorithm developed by Freund after the **boost-by-majority** algorithm. It
is a boosting by sampling method.

We assume that a sequence of $N$ traning examples $(x_1, y_1), \cdots, (x_N, y_N)$ is drawn randomly from
$X \times Y$ according to fixed but unknown distribution $\mathscr{P}$. We use boosting to find a hypothesis
$h_f$ which is consistent with most of the sample (i.e. $h_f(x_i) = y_i$ for most $1 \le i \le N$). In
general, a hypothesis which is accurate on the training set might not be accurate on examples outside the
training set; this problem is sometimes referred to as overfitting. Often, overfitting can be avoided by
restricting the hypothesis to be simple so that generalization error is closed to the empirical error on 
the training dataset. Or we can view this issue from the prespective of bagging. In the bagging method,
we are shown that the final aggregation is better if the diversity of each $h_t$ is large. Here the we can
increase the diversity of each **WeakLearn** by restricting them to be simple.

AdaBoost calls a given weak or base learning algorithm repeatedly in a series of
rounds $t=1,\cdots, T$. One of the main ideas of the algorithm is to maintain a distribution or set of
weights over the training dataset. The weight of this distribution on training example $i$ on round $t$ is
denoted $D_t(i)$. Initially, all weights are set equally, but on each round, the weights of incorrectly
classified examples are increased so that the weak learner is forced to focus on the hard examples
in the training set.

First let us assume all $h_t$ are continuous function mapping $x_i$ to $[0,1]$. This assumption is valid if
we are considering logisitic regression as the weak hypotheses. $h_t$ now is nothing but the fitted 
probability of success $\hat \pi_i$.
$$
\begin{aligned}
\textbf{Input:} & \;\; \text{sequence of N labeled examples} \;\; (x_1, y_1), \cdots, (x_N, y_N) \\
& \;\; \text{distribution D over the N examples} \\
& \;\; \text{week learning algorithm WeakLearn} \\
& \;\; \text{integer T specifying number of iterations} \\
\textbf{Initialize:} & \;\; \text{the weight vector:} D_1(i) = \frac{1}{N} \;\; \text{for} \;\; i = 1, \cdots, N \\
\textbf{Do for} & t = 1, \cdots, T: \\
& \;\; 1. \; \text{Set} \;\;\;\;\;\;\;\;\; \mathbf{p}^t = \frac{\mathbf{D}_t}{\sum_{i=1}^N D_t(i)} \\
& \;\; 2. \; \text{Call WeakLearn, provideing it with the distribution} \mathbf{p}^t; \text{get back a}  \\
& \;\; \text{hypothesis} \;\; h_t: X \rightarrow [0,1] \\
& \;\; 3. \; \text{Calculate the error of} \;\; h_t:\epsilon_t = \sum_{i=1}^N p_i^t|h_t(x_i) - y_i| \\
& \;\; 4. \; \text{Set} \;\; \alpha_t = \frac{1-\epsilon_t}{\epsilon_t} \\
& \;\; 5. \; \text{Set the new weights vector to be} D_{t+1}(i) = D_t(i)\alpha_t^{|h_t(x)-y_i|-1} \\
\textbf{Output:} & \text{the hypothesis}
\end{aligned}
$$
\[ h_f(x) = \left\{
  \begin{array}{l l}
    1 & \quad \text{if $\sum_{i=1}^T\log\alpha_th_t(x) \ge \frac{1}{2}\sum_{t=1}^T\log\alpha_t$} \\
    0 & \quad \text{otherwise}
  \end{array} \right.\]

Secondly if we assume all $h_t$ are binary function mapping $x_i$ to ${-1, +1}$. This assumption is valid if
we are considering decision stump as the weak hypotheses. $h_t$ now is nothing but the fitted binary class
$-1$ or $+1$.
$$
\begin{aligned}
\textbf{Input:} & \;\; \text{sequence of N labeled examples} \;\; (x_1, y_1), \cdots, (x_N, y_N) \\
& \;\; \text{distribution D over the N examples} \\
& \;\; \text{week learning algorithm WeakLearn} \\
& \;\; \text{integer T specifying number of iterations} \\
\textbf{Initialize:} & \;\; \text{the weight vector:} D_1(i) = \frac{1}{N} \;\; \text{for} \;\; i = 1, \cdots, N \\
\textbf{Do for} & t = 1, \cdots, T: \\
& \;\; 1. \; \text{Set} \;\;\;\;\;\;\;\;\; \mathbf{p}^t = \frac{\mathbf{D}_t}{\sum_{i=1}^N D_t(i)} \\
& \;\; 2. \; \text{Call WeakLearn, provideing it with the distribution} \mathbf{p}^t; \text{get back a}  \\
& \;\; \text{hypothesis} \;\; h_t: X \rightarrow \{-1, +1\} \\
& \;\; 3. \; \text{Calculate the error of} \;\; h_t:\epsilon_t = \sum_{i:h_t(x_i) \ne y_i} p_i^t \\
& \;\; 4. \; \text{Set} \;\; \alpha_t = \frac{1-\epsilon_t}{\epsilon_t} \\
& \;\; 5. \; \text{Set the new weights vector to be} 
\end{aligned}
$$
\[ D_{t+1}(i)  = \left\{
  \begin{array}{l l}
    D_t(i)\alpha_t^{-\frac{1}{2}} & \quad \text{if $h_t(x_i) = y_i$} \\
    D_t(i)\alpha_t^{\frac{1}{2}} & \quad \text{if $h_t(x_i) \ne y_i$} 
  \end{array} \right.\]
$$
\begin{aligned}
&\textbf{Output:} \;\; \text{the hypothesis} \;\;\; h_f(x) = \text{sign}\left( \sum_{i=1}^T \log\alpha_t h_t(x)\right)
\end{aligned}
$$

#### Note ####

- $\epsilon_t$ is the expectation of error with respect to the distribution of $D_t$
- the intuition of $\alpha_t$ is trying to make $h_t$ more different than $h_{t+1}$. We want the new weight $D_{t+1}$ 
to make error of $h_t$ to be random-like.
### Example bagging ###

### Random features ###
