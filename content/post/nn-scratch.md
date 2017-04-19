+++
date = "2017-04-12T12:00:00"
draft = false
tags = ["machine learning", "deep learning"]
title = "Neural networks from scratch"
math = true
summary = """
Understanding a bit of the math from neural nets and its implementation on Numpy.
"""

+++

## Introduction

I have used a handful of machine learning models in the past. These include simple linear regression models, support vector machines (SVMs) and neural networks. While working on different projects I was mostly concerned in solving a specific application problem and did not worry about the inner workings of the models I was using. I have delved into more details of SVM during my final undergraduate project but did not have the time to do the same for neural networks.

After reading and learning about it in more detail it has come the time for me to share it. In this post I will give some brief introduction to neural nets and derive some of the maths behind a simple architecture as well as its implementation using only [Numpy](http://www.numpy.org/). I have based most of this post on [Stanford's CS231](http://cs231n.github.io/neural-networks-case-study/) notes which I found very useful for beginners. Some of the math derived here was not available there so it might be useful for a more interested reader.

Neural networks are organized in layers, each consisting of many units. These units are responsible for crunching the data. Each unit can receive many inputs but has a single output that can be connected to many more units in the following layer. In this study we will address a fully-connected structucture, meaning that every unit in a single layer receives input from all the units from the previous layer. To compute its output, each unit perform a weighted sum of its inputs and a bias term, then apply an activation funcion, in general non-linear, to the calculated sum. The model parameters consists of the connection weights between the units and their bias terms.

## Defining the model

To simplify our study we are going to use a two layer network architecture (the input layer is not counted). The input layer $x$ has $N_x$ units, the hidden layer $h$ has $N_h$ and the output layer $y$ has $N_y$ units.

We could specify this architecture to perform multi-class classification, so the output layer would give the probability the sample belongs to each of the available classes. In this case we use $N_y$ as the number of classes and to enforce a probabilistic output we use an activation function called **softmax** on the output layer, which also guarantess that the sum of the classes probabilities sums to 1.

As for the hidden layer we can use a **RELU** activation function, because it is avery simple to evaluate non-linear function: $\text{RELU}(z) = \max(0, z)$. There are other reasons why we would use it, mostly to avoid a problem called vanishing gradient, but in a shallow network such as this one this would not be a problem.

We define the weights of the connections between the input and hidden layer units as the matrix $W1 \in \mathbb{R}^{N\_x \times N\_h}$, where $W1\_{i,j}$ is the weight of the connection between the $i$-th input layer unit to the $j$-th hidden layer unit. The bias vector of the hidden layer is defined as $b1 \in \mathbb{R}^{N_h}$. Similarly we have the parameters of connection between the hidden layer to the output layer: $W2 \in \mathbb{R}^{N\_h \times N\_y}$ and $b2 \in \mathbb{R}^{N\_y}.$

![nn-diagram](/notebooks/nn-diagram.png)

## Forward propagation

Assuming we have all the ideal model parameters $W1, W2, b1, b2$, how do we get the output of the network for a given input sample? This is called forward propagation, since the data flows from the input layer, through the hidden layers and finally to the output layer as presented in the following equations.

<div>$$\begin{align*}
h &= \max(0,x W1 + b1) \\
y &= \text{softmax}(\underbrace{h W2 + b2}_\textrm{score})
\end{align*}$$</div>


The **softmax** activation function can be expressed as
$$ \text{softmax}(z)_j = \frac{e^{z_j}}{\sum\_{i=0}^{N_z}e^{z_i}} $$

This formulation allows the input $x$ to have many samples, each in a row, so $x \in \mathbb{R}^{n \times N\_\text{features}}$, where $n$ is the number of samples in a batch, also called batch size, and $N\_\text{features}$ is the number of features, or the dimension of each sample. This would yield an output $y \in \mathbb{R}^{n \times N\_y}$, where each sample probability distribution among classes is given in a row.

## Training process

In order to get the ideal model parameters we have to train the network on a training set. This consists of an optimization process where we try to minimize a loss function that tells how close the network output $y$ is to the real labels $\hat{y}$. At each iteration of this process we obtain new values for the parameters that will hopefully decrease the value of the loss function.

In this example, since we assumed a multi-class classification problem with a probabilistic output, the ideal loss function to use is the categorical cross-entropy function, given by
$$ L\_i = -\log y\_{\hat{y}\_i}$$

To make sense of this function we can analyse its behaviour. For a given sample $x_i$ belonging to class $\hat{y}_i$ it will compute the negative log of the output probability of the sample belonging to classs $\hat{y}_i$ (given by the $\hat{y}_i$-th component of $y$). Ideally this probability would be 1, which would make $L_i=0$. Whenever this is not the case, there is a loss associated with the sample $x_i$.

Now we have a measure to evaluate how good our classifier is doing on the training set we can try to optimize the network parameters to get the loss as close to zero as possible. We do this using an iterative algorithm called **Stochastic Gradient Descent**. To give a brief overview of this method, assume all network parameters are represented in a vector $\theta$. We can compute the variation of the loss function $\Delta L$ given a variation of the parameters vector $\Delta \theta$ as $$\Delta L = \Delta \theta \cdot \nabla L$$ where $\nabla L$ is the gradient of the loss with respect to the parameters $\theta$. We always want to descrease the loss, so we want $\Delta L < 0$. One way to guarantee this condition is to choose the variation of parameters as $$\Delta \theta = - \eta \nabla L$$ for a small-enough learning rate $\eta >0$, which would yield $$\Delta L = -\eta |\nabla L|^2.$$

This description is for vanilla Gradient Descent, also called batch-GD (where $L = \sum L_i$), so every sample is considered in the gradient. The stochastic part comes when you only consider a single sample at each iteration step, what reduces training time, especially on big datasets. Although it may seem attractive, this method offers slower convergence since there is a lot of zig-zagging between samples optimizations. To overcome this another variation called mini-batch GD can be used. This method lies in between batch-GD (uses one sample) and SGD (uses all samples), since it considers a mini-batch of size $N_b$ to compute the gradient, thus reducing training time and still allowing faster convergence.

By iteratively running this optimization algorithm we can reduce the loss function and train our model. There is one missing step though: how to compute the loss function gradient $\nabla L.$

## Backpropagation and gradient computation

Even considering our small network the loss is a rather complex function of the network parameters given all multiplications and non-linear activations. To compute its gradient $\nabla L$ we must find the derivative of $L$ with respect to all model parameters, which can be difficult to be done analytically. We then use an algorithm called Backpropagation, which is basically the use of calculus' chain-rule. By multiplying the local derivatives from layer to layer we can numerically evaluate the derivative of the loss with respect to any parameter.

We start from the output layer $y$. Since $$ L\_i = -\log y\_{\hat{y}\_i}$$ we have $$\frac{\partial L_i}{\partial y_k} = \frac{-1}{y_k} 1(\hat{y}_i=k)$$ where 1(z) is the indicator function (1 if argument true, otherwise 0).

$\DeclareMathOperator{\score}{score}$
We then calculate the derivatives of the output $y_k$ with respect the intermediate variable $\score = h W2 + b2$, with $ y = \text{softmax}(h W2 + b2)$. We must consider two cases, one for the derivative of $y_k$ with respect to $\score\_j$ with $k \neq j$ and another with $k=j$.

For the first case we can write $$ y\_k = \frac{e^{\score\_k}}{e^{\score\_j} + \sum_{i \neq j} e^{\score\_i}}$$ then using the quotient rule for derivatives we have:
$$\frac{\partial y_k}{\partial \text{score}\_j} = \frac{-e^{\score_k}e^{\score_j}}{(\sum_i e^{\score\_i})^2} = -\frac{e^{\score_k}}{\sum_i e^{\score\_i}} \frac{e^{\score_j}}{\sum_i e^{\score\_i}} = -y_k y_j$$

For the second case, when $k=j$ we can write $$ y\_k = \frac{e^{\score\_k}}{e^{\score\_k} + \sum_{i \neq k} e^{\score\_i}}$$ and then the derivative becomes:

<div>\begin{eqnarray}
\frac{\partial y_k}{\partial \text{score}_k} &=& \frac{e^{\score_k}(\sum_i e^{\score_i}) - e^{2\score_k}}{(\sum_i e^{\score_i})^2} \\
 &=& \frac{e^{\score_k}}{\sum_i e^{\score_i}} - \frac{e^{\score_k}}{\sum_i e^{\score_i}} \frac{e^{\score_k}}{\sum_i e^{\score_i}} \\
 &=&  y_k-y_k y_k \\
 &=& y_k(1-y_k)
\end{eqnarray}</div>

Now, to calculate the derivative of the loss with respect to the score intermediate variables we use the chain-rule as follows:

<div>\begin{eqnarray}
  \frac{\partial L_i}{\partial \score_j} & = & \frac{\partial L_i}{\partial y_k} \frac{\partial y_k}{\partial \score_j} &\\
   & = & \frac{-1}{y_{\hat{y}_i}} \times -y_{\hat{y}_i} y_j = y_j, &\text{ if } j \neq \hat{y}_i \\
   & = & \frac{-1}{y_{\hat{y}_i}} \times y_{\hat{y}_i}(1-y_{\hat{y}_i}) = y_{\hat{y}_i} -1, &\text{ if } j=\hat{y}_i
\end{eqnarray}</div>

For implementation reasons we can call a vector $\text{dscore} \in \mathbb{R}^{N_y}$ where each component is the derivative of the loss regarding a component of the score variable. In a compact form: $$ \text{dscore}_j = \frac{\partial L_i}{\partial \score_j} = y_j - 1(j=\hat{y}_i)$$



We must now propagate this derivative to the parameters of the output layer. From definition we have $\score = h W2 + b2$. To improve visualization we can expand it in the form (considering a single sample batch, n=1):

<div>$$\begin{align*}
    \score_1 &= h_1 W2_{11} + h_2 W2_{21} + h_3 W2_{31} + \cdots + b2_1 \\
    \score_2 &= h_1 W2_{12} + h_2 W2_{22} + h_3 W2_{32} + \cdots + b2_2 \\
    \score_3 &= h_1 W2_{13} + h_2 W2_{23} + h_3 W2_{33} + \cdots + b2_3 \\
    & \vdots & \\
    \score_j &= h_1 W2_{1j} + h_2 W2_{2j} + h_3 W2_{3j} + \cdots + b2_j \\
\end{align*}$$</div>

It is easy to see that $$\frac{\partial \score_j}{\partial b2_j} = 1 \implies \frac{\partial L_i}{\partial b2_j} = \frac{\partial L_i}{\partial \score_j} \frac{\partial \score_j}{\partial b2_j} = \frac{\partial L_i}{\partial \score_j}.$$ This imples that the vector of weights update for $b2$ is $db2 = \text{dscore}$

Similarly, we have $$ \frac{\partial \score\_j}{\partial W2\_{kj}} = h\_k \implies \frac{\partial L_i}{\partial W2\_{kj}} =  \frac{\partial L_i}{\partial \score\_j} \frac{\partial \score\_j}{\partial W2\_{kj}} = h\_k \frac{\partial L_i}{\partial \score\_j}$$

In this case, the matrix of weights updates is given by

<div>$$\begin{align*}
dW2 &= \begin{pmatrix}
\frac{\partial L}{\partial W_{11}} & \frac{\partial L}{\partial W_{12}}  & \cdots \\
\frac{\partial L}{\partial W_{21}} & \frac{\partial L}{\partial W_{22}}  & \cdots \\
\vdots & \vdots & \ddots
\end{pmatrix} \\
  &= \begin{pmatrix}
    h_1 \text{dscore}_1 & h_1 \text{dscore}_2  & \cdots \\
    h_2 \text{dscore}_1 & h_2 \text{dscore}_2  & \cdots \\
    \vdots & \vdots & \ddots
    \end{pmatrix} \\
  & = \begin{pmatrix}
      h_1 \\
      h_2  \\
      \vdots
      \end{pmatrix}
      \begin{pmatrix}
       \text{dscore}_1 &  \text{dscore}_2  & \cdots \\
      \end{pmatrix}
  & = h^T \text{dscore}

\end{align*}$$</div>


To propagate the gradient to the hidden layer we must first calculate the gradient with respect to $h$: $$\frac{\partial \score\_j}{\partial h\_k} = W2\_{kj}$$

The differential parameter vector $dh$ is given by:

<div>$$\begin{align*}
dh^T &= \begin{pmatrix}
\frac{\partial L}{\partial h_1} \\
\frac{\partial L}{\partial h_2}  \\
\vdots
\end{pmatrix} \\
&=
\begin{pmatrix}
\frac{\partial L}{\partial\score_1}\frac{\partial \score_1}{\partial h_1}+ \frac{\partial L}{\partial\score_2}\frac{\partial \score_2}{\partial h_1} + \cdots\\
\frac{\partial L}{\partial\score_1}\frac{\partial \score_1}{\partial h_2}+ \frac{\partial L}{\partial\score_2}\frac{\partial \score_2}{\partial h_2} + \cdots\\
\vdots
\end{pmatrix}\\
&=
\begin{pmatrix}
\frac{\partial \score_1}{\partial h_1} & \frac{\partial \score_2}{\partial h_1} & \cdots\\
\frac{\partial \score_1}{\partial h_2} & \frac{\partial \score_2}{\partial h_2} & \cdots\\
\vdots & \vdots & \ddots
\end{pmatrix}
\begin{pmatrix}
\frac{\partial L}{\partial\score_1}\\
\frac{\partial L}{\partial\score_2}\\
\vdots
\end{pmatrix} \\
&= W2 \times \text{dscore}^T\\

dh &= \text{dscore} \times W2^T
\end{align*}$$</div>

Next we consider the RELU activation: $h=max(0, \underbrace{xW1+b1}\_\textrm{r})$. $\frac{dh}{dr} = 1(r>0)$. Thus, $dr = dh \times 1(h>0)$.

Finally, for the hidden layer, we can observe $dh$ as the output and $x$ as input, so by extending the equations for $dW2$ and $db2$ we have

<div>$$\begin{align*}
  dW1 &= x^T \text{dr} \\
  db1 &= \text{dr}
\end{align*}$$</div>

## Numpy implementation
The forward propagation is straightforward as can be seen below.

```python
h = np.maximum(0, np.dot(x,W1)+b1)
score = np.dot(h,W2)+b2
y = np.exp(score)
y /= np.sum(y, axis=1, keepdims=True)
```

Although understanding backpropagation can be difficult, the resulting equations are somewhat simple to implement, because they only need to consider the local derivatives at each step.

Primarily we computer the $\text{dscore}$ intermediate variable, where y_t represents $\hat{y}$, the true class label.
```python
dscore = np.copy(y)
dscore[range(n), y_t] -= 1
dscore /= n
```

Next the output layer parameters updates are calculated. Since we now have $n$ training samples, we have $L = \frac{1}{n} \sum\_{i=0}^n L_i$, so dscore gets summed (the division by $n$ has already taken place in the previous step).
```python
dW2 = np.dot(h.T, dscore)
db2 = np.sum(dscore, axis=0, keepdims=True)
```

The hidden layer activation derivative is calculated, followed by the parameter updates.
```python
dh = np.dot(dscore, W2.T)
dr = dh
dr[h <= 0] = 0

dW1 = np.dot(x.T, dr)
db1 = np.sum(dr, axis=0, keepdims=True)
```

Finally we apply the weight updates using the a specified learning rate $\eta$ (lr).
```python
W1 -= lr*dW1
b1 -= lr*db1
W2 -= lr*dW2
b2 -= lr*db2
```

To check the code and results, please visit [this notebook](/notebooks/nn-scratch.html).
