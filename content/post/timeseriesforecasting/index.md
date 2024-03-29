---
title: "Forecasting sales with Gaussian Processes and Autoregressive models"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
math : true
date: 2020-07-23T15:57:57+01:00
lastmod: 2020-07-23T15:57:57+01:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

---
# Problem definition

Sales forecasting is a very common problem faced by many companies. Given a history of sales of a certain product we would like to predict the demand of that product for a time window in the future. This is useful as it allows companies and industries to plan their workload and reduce waste of resources. This problem is a common example of time series forecasting and there are many approaches to tackle it.

Today we will learn a bit more about this with a practical example: the [Kaggle Predict Future Sales dataset](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/). This challenge aims to predict future sales of different products for the next month in different shops of a retail chain. In this notebook we do not aim at solving this challenge, rather, we will explore some concepts of time series forecasting that could be used to solve such problem.

You can download the Jupyter notebook version of this tutorial [here](https://github.com/eduardohenriquearnold/eduardohenriquearnold.github.io/blob/master/notebooks/timeseriesforecast.ipynb).

# Exploring the dataset

First we need to download the whole dataset and extract it to a `data` folder in the root of this notebook path.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact_manual
plt.style.use('ggplot')
%matplotlib inline
```


```python
#load sales and rename columns to ease understanding
dtype = {'date_block_number':np.uint16, 'shop_id':np.uint32, 'item_id':np.uint32, 'item_cnt_day':np.float32}
sales = pd.read_csv('data/sales_train.csv', dtype=dtype) \
          .rename(columns={'date_block_num': 'month', 'item_cnt_day':'sold'})
```


```python
sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>month</th>
      <th>shop_id</th>
      <th>item_id</th>
      <th>item_price</th>
      <th>sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>02.01.2013</td>
      <td>0</td>
      <td>59</td>
      <td>22154</td>
      <td>999.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>05.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2552</td>
      <td>899.00</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2554</td>
      <td>1709.05</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.01.2013</td>
      <td>0</td>
      <td>25</td>
      <td>2555</td>
      <td>1099.00</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, we have daily records for each shop and item. Note that some of the `sold` values are negative because they include returns.

Firstly, since we are interested in estimating the monthly sales, we will aggregate the dataset by month using the handy `month` feature, which ranges from 0 (representing January 2013) to 33 (October 2015). We will group the sales by the shop and item ids. The aggregation will sum all `sold` fields during the month for each product and shop, and average the item_price (it is likely to change from month to month).


```python
gsales = sales.groupby(['shop_id','item_id','month']).agg({'item_price':np.mean, 'sold':np.sum})
```


```python
gsales[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>item_price</th>
      <th>sold</th>
    </tr>
    <tr>
      <th>shop_id</th>
      <th>item_id</th>
      <th>month</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="20" valign="top">0</th>
      <th>30</th>
      <th>1</th>
      <td>265.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>31</th>
      <th>1</th>
      <td>434.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">32</th>
      <th>0</th>
      <td>221.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>221.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">33</th>
      <th>0</th>
      <td>347.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>347.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">35</th>
      <th>0</th>
      <td>247.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>247.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>36</th>
      <th>1</th>
      <td>357.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>40</th>
      <th>1</th>
      <td>127.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>42</th>
      <th>1</th>
      <td>127.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>43</th>
      <th>0</th>
      <td>221.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>49</th>
      <th>1</th>
      <td>127.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">51</th>
      <th>0</th>
      <td>128.5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>127.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>57</th>
      <th>1</th>
      <td>167.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>59</th>
      <th>1</th>
      <td>110.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>61</th>
      <th>0</th>
      <td>195.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75</th>
      <th>0</th>
      <td>76.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>85</th>
      <th>1</th>
      <td>190.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



We can now observe that for the first shop many items only have selling records for the first two months. We would like to investigate how this varies for different products across all 60 shops.


```python
@interact_manual(shop_id = (0,59,1))
def plot_product_record_frequency(shop_id):
    count_months = gsales.reset_index().groupby(['shop_id','item_id']).size()[shop_id]
    plt.bar(count_months.keys(), count_months)
    plt.xlabel('product_id')
    plt.ylabel('Num months available')
```


    interactive(children=(IntSlider(value=29, description='shop_id', max=59), Button(description='Run Interact', s…


However, this interactive visualisation will not work unless you have a notebook running, so we will plot the sales frequency for some of the stores below:


```python
def plot_product_record_frequency(shop_id):
    count_months = gsales.reset_index().groupby(['shop_id','item_id']).size()[shop_id]
    plt.bar(count_months.keys(), count_months)
    plt.xlabel('product_id')
    plt.ylabel('Num months available')
    plt.title(f'shop_id {shop_id}')
    
fig=plt.figure(figsize=(12, 8), dpi= 80, facecolor='w', edgecolor='k')    
for i, shop_id in enumerate([0,1,29,31]):
    plt.subplot(2,2,i+1)
    plot_product_record_frequency(shop_id)
plt.tight_layout()
```


![png](output_15_0.png)


We can observe that some shops have a very limited record of sales, e.g. shop 0 and 1. On the other hand, shop 29 and 31 have a considerable number of items with a record above 10 weeks.

Perhaps we should look into the shops with the larger number of sales, which we can inspect by observing the distribution of `shop_id`:


```python
sales['shop_id'].plot.hist(bins=60)
plt.xlabel('product_id');
```


![png](output_18_0.png)


If we observe the previous histogram for the number of weeks of records for each product of a given shop, we will observe indeed that the shop 31 has a comprehensive number of products with a significant history of sales.

Considering the behaviour of multiple shops introduce complexity steaming from this biased distribution of sales. Some shops will have very little preior information about their sales, so it would be difficult to make a good prediction of sales in these shops. For this reason we will neglect different shops and instead try to estimate a new volume of sales for all products in the whole supermarket chain. So, we modify our aggregated sales DataFrame `gsales` accordingly:


```python
gsales = sales.groupby(['item_id','month']).agg({'item_price':np.mean, 'sold':np.sum})
```

We can also observe what is the number of months available for each product sale history in this chain-wide collection:


```python
count_months = gsales.reset_index().groupby(['item_id']).size()
plt.bar(count_months.keys(),count_months);
plt.xlabel('product_id')
plt.ylabel('Num months available');
```


![png](output_23_0.png)


Although in a real forecast scenario we would like to have a good prediction of sales even when the history for a given product is minimal, in this example we will focus on products with full history for all 34 months.


```python
productsIdx = count_months[count_months == 34].index.values
selected_sales = gsales.loc[productsIdx]
productsIdx.shape
```




    (523,)



We obtained 523 unique products that have a selling history for all of the 33 months. An example of the first 60 items are below:


```python
selected_sales[:60]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>item_price</th>
      <th>sold</th>
    </tr>
    <tr>
      <th>item_id</th>
      <th>month</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="34" valign="top">32</th>
      <th>0</th>
      <td>338.110349</td>
      <td>299.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>337.771930</td>
      <td>208.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>343.794702</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>341.888889</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>347.000000</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>342.070886</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>345.951190</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>340.172727</td>
      <td>72.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>340.017544</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>184.592593</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>144.316456</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>147.994444</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>144.710526</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>144.066667</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>149.000000</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>143.714286</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>149.000000</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>149.000000</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>130.500000</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>148.991176</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>144.771429</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>148.991667</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>146.060714</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>149.000000</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>145.572727</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>146.387333</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>146.869167</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>149.000000</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>149.000000</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>149.000000</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>149.000000</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>148.714286</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>149.000000</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>149.000000</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th rowspan="26" valign="top">33</th>
      <th>0</th>
      <td>488.517241</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>484.170732</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>490.870968</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>489.500000</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>499.000000</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>205.046512</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>195.439130</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>197.277778</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>198.052381</td>
      <td>43.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>195.915152</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>194.866667</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>195.900000</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>197.487805</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>196.862069</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>197.673333</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>199.000000</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>196.658824</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>199.000000</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>199.000000</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>197.756250</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>199.000000</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>199.000000</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>195.460000</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>199.000000</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>199.000000</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>199.000000</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
</div>



We have explored the dataset and observed some items of interest. The next step is using this information to predict the number of sales over another month

# Time Series Analysis

Firstly, let's observe what the sales time series look like for some of the products:


```python
fig=plt.figure(figsize=(12, 8), dpi= 80, facecolor='w', edgecolor='k')
for i, idx in enumerate(selected_sales.index.get_level_values(0).unique()[10:20]):
    plt.subplot(5,2,i+1)
    selected_sales.loc[idx]['sold'].plot(style='r.')
    plt.title(f'Sales for item_id {idx}')
    plt.tight_layout()
```


![png](output_31_0.png)


We can see some seasonality patterns in the plotted data, so we will decompose one of the observed time series into a trend and seasonal effects with a multiplicative model $Y(t) = T(t) S(t) r(t)$ where $T(t)$ is the trend, $S(t)$ the the seasonal component and $r(t)$ is the residual.


```python
from statsmodels.tsa.seasonal import seasonal_decompose
item_id = 491
sales_product = selected_sales.loc[item_id]['sold'].values
dec = seasonal_decompose(sales_product, model='multiplicative', period=12)
dec.plot();
```


![png](output_33_0.png)


We observed that the residual values tend to be close to 1, which means the observed series has a good fit to the seasonal and trend decomposition. 

This analysis shows that the observed time series have a trend that decreases throughout the months, and a seasonal component, which seems to peak around August and December. This decomposition could be exploited to improve prediction results, however we would require a model that incorporate this seasonality pattern. 

# Time series forecasting

In this section we will evaluate two different models for time series forecasting: Auto Regressive Integrated Moving Average (ARIMA) and Gaussian Process (GP).

Firstly, we train the models on each product time series up to month 30. Then, we evaluate for the remaining of months available across all selected products. Our metric will be the Root Mean Squared Error (RMSE) computed with the predicted and ground-truth time series. Please note that this metric is the same as used in the original challenge.

### Gaussian Processes

GPs are non-parametric models that can represent a posterior over functions. They work well when we do not wish to impose strong assumptions on the generative data process. For a more detailed explanation of GPs, please visit this [distill post](https://distill.pub/2019/visual-exploration-gaussian-processes/).

We chose experimented with a multitude of kernels, the best performing were the simpler ones: RBF and Matern. Note that the parameters of these kernels are optimised (within the given bounds) during the `fit` process. Please note that we normalise the target sales by the maximum value such that the target number of sales ranges from [0,1].

First, let's observe what happens using a single item_id and extrapolating between the months:



```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern, ExpSineSquared, ConstantKernel

kernel = RBF()
gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=2, alpha = 1e-5, normalize_y=True)
```


```python
item_id = 53
X = np.arange(0,34,0.05).reshape(-1,1)
Y = selected_sales['sold'][item_id].values.reshape(-1,1)
ymax = Y.max()

gp.fit(np.arange(30).reshape(-1,1), Y[:30]/ymax)
Y_pred, Y_pred_std = gp.predict(X, return_std=True)
Y_pred = Y_pred.reshape(-1)*ymax

fig=plt.figure(figsize=(6, 4), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(X, Y_pred, 'r.', label='Pred')
plt.fill_between(X.reshape(-1), Y_pred - Y_pred_std, Y_pred + Y_pred_std, alpha=0.2, color='k');
plt.plot(np.arange(34), Y, 'bx', label='GT')
plt.axvline(29.5, 0, 1, color='black', ls='--', label='Train/Test split')
plt.legend();
```


![png](output_44_0.png)


Now we will create a GP for each item_id time series within our selected sales:


```python
Y_preds_gp = []
Y_gts = []
for item_id in selected_sales.index.get_level_values(0).unique():
    X = np.arange(34).reshape(-1,1)
    X_train, X_test = X[:30], X[30:]
    
    Y = selected_sales['sold'][item_id].values.reshape(-1,1)
    Y_train, Y_test = Y[:30], Y[30:]
    ymax = Y_train.max()
    
    kernel = RBF()
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=0, alpha = 1e-2, normalize_y=True)
    gp.fit(X_train, Y_train/ymax)
    ypred = gp.predict(X_test)*ymax
    
    Y_preds_gp.append(ypred)
    Y_gts.append(Y_test)
    
Y_preds_gp = np.concatenate(Y_preds_gp, axis=0)
Y_gts = np.concatenate(Y_gts, axis=0)
```

### ARIMA

Auto Regressive Integrated Moving Average models the time series using
$$Y(t) = \alpha + \beta_1 Y(t-1) + \beta_2 Y(t-2) + \dots + \beta_p Y(t-p) + \gamma_1 \epsilon(t-1) + \gamma_2 \epsilon(t-2) + \dots + \gamma_q \epsilon(t-q) + \epsilon(t)$$
where $\epsilon(t)$ is the residual from the ground-truth and estimated value. The parameters $\alpha, \beta, \gamma$ are optimised during fitting. The hyper-parameters $p,d,q$ correspond to the order of the process, **i.e.** how many terms of previous timestamps, how many previous error terms and how many times to differentiate the time series until it becomes stationary.

For simplicity, we assume our time-series are stationary and use $p=1$, $q=0$, $d=0$

Again, let's start by visualising a single time-series and the resulting ARIMA prediction.


```python
from statsmodels.tsa.arima_model import ARIMA

item_id = 53
Y = selected_sales['sold'][item_id].values.reshape(-1,1)

model = ARIMA(Y[:30], order=(1,0,0)).fit(trend='nc')
Y_pred = model.predict(0,33)

fig=plt.figure(figsize=(6, 4), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(X, Y_pred, 'r.', label='Pred')
plt.plot(np.arange(34), Y, 'bx', label='GT')
plt.axvline(29.5, 0, 1, color='black', ls='--', label='Train/Test split')
plt.legend();
```


![png](output_51_0.png)


We may also observe the approximate distribution of the residuals:


```python
residuals = pd.DataFrame(model.resid)
residuals.plot.kde();
```


![png](output_53_0.png)


Next, we forecast for all items in the selected sales for the remaining 4 months (30,31,32,33):


```python
Y_preds_arima = []
for item_id in selected_sales.index.get_level_values(0).unique():  
    Y = selected_sales['sold'][item_id].values.reshape(-1,1)
    Y_train, Y_test = Y[:30], Y[30:]
    
    ypred = gp.predict(X_test)
    model = ARIMA(Y[:30], order=(1,0,0)).fit(trend='nc')
    ypred = model.predict(30,33)
    
    Y_preds_arima.append(ypred)
    
Y_preds_arima = np.concatenate(Y_preds_arima, axis=0)
```

### Evaluation

Finally computing the RMSE metric between predictions (of GP and ARIMA) and ground-truths for all selected items for the remaining 4 months (30,31,32,33):


```python
from sklearn.metrics import mean_squared_error as mse
rmse_gp = mse(Y_gts, Y_preds_gp, squared=False)
rmse_arima = mse(Y_gts, Y_preds_arima, squared=False)
print(f"RMSE GP {rmse_gp} \nRMSE ARIMA {rmse_arima}")
```

    RMSE GP 58.58414105624866 
    RMSE ARIMA 50.27118697649368


Knowingly, forecasting 4 months into the future would be difficult. Instead we could consider only the 30th month forecast of all items:


```python
rmse_gp = mse(Y_gts[::4], Y_preds_gp[::4], squared=False)
rmse_arima = mse(Y_gts[::4], Y_preds_arima[::4], squared=False)
print(f"RMSE GP {rmse_gp} \nRMSE ARIMA {rmse_arima}")
```

    RMSE GP 28.212626816686967 
    RMSE ARIMA 7.92125195871772


# Conclusion

We observed that the ARIMA model performed better under the RMSE metric for the presented dataset. Still, both RMSE are quite high and should be reduced if the model were to be used in practice.

To further improve the results and to account to a more realistic setting where some items or shops will have a limited history of sales, we must work on feature engineering to overcome these limitations. Example of features that could be exploited include, but are not limited to:
 
1. Shop-specific aggregated stats (mean num of sales)
2. City-specific aggregated stats (mean num of sales, pop. size, etc).
3. Item categories aggregated stats (mean num of sales for a specific product category)
4. Item price and price variations

# Further reading

If you are interested in time series forecasting I highly recommend the above blog posts:

1. https://www.linkedin.com/pulse/how-use-machine-learning-time-series-forecasting-vegard-flovik-phd
2. https://towardsdatascience.com/an-overview-of-time-series-forecasting-models-a2fa7a358fcb
