# MTJ_Trend_Forecasting

This is a Python implementation of the framework proposed in the paper: "[Artificial Intelligence strategic planning on Magnetic Tunnel Junction[]()".

## Abstract

Advances in magnetic tunnel junction (MTJ) technology have expanded interest in MTJ-based neuromorphic computing for adaptive and online brain-computer interface (BCI) decoding. MTJ-based neuromorphic computing integrates multiple technological domains, including spintronic device physics, complementary metal-oxide-semiconductor (CMOS)  integration, and low-power continual learning. Because progress in these domains is strongly interdependent, advances or bottlenecks in one domain may reshape research priorities and expose development gaps in others. An integrated assessment is therefore needed to characterize their relative development trajectories and identify potential imbalances across domains. However, quantitative analyses that jointly chracterize the relative research trajectories and prospective development of these interconnected domains remains limited. In this study, we systematically analyzed the research trends in MTJ-based neuromorphic computing for adaptive, online BCI decoding and forecast their evolution over the subsequent 36 months. We developed a hierarchical node-based taxonomy in which Blue nodes represented broad technological domains and Green nodes represented the specific technological elements supporting or implementing those domains. Using application programming interfaces (APIs), we collected metadata for XX publications published between MONTH/YEAR and MONTH/YEAR and constructed monthly time series based on the Number of Mentions (NoM), defined as the monthly frequency of node-associated terms in publication titles, abstracts, and author keywords. We developed a Bayesian Multivariate Time-Series Graph Neural Network (B-MTGNN) to forecast monthly node-level NoM over the 2026–2028 period by jointly modeling the historical time series and the dependencies encoded in the Blue–Green graph. We then calculated the gap from the difference between the normalized forecasted NoM values of each connected Blue and Green node pair. The spintronic synapse and neuron (Blue node 1) domain showed the largest average gap, whereas the co-design of learning rules with device physics (Blue node 2) domain showed the smallest average gap and limited gap changes. Hybrid CMOS–spintronic integration exhibited heterogeneous gap trajectories across its constituent technologies, whereas low-power continual learning showed the most consistent widening of the gap. We integrated these forecasts with qualitative assessments of each technology’s defining characteristics and research maturity using the newly proposed Hype Cycle framework. This analysis placed B1 and most B4 Green nodes in the Innovation Trigger, whereas B2 and B3 Green nodes mainly occupied the Peak of Inflated Expectations. These findings provide a systematic account of the evolving balance between major MTJ-based neuromorphic computing domains and their enabling technologies, identifying areas in which projected research attention may outpace or lag behind domain-level development.

## Dataset
The data can be found in the directory [**data**](https://github.com/zaidalmahmoud/MTJ_Trend_Forecasting/tree/main/data).

## Key files:
**train_test.py**: Trains and evaluates the model to identify the optimal hyperparameter configuration.
**train.py**: Trains the final model on the complete dataset using the optimal hyperparameters and saves the resulting operational model for forecasting.
**forecast.py**: Uses the operational model to forecast future trends and generate the corresponding prediction data and future gaps.
**net.py**: Contains the implementation of the Bayesian Multivariate Time-Series Graph Neural Network (B-MTGNN) architecture.


## Citation
```

}
```
