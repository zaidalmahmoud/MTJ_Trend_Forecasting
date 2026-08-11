# MTJ_Trend_Forecasting

This is a Python implementation of the framework proposed in the paper: "[Artificial Intelligence strategic planning on Magnetic Tunnel Junction]()".

## Abstract

Advances in magnetic tunnel junction (MTJ) technology have expanded interest in MTJ-based neuromorphic computing for adaptive and online brain-computer interface (BCI) decoding. MTJ-based neuromorphic computing integrates multiple technological domains, including spintronic device physics, complementary metal-oxide-semiconductor (CMOS)  integration, and low-power continual learning. These domains continuously interact with each other. Analyzing the development trend of an individual technology requires a simultaneous quantitative evaluation that compares technological progress across them. However, quantitative trend analysis and forecasting across interconnected domains remain limited. Here, this study systematically analyzes research trends in MTJ-based neuromorphic computing for adaptive, online BCI decoding and forecasts their evolution over the next 36 months. The dataset constructs a hierarchical, node-based taxonomy representing the domains, collects 9114 publications for these domains, and counts the monthly mentioned term frequency. The Bayesian multivariate time-series graph neural network (B-MTGNN) predicts the future mention frequency of nodes and shows that the difference in gaps varies depending on the node. The MTJ Neuromorphic Hype Cycle (MNHC) provides the quantitative technical interpretation and guidance using the forecasting results. These findings contribute to systematically reviewing the trends in MTJ-based neuromorphic computing and its underlying technologies and to identifying promising directions for future research.

## Dataset
The data can be found in the directory [**data**](https://github.com/zaidalmahmoud/MTJ_Trend_Forecasting/tree/main/data).

## Key Files

- **`train_test.py`**: Trains and evaluates the model to identify the optimal hyperparameter configuration.
- **`train.py`**: Trains the final model on the complete dataset using the optimal hyperparameters and saves the resulting operational model for forecasting.
- **`forecast.py`**: Uses the operational model to forecast future trends and generate the corresponding predictions and future gap estimates.
- **`net.py`**: Contains the implementation of the Bayesian Multivariate Time-Series Graph Neural Network (B-MTGNN) architecture.

## Citation

```bibtex
@article{author2026title,
  title   = {},
  journal = {},
  volume  = {},
  pages   = {},
  year    = {},
  issn    = {},
  doi     = {},
  url     = {},
  author  = {}
}
