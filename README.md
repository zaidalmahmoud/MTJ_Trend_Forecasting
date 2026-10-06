# MTJ_Trend_Forecasting

This is a Python implementation of the framework proposed in the paper: "[Artificial Intelligence strategic planning on Magnetic Tunnel Junction]()".

## Abstract

The rapid development of magnetic tunnel junction (MTJ) technologies is opening new opportunities for neuromorphic computing, where magnetic materials, spin-dependent transport, device physics, and CMOS integration converge to enable energy-efficient, adaptive information processing. Identifying the material platforms, physical mechanisms, and device technologies likely to drive advances is challenging because these domains evolve and interact concurrently. Here, we establish a data-driven framework to map and forecast the evolution of the technologies underlying MTJ-based neuromorphic computing, with particular relevance to adaptive and online brain–computer interface (BCI) decoding. We construct a hierarchical taxonomy to represent correlations between materials, physical phenomena, devices, integration technologies, and application-level concepts, and apply it to a corpus of 9,114 publications. We extract monthly term-mention frequencies to quantify the temporal evolution of individual research nodes. We use a Bayesian multivariate time-series graph neural network (B-MTGNN) to forecast research-domain evolution over a 36-month horizon. The forecast reveals distinct trajectories among coupled materials and technology domains, allowing the proposed MTJ Neuromorphic Hype Cycle (MNHC) to identify and interpret relative gaps in their projected development. This framework provides a quantitative perspective and strategy for emerging research domains in MTJ-based neuromorphic computing.

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
