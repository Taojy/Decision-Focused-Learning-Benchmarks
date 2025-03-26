# Decision-Focused-Learning-Benchmarks
This repository contains the code for the paper Decision-Focused Learning for Power System Decision-Making under Uncertainty. We developed the benchmark model based on load forecasting for two-stage electricity generation scheduling in the IEEE 118-Bus system, a widely recognized test case.

Here, we compare the performance of six typical approaches from widely recognized papers: 

a. Deterministic Approach: Point forecast（MLR）+ deterministic optimization [1,2].

b. Probabilistic Approach: Probabilistic forecast + uncertain optimization [3].

c. DFL Approach: As suggested by other reviewers, the DFL methods are further categorized into four types based on direct/indirect, gradient based/gradient free. We select one baseline paper for each method and replicated on our benchmark test case: Indirect Gradient-Based Methods [4], Indirect Gradient-Free Methods [5], Direct Gradient-Based Methods [6], and Direct Gradient-Free Methods [7].

Two pre-experimental remarks are made for this comparison analysis: 

Remark I: In consideration of the significant impact of forecasting model on decision performance, two sub-models are further evaluated for SBL method, probabilistic method, and indirect DOL methods as shown in Table III, namely a naïve regression model and a time-specific ridge regression model. The naïve model is a unified forecasting model trained for all time intervals as in [1] and the time-specific approach trains 24 distinct regression models (one per hour), effectively creating 24 surrogate functions as in [4]. L2 regularization is added to prevent overfitting [8].

Remark II: It is noted that the quantitative results might vary on different testing data sets, model selections and scenario parameters such as generation cost and renewable penetrations. The function of this open-source framework is to serve as an experiment platform and baseline case. Future works are encouraged to adjust the models and parameters in this comparison to test different methods. 

