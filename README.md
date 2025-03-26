# Decision-Focused-Learning-Benchmarks
This repository contains the code for the paper Decision-Focused Learning for Power System Decision-Making under Uncertainty. We developed the benchmark model based on load forecasting for two-stage electricity generation scheduling in the IEEE 118-Bus system, a widely recognized test case. This work corresponds to the paper [Decision-Focused Learning for Power System Decision-Making under Uncertainty](https://arxiv.org/abs/2401.03680), where we elaborate on our methodology, implementation details, and experimental results.

Here, we compare the performance of six typical approaches from widely recognized papers: 

**a. Deterministic Approach: Point forecast（MLR）+ deterministic optimization [1,2].**

**b. Probabilistic Approach: Probabilistic forecast + uncertain optimization [3].**

**c. DFL Approach: As suggested by other reviewers, the DFL methods are further categorized into four types based on direct/indirect, gradient based/gradient free. We select one baseline paper for each method and replicated on our benchmark test case: Indirect Gradient-Based Methods [4], Indirect Gradient-Free Methods [5], Direct Gradient-Based Methods [6], and Direct Gradient-Free Methods [7].**

Two pre-experimental remarks are made for this comparison analysis: 

**Remark I: In consideration of the significant impact of forecasting model on decision performance, two sub-models are further evaluated for SBL method, probabilistic method, and indirect DOL methods as shown in Table III, namely a naïve regression model and a time-specific ridge regression model. The naïve model is a unified forecasting model trained for all time intervals as in [1] and the time-specific approach trains 24 distinct regression models (one per hour), effectively creating 24 surrogate functions as in [4]. L2 regularization is added to prevent overfitting [8].**

**Remark II: It is noted that the quantitative results might vary on different testing data sets, model selections and scenario parameters such as generation cost and renewable penetrations. The function of this open-source framework is to serve as an experiment platform and baseline case. Future works are encouraged to adjust the models and parameters in this comparison to test different methods.**

![Alt Text](Direct and indirect approaches to achieve decision-focused learning.png)

References:

[1] T. Hong, P. Wang and H. L. Willis, “A Naïve multiple linear regression benchmark for short term load forecasting,” 2011 IEEE Power and Energy Society General Meeting, Detroit, MI, USA, 2011, pp. 1-6.

[2] Stoft, S. (2002). Power system economics: Designing markets for electricity. New York: John Wiley & Sons, IEEE Press.

[3] J. M. Morales, A. J. Conejo and J. Perez-Ruiz, “Economic valuation of reserves in power systems with high penetration of wind power,” IEEE Trans. Power Syst., vol. 24, no. 2, pp. 900-910, May 2009.

[4] Zhang, Jialun, Yi Wang, and Gabriela Hug. "Cost-oriented load forecasting." Electric Power Systems Research 205 (2022): 107723.

[5] J. M. Morales, M. Á. Muñoz, and S. Pineda, “Prescribing net demand for two-stage electricity generation scheduling,” Available at SSRN 4211573.

[6] A. Stratigakos, S. Camal, A. Michiorri, and G. Kariniotakis, “Prescriptive trees for integrated forecasting and optimization applied in trading of renewable energy,” IEEE Trans. Power Syst., vol. 37, no. 6, pp. 46964708, 2022.

[7] T. Carriere and G. Kariniotakis, “An integrated approach for value-oriented energy forecasting and data-driven decision-making application to renewable energy trading,” IEEE Trans. Smart Grid, vol. 10, no. 6, pp. 6933-6944, Nov. 2019.

[8] Hoerl, Arthur E., and Robert W. Kennard. "Ridge regression: Biased estimation for nonorthogonal problems." Technometrics 12.1 (1970): 55-67.


