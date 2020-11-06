# SOC_estimation_with_GRU-RNN
This research is cited from: C. Li, F. Xiao, Y.Fan. Approach to State of Charge Estimation of Lithium-ion Batteries based on Recurrent Neural Networks with Gated Recurrent Unit. Energies,2019: 12(9)

Abstract
State of charge (SOC) represents the amount of electricity stored and is calculated and used by battery management systems (BMSs). However, SOC cannot be observed directly, and SOC estimation is a challenging task due to the battery's nonlinear characteristics when operating in complex conditions. In this paper, based on the new advanced deep learning techniques, a SOC estimation approach for Lithium-ion batteries using a recurrent neural network with gated recurrent unit (GRU-RNN) is introduced where observable variables such as voltage, current, and temperature are directly mapped to SOC estimation. The proposed technique requires no model or knowledge of the battery's internal parameters and is able to estimate SOC at various temperatures by using a single set of self-learned network parameters. The proposed method is evaluated on two public datasets of vehicle drive cycles and another high rate pulse discharge condition dataset with mean absolute errors (MAEs) of 0.86%, 1.75%, and 1.05%. Experiment results show that the proposed method is accurate and robust.

Dependencies
Keras
Tensorflow
numpy
pandas
sklearn
