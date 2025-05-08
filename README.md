## Code for Feedback-Calibrated Meta-Adaptation

This repository contains the code for our paper [Towards Non-Structural Disturbance Prediction: Feedback-Calibrated Meta-Adaptation](https://nonstructural-metalearn.github.io), submitted to CoRL 2025. 

The modules and theirs contents are listed as follows.

| Module             | Content                                                      |
| ------------------ | ------------------------------------------------------------ |
| meta_learning      | The algorithm for learning meta-representations with given batch trajectory data. |
| online_adapt_sim   | Simulations of online model adaptation in quadrotor trajectory tracking control, also with the ablation study on the models. |
| quadrotor_batchsim | A batch simulator that generates quadrotor motions for meta-learning, including a trajectory generator, a baseline controller and a disturbance generator. |

This is only a rough version. Further refinements of the documentary and API are on the way.
