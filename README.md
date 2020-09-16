# Crowdsourcing user corrections for supervised learning models.

This repository contains the source code of my research project collaborated with JP Morgan AI Research: **Crowdsourcing user corrections for supervised learning models**, completed as part requirement for the MSc Data Science and Machine Learning at University College London.

## Recommended requirements
- Python 3.7.4
- PyTorch 1.4.0
- Torchvision 0.5.0
- Numpy 1.18.1 
- Matplotlib 3.1.3 

## Download data

Datasets are not included in this repo. Please
- download MNIST dataset from [here](https://drive.google.com/drive/folders/1LiEqIyZbTOmNKRDHgt-qxTQgyGkJl0Jn?usp=sharing) and save them as `./mnist/X.npy` and `./mnist/y.npy`;
- download CIFAR-10 dataset from [here](https://drive.google.com/drive/folders/1B43VTfMrJ4GPA_3O5L3HzeO0q0umtSFf?usp=sharing) and save them as `./cifar10/X.npy`, `./cifar10/y.npy`, `./cifar10/X_test.npy` and `./cifar10/y_test.npy`.

## Run experiment

If you want to reproduce the results.

Run `four_models.py` from your terminal with arguments specifying skill levels ($m_0$, other users) and the dataset *e.g.* `python four_models.py 0.5 0.4 mnist`, or `python four_models.py 0.5 0.5 cifar10` to compare the performance of our model with three benchmarks:
- Vanilla NN
- Weighted majority vote
- MBEM
`run_experiment.cmd` provides an example of multiple runs. 

By default, plotting is disabled in `four_models.py`. You can uncomment line 56 and `# plot_conf_mat_h(est_conf, conf)` to enable plotting confusion matrix estimates and final result.

`ablation_study.py` is for ablation study.

`balance_vs_imblance.py` compares the performance of the vanilla-NN in balanced label count setting and unbalanced label count setting.

`confmat_evolution.py` shows how confusion matrix estimate evolves as abstention rate increases.



