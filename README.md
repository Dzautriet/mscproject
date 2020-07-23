# MSc Project: Crowdsourcing user corrections for supervised learning models.

Z to J: 
Please
- download MNIST dataset from [here](https://drive.google.com/drive/folders/1LiEqIyZbTOmNKRDHgt-qxTQgyGkJl0Jn?usp=sharing) and save them as `./mnist/X.npy` and `./mnist/y.npy`;
- download CIFAR-10 dataset from [here](https://drive.google.com/drive/folders/1B43VTfMrJ4GPA_3O5L3HzeO0q0umtSFf?usp=sharing) and save them as `./cifar10/X.npy`, `./cifar10/y.npy`, `./cifar10/X_test.npy` and `./cifar10/y_test.npy`;
- go to line 105 and line 107 in `reg_main_4.py` to modify number of samples you want to train the classifer with;
- go to line 115 to modify copy probability range.

You can now run `reg_main_4.py` using command line with arguments specifying skill levels (busy user, other users) and dataset. *e.g.* `python reg_main_4.py 0.5 0.4 mnist`, or `python reg_main_4.py 0.5 0.5 cifar10`.

Good luck!

***

