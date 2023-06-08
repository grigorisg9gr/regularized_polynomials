===================================================
Regularization of polynomial networks for image recognition
===================================================

.. image:: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
	:target: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
	:alt: License

.. image:: https://img.shields.io/badge/Preprint-ArXiv-blue.svg
	:target: https://arxiv.org/abs/2303.13896
	:alt: ArXiv

Official implementation of the image classification experiments in the CVPR'23 paper `"**Regularization of polynomial networks for image recognition**" <https://openaccess.thecvf.com/content/CVPR2023/papers/Chrysos_Regularization_of_Polynomial_Networks_for_Image_Recognition_CVPR_2023_paper.pdf>`_ .

Browsing the folders
====================
The folder structure is the following:

*    ``models``: The folder contains the neural network architectures of R-PolyNets, D-PolyNets, R-PDC and D-PDC.

*    ``configs``: The folder contains the yml files for the configuration, e.g., epochs to run, learning rate changes. Most of the options for hyper-parameters can be changed here and are propagated to the network. For instance, you can change the dataset from CIFAR10 to CIFAR100 by changing the respective name in yml.

*    ``utils``: Misc functions required for training.

Train the network
=================

To train the network for image classification, you can execute the following command::

   python train_main_label_smooth.py --config configs/R_PolyNets_no_activation_functions_cifar10.yml --label any-name-you-want-as-label

To train the network for audio classification, you can execute the following command::

   python train_main_speech_label_smooth.py --config configs/R_PolyNets_no_activation_functions_speech.yml --label any-name-you-want-as-label

You can choose any yml file in ``utils``.

**Changing the dataset:** You can change the dataset by changing the name of the ``dataset/db`` field (in the yml). The datasets that exist by default in PyTorch, e.g., CIFAR10/CIFAR100/MNIST, are automatically downloaded if they do not exist. They are exported in the path ``dataset/root``.

**Changing the model:** You have several options to change the model, e.g., by changing a) the ``model/name`` (in the yml) to the model you want, or b) by changing the ``model/name`` and specify the number of of blocks with the argument ``num_blocks``. Overall, to change the model, the best way to do this is to modify the yml arguments. 

Package Dependencies
====================

Apart from PyTorch, we use a number of standard packages. For instance, pandas, pyyaml and also dropblock. All of those can be install with the pip install command. 


Citing
======
If you use this code, please cite [1]_:

*BibTeX*:: 

  @inproceedings{chrysos2023regularization,
     title={Regularization of polynomial networks for image recognition},
     author={Chrysos, Grigorios and Wang, Bohan and Deng, Jiankang and Cevher, Volkan},
     booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
     year={2023}
  }


References
==========

.. [1] Grigorios G. Chrysos, Bohan Wang, Jiankang Deng, and Volkan Cevher, **Regularization of polynomial networks for image recognition**, *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.


