*SLL_IQA*
===============
**This is official repository of article ：“Semi-Supervised Authentically Distorted Image Quality Assessment with Consistency-Preserving Dual-Branch Convolutional Neural Network” . TMM2022.
DOI: 10.1109/TMM.2022.3209889
https://ieeexplore.ieee.org/abstract/document/9903545**
===============

## SSL-IQA

<img src="https://user-images.githubusercontent.com/72659127/232423969-ca56ff3f-ed75-4ecc-be5a-96aef370ac4e.png" width="500" />

## Install

```js
install requeried tools: pytoch numpy, ect
```

## Usage
```
$ readme README.md
```

```js

// Checks readme in current working directory

You just need 
1)define your own dataset and network.(dataset  and   model_hub)
2)define your own parse_config
3)run baseline.py to get the baseline model.
4)run train_kd_semi.py to get the SSL-IQA model.
5)change the parse_config (train to test) and choose the model you saved for testing.


Among them, checkpoint, data, logs, results, and runs are respectively used to save the trained model,
the dataset and partitions required for training, training logs, and training results. It needs to be 
adjusted according to the path in your computer, etc. 
```

## Contributing

In this paper, we propose a semi-supervised NR-IQA framework, termed SSLIQA, for authentically distorted images.SSLIQA adopts an asymmetric parallel dual-branch structure,and its success lies in simultaneously exploiting both labeled and unlabeled images with the assistance of a consistency preserving strategy. Concretely, such a strategy, inspired by the subjective scoring behaviors, enforces the student to mimic activations of the teacher, and helps to explore the intrinsic relation between images. Extensive experiments and ablation studies demonstrate that our SSLIQA is superior to ten state of-the-art NR-IQA methods with considerable effectiveness
and generalization. Moreover, benefiting from the consistency preserving strategy and the asymmetric network structure, our SSLIQA can effectively exploit the unlabeled data to achieve higher IQA performance with a smaller network. This points to an interesting avenue for future work.

### Citation
```
@article{yue2022semi,
  title={Semi-supervised authentically distorted image quality assessment with consistency-preserving dual-branch convolutional neural network},
  author={Yue, Guanghui and Cheng, Di and Li, Leida and Zhou, Tianwei and Liu, Hantao and Wang, Tianfu},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}

```
