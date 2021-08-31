# RESCALING CNN THROUGH LEARNABLE REPETITION OF NETWORK PARAMETERS

## Prerequisites

Install prerequisites with:  
```
pip install -r requirements.txt
```

### Note - 
Make sure to comment out "e2cnn/nn/modules/r2_conv/r2convolution.py" Line number 176 and 177.

# Usage (summary)

The main script offers many options; here are the most important ones:

```
usage: python main.py --model_name {model name} --dataset {dataset: CIFAR10, CIFAR100, rot-MNIST} --num_classes {num_classes} --epochs {number of epochs to train}
```

## Citation
Please cite our paper in your publications if it helps your research. Even if it does not,and you want to make us happy, do cite it :)

    @INPROCEEDINGS{9506158,
    author={Chavan, Arnav and Bamba, Udbhav and Tiwari, Rishabh and Gupta, Deepak},
    booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
    title={Rescaling CNN Through Learnable Repetition of Network Parameters}, 
    year={2021},
    volume={},
    number={},
    pages={754-758},
    doi={10.1109/ICIP42928.2021.9506158}}


## License

This project is licensed under the MIT License.
