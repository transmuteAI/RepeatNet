# Re-scaling-Neural-Networks-through-Learnable-Repetition-of-Network-Parameters

## Prerequisites

Install prerequisites with:  
```
pip install -r requirements.txt
```

### Note - 
Make sure to comment out "e2cnn/nn/modules/r2_conv/r2convolution.py" Line number 176 and 177

# Usage (summary)

The main script offers many options; here are the most important ones:

```
usage: python main.py --model_name {model name} --dataset {dataset: CIFAR10, CIFAR100, rot-MNIST} --num_classes {num_classes} --epochs {number of epochs to train}
```