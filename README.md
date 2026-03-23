# deeplearning-algorithms-pytorch

## Linear regression

### Steps

1. Create a sample dataset with `sample_size=10000`
2. Generate the labels by linear function `y = Xw + b + noise`
3. Define true `w` and `b` value
4. Read a batch of dataset randomly
5. Initialize the model parameters (`w`, `b`)
6. Define the model, loss function, and stochastic gradient descent function
7. Train the model

### Optimization

This linear regression algorithm uses stochastic gradient descent to take advantage of parallel computing

```python
def sgd(params, lr, batch_size):
    # small batch size sgd
    with torch.no_grad():
        for param in params:
            # update params
            param -= lr * param.grad/batch_size
            param.grad.zero_()
```

### Result

```
epoch 1, loss 0.000050
epoch 2, loss 0.000050
epoch 3, loss 0.000050

w's error of estimate: tensor([-4.7779e-04, -6.6757e-05], grad_fn=<SubBackward0>)
b's error of estimate: tensor([-0.0003], grad_fn=<RsubBackward1>)
```

## Softmax

### Training result

<img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/49bbeef9-ca5f-43e9-8f85-39199bb369b4" />

## MLP
    Activation function: ReLU
### Training result

<img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/8de1f4a3-6a3c-47d9-aece-22fd02a95369" />

## Reference

All the algorithms refer to the book [_Dive into Deep Learning_](https://d2l.ai/)
