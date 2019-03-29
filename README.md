![# PyCandle](logo.png)

PyCandle is a lightweight library for pytorch that makes running experiments easy, structured, repeatable and avoids boilerplate code. It maintains flexibilty and allows to train also more complex models like recurrent or generative neural networks conveniently.

### Usage

This code snippet creates a timestamped directory for the current experiment, runs the training of the model, creates a backup of all used code, logs current git hash and forks console output into a log file:

```python
model = Net().cuda()
experiment = Experiment('mnist_example')
train_loader = load_dataset(batch_size_train=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model_trainer = ModelTrainer(model, optimizer, F.nll_loss, 20, train_loader, gpu=0)
model_trainer.start_training()
```

A complete example for training a model to classify hand-written MNIST digits can be found in [minimal_example/mnist.py](minimal_example/mnist.py).
