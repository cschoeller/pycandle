# PyCandle

PyCandle is a lightweight library to make running pytorch experiments easier, better
structured, repeatable and to avoid repetitive boilerplate code.

Here is how this library is used:
```python
train_loader, val_loader = load_datasets(batch_size_train=64, batch_size_test=1000)
experiment = Experiment('mnist_example')
model = Net().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model_trainer = ModelTrainer(model=model, optimizer=optimizer, loss=F.nll_loss, epochs=run_config.epochs, train_data_loader=train_loader, val_data_loader=val_loader, gpu=0)
model_trainer.start_training()
```

This code creates a directory for the current experiment run, runs training and validation of the model, creates a backup of all used code and logs console outputs.

A complete but minimal example for training a model to classify hand-written MNIST digits can be found in [minimal_example/mnist.py](minimal_example/mnist.py).

