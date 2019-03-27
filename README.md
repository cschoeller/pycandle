# PyCandle

PyCandle is a lightweight library for pytorch to make running pytorch experiments easy, structured, repeatable and to avoid boilerplate code.

Here is how PyCandle is used:
```python
train_loader, val_loader = load_datasets(batch_size_train=64, batch_size_test=1000)
experiment = Experiment('mnist_example')
model = Net().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model_trainer = ModelTrainer(model=model, optimizer=optimizer, loss=F.nll_loss, epochs=20, 
                        train_data_loader=train_loader, val_data_loader=val_loader, gpu=0)
model_trainer.start_training()
```

This code snippet creates a directory for the current experiment, runs training and validation of the model, creates a backup of all used code and forks console output in a log file.

A complete example for training a model to classify hand-written MNIST digits can be found in [minimal_example/mnist.py](minimal_example/mnist.py).
