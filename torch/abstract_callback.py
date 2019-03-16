class AbstractCallback:

    def __call__(self, epoch, step, loss, context):
        raise NotImplementedError

    def close(self):
        """ Handle cleanup work if necessary. Will be called after training. """
        pass