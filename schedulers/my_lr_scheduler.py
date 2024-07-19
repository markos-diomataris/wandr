from torch.optim.lr_scheduler import ReduceLROnPlateau

class MyReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, **kwargs):
       super().__init(**kwargs) 
       self.times_reduced_lr = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
        self.times_reduced_lr += 1
