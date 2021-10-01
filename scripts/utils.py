import os
import time

class timer():
    def __init__(self):
        self.laps = []

    def start(self):
        self.tic = time.time()

    def stop(self):
        self.laps.append(time.time() - self.tic)
        return self.laps[-1]

    def sum(self):
        return sum(self.laps)

# fast model boot up
def load_IT(modelpkl, target='GA', eval_trainset=False):
    # get md and error for node color
    import sys
    sys.path.append('/home/ngr4/project')
    from wearables.scripts import train_v3 as weartrain
    
    trainer = weartrain.InceptionTime_trainer(exp='preload', target=target)
    res = trainer.eval_test(modelpkl, eval_trainset=eval_trainset)

    return trainer, res