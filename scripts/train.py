import torch
import torch.nn as nn

import sys
sys.path.append('/home/ngr/gdrive/wearables/scripts/')
import models as wearmodels
import data as weardata



class InceptionTimeRegressor_trainer():
    def __init__(self, device=torch.cuda if torch.cuda.is_available() else torch.cpu()):
        self.model = wearmodels.InceptionTime(1, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
