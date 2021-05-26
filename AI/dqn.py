import torch.nn as nn
import torch.nn.functional as F
import time

class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()

        print(img_height, img_width)
        #time.sleep(5)


        self.fc1 = nn.Linear(in_features=img_height*img_width, out_features=24)   
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=4)
        
    def forward(self, t):
        #print(t.size())
        #time.sleep(5) 
        t = t.flatten(start_dim=0) 
        #print(t.size())
        #time.sleep(5) 
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)
        return t
