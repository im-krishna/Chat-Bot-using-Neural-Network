import torch
import torch.nn as nn


#we are going to implement feed forward neural net with 2 hidden layers
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.l3=nn.Linear(hidden_size,num_classes)
        self.relu= nn.functional.relu
        
    # x is input    
    def ffnn(self,x):
        #converted the tensor to float and forced into cuda 
        x=x.type(torch.FloatTensor).to('cuda')
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        out=self.relu(out)
        out=self.l3(out)
        return out
    #final cross entropy will be applied which will give us the final probability
