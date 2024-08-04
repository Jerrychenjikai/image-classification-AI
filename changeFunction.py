import torch
import math

def change_function(x,kind,m,b):
    decrease=300
    if torch.is_tensor(x):
        x=(m*x+b).float()

        mask=torch.logical_and(kind<=-1/2, x<0)

        x[mask]=0

        mask=torch.logical_and(kind>-1/2, kind<=-1/4)

        x[mask]=torch.sigmoid(x[mask])

        mask=torch.logical_and(kind>-1/4, kind<=0)

        x[mask]=torch.nn.functional.softplus(x[mask])

        mask=torch.logical_and(kind>0, x>0)

        x[mask]=0

        x=x/decrease

        return x

    if kind<=-1/2:#ReLU
        return max(0,x)/decrease
    elif kind<=-1/4:
        try:
            return 1/(1+math.exp(-x))/decrease
        except:
            return 0
    elif kind<=0:#tanh
        try:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))/decrease
        except:
            return 0
    else:
        return min(0,x)/decrease
