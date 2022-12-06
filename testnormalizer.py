# from collections import namedtuple
# ex = namedtuple('ex', ['a', 'b'])

# x= ex(1,3)
# #x.a=2
# x= x._replace(a=1, b=10)
# print(ex.keys)
# print(x.b)

import torch 
from normalise import Normalizer
if torch.cuda.is_available():
    device= torch.device('cuda:0')
else: 
    device= torch.device('cpu')

a= torch.tensor([[[-0.8295, -5.6911],
                  [-0.1219, 47.4230]],

                  [[-8.3167,  7.1806],
                  [ 4.0020,  9.5887]],

                  [[-5.6678, -4.3174],
                  [-6.9842,  2.7662]]], device=device)

scaler = Normalizer(name='output')#.to(device)
scaler(a.clone())
# print(scaler.inverse(scaler(a, accumulate=False)).get_device())
torch.save(scaler, "./scaler.pth")

scaler= torch.load("./scaler.pth")
# print(scaler.inverse(scaler(a, accumulate=False)).get_device())

assert (scaler.inverse(scaler(a, accumulate=False))== a).all() 

