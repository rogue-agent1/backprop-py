#!/usr/bin/env python3
"""Backpropagation from scratch — train a neural network on XOR."""
import sys,math,random

class Layer:
    def __init__(self,nin,nout,rng):
        s=1/math.sqrt(nin)
        self.W=[[rng.uniform(-s,s)for _ in range(nout)]for _ in range(nin)]
        self.b=[0]*nout;self.x=None;self.dW=None;self.db=None
    def forward(self,x):
        self.x=x
        return[sum(x[i]*self.W[i][j]for i in range(len(x)))+self.b[j] for j in range(len(self.b))]
    def backward(self,grad):
        self.dW=[[self.x[i]*grad[j]for j in range(len(self.b))]for i in range(len(self.x))]
        self.db=list(grad)
        return[sum(self.W[i][j]*grad[j]for j in range(len(self.b)))for i in range(len(self.x))]
    def update(self,lr):
        for i in range(len(self.x)):
            for j in range(len(self.b)):self.W[i][j]-=lr*self.dW[i][j]
        for j in range(len(self.b)):self.b[j]-=lr*self.db[j]

def relu(x):return[max(0,v)for v in x]
def relu_grad(x,grad):return[g if v>0 else 0 for v,g in zip(x,grad)]
def sigmoid(x):return[1/(1+math.exp(-max(-500,min(500,v))))for v in x]

class MLP:
    def __init__(self,sizes,seed=42):
        rng=random.Random(seed)
        self.layers=[Layer(sizes[i],sizes[i+1],rng)for i in range(len(sizes)-1)]
    def forward(self,x):
        self.pre_acts=[]
        for i,layer in enumerate(self.layers):
            x=layer.forward(x)
            self.pre_acts.append(list(x))
            if i<len(self.layers)-1:x=relu(x)
        return sigmoid(x)
    def backward(self,output,target,lr=0.1):
        grad=[2*(o-t)*o*(1-o)for o,t in zip(output,target)]
        for i in range(len(self.layers)-1,-1,-1):
            grad=self.layers[i].backward(grad)
            self.layers[i].update(lr)
            if i>0:grad=relu_grad(self.pre_acts[i-1],grad)

def main():
    if len(sys.argv)>1 and sys.argv[1]=="--test":
        net=MLP([2,8,1],seed=42)
        X=[[0,0],[0,1],[1,0],[1,1]];Y=[[0],[1],[1],[0]]
        for _ in range(5000):
            for x,y in zip(X,Y):
                out=net.forward(x);net.backward(out,y,lr=0.5)
        preds=[round(net.forward(x)[0])for x in X]
        assert preds==[0,1,1,0],f"Got {preds}"
        print("All tests passed!")
    else:
        net=MLP([2,8,1]);X=[[0,0],[0,1],[1,0],[1,1]];Y=[[0],[1],[1],[0]]
        for _ in range(5000):
            for x,y in zip(X,Y):out=net.forward(x);net.backward(out,y,lr=0.5)
        for x in X:print(f"  {x} -> {net.forward(x)[0]:.4f}")
if __name__=="__main__":main()
