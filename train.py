import numpy as np
import re
import torch
import gc
import sys  
from torch import autograd
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import GPT


"""hyperparameters:"""
batch_size = 64
context_len = 256
num_epochs = 5
val_step = max(1,num_epochs/10)
lr = 4e-4
in_dim = 384
num_heads = 6
n_layers = 6
device = "mps" if torch.backends.mps.is_available() else "cpu"
vocab_size=70
train = False

decoder,encoder = None,None

if len(sys.argv)>1:
    num_epochs=sys.argv[1]



with open("drake.txt","r",encoding="UTF-8") as file:
    corpus = file.read()


def preprocess(corpus):
    cleaned_text = re.sub(r'[^a-zA-Z0-9&.[\]()!{}:"\'/\\,]', ' ', corpus)
    vocab = list(sorted(set(cleaned_text)))
    idx = list(range(len(vocab)))
    encode = {k:v for k,v in zip(vocab,idx)}
    decode = {k:v for v,k in encode.items()}
    global encoder,decoder
    encoder = lambda word: [encode[w] for w in word]
    decoder = lambda input: ''.join([decode[x] for x in input])


    global vocab_size
    vocab_size =len(decode)

    return cleaned_text

corpus = preprocess(corpus)



        
class dataset(Dataset):
    def __init__(self,corpus,context_len):
        super().__init__()

        self.input = encoder(corpus)
        self.input = torch.tensor(self.input)
        self.size = len(corpus)-context_len

        self.context_len = context_len
        self.idx = list(range(self.size))

    def __len__(self):
        return self.size
    
    def __getitem__(self,index):
        
        x = self.input[index:index+self.context_len]
        y = self.input[index+1:index+self.context_len+1]
        return x,y
        
    
def dataloader(corpus,batch_size,context_len,shuffle=True,cv=0.1,num_workers=0):
    tr_dataset = dataset(corpus,context_len=context_len)
    cv_dataset = dataset(corpus,context_len=context_len)

    idx = list(range(len(tr_dataset)))
    
    if shuffle:
        np.random.shuffle(idx)

    n = int(len(idx)*cv)   
    
    tr_idx = idx[:n]
    cv_idx = idx[n:]
    # print(tr_idx)
    tr_sampler = SubsetRandomSampler(tr_idx)
    cv_sampler = SubsetRandomSampler(cv_idx)

    tr_loader = DataLoader(tr_dataset,batch_size=batch_size,num_workers=num_workers,sampler=tr_sampler)
    cv_loader = DataLoader(cv_dataset,batch_size=batch_size,num_workers=num_workers,sampler=cv_sampler)

    return tr_loader,cv_loader



if __name__ == "__main__":

    
    train,val = dataloader(corpus,batch_size,context_len,num_workers=4)
    print(vocab_size)

    model = GPT(vocab_size,context_len,n_layers,in_dim,num_heads,device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    for epoch in range(num_epochs):
        
        for i,(x,y) in enumerate(train):
            x,y = x.to(device),y.to(device)
            logits,loss = model(x,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            del x,y,logits
            torch.cuda.empty_cache()
            gc.collect()

        print(f"train set loss:{loss}")        
        if epoch%val_step==val_step-1:

            with torch.no_grad():
                net_loss,cnt = 0,0

                for (x,y) in val:
                    x,y=x.to(device),y.to(device)
                    _,loss = model(x,y)
                    net_loss+=loss
                    cnt+=1
    
            print(f"val set loss:{net_loss/cnt}")







    


