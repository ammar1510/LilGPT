import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):     
    def __init__(self,in_dim,head_dim,dropout=0.2):
        super().__init__()
        self.key = nn.Linear(in_dim,head_dim)
        self.query = nn.Linear(in_dim,head_dim)
        self.val = nn.Linear(in_dim,head_dim)
        self.register_buffer('mask',torch.tril(torch.ones((512,512))))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        T = x.shape[1]
        k = self.key(x)         #[b,t,head_dim]
        q = self.query(x)
        v = self.val(x)
        wei = k@q.transpose(-2,-1)

        wei = wei.masked_fill(self.mask[:T,:T]==0,float('-inf'))

        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        out = wei@v

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,in_dim,n_heads,dropout=0.2):
        super().__init__()
        assert in_dim%n_heads==0
        head_dim=in_dim//n_heads
        self.in_proj = nn.Linear(in_dim,in_dim)
        self.mha = nn.ModuleList([Attention(in_dim,head_dim) for _ in range(n_heads)])        
        self.out_proj = nn.Linear(in_dim,in_dim)
    
    def forward(self,x):
        x = self.in_proj(x)
        x = torch.concat([attn(x) for attn in self.mha],dim=-1)

        x = self.out_proj(x)
        return x 

class FeedForward(nn.Module):
    def __init__(self,in_dim,dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim,4*in_dim),
                                    nn.ReLU(),nn.Linear(4*in_dim,in_dim),
                                    nn.Dropout(dropout)
                                    )

    def forward(self,x):
        return self.layers(x)

        
class Layer(nn.Module):
    def __init__(self,in_dim,n_heads,dropout=0.2):
        super().__init__()
        self.attn = MultiHeadAttention(in_dim,n_heads=n_heads,dropout=dropout)
        self.ffw = FeedForward(in_dim,dropout=dropout)
        self.lna =  nn.LayerNorm(in_dim)
        self.lnf = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.attn(self.lna(x))+x
        
        x = self.ffw(self.lnf(x))+x
        x = self.dropout(x)
        return x
        
class GPT(nn.Module):
    def __init__(self,vocab_len,context_len,n_layers,in_dim,n_heads,device,dropout=0.2):
        #add token embedding and position embedding
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_len,in_dim)
        self.pos_emb = nn.Embedding(context_len,in_dim)
        self.device=device

        self.vocab_len = vocab_len
        self.context_len = context_len
        
        self.layers = nn.Sequential(*[Layer(in_dim,n_heads,dropout=dropout) for _ in range(n_layers)])
        self.out = nn.Linear(in_dim,vocab_len)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x,target=None):
        
        pos = self.pos_emb(torch.arange(x.shape[1],device=self.device))
        x = self.tok_emb(x)+pos
        
        x = self.layers(x)
        logits = self.out(x)
        if target is not None:
            logits = logits.view((-1,self.vocab_len)) 
            target = target.view((-1,))
            loss = self.criterion(logits,target)
        else:
            loss = None
        
        return logits,loss
    
    def generate(self,input,num_tokens):
        
        
        for i in range(num_tokens):
            idx = input[:,-self.context_len:]
            logits,_ = self.forward(idx)
            logits = logits[:,-1,:]
            logits = F.softmax(logits,dim=1)
            pred = torch.multinomial(logits,num_samples=1,replacement=True)
            # print("input: ",input)
            # print("pred: ",pred)
            input = torch.cat((input,pred),dim=1)
        
        return input


