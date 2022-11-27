import torch
from torch import nn
import torch.nn.functional as F
import random

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self,hid_dim,n_heads,dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0

        self.hid_dim_per_heads = hid_dim // n_heads
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        self.wq = nn.Linear(hid_dim,hid_dim)
        self.wk = nn.Linear(hid_dim,hid_dim)
        self.wv = nn.Linear(hid_dim,hid_dim)

        self.fc = nn.Linear(hid_dim,hid_dim)
        self.drop = nn.Dropout(dropout)

    # def _tensor_splite_mul(self,a,b,gpu_id,piece=2,dim=2):
    #     # a = a.cpu()
    #     spl_a = torch.chunk(a,piece,dim)
    #     res = []
    #     # bc = b.cuda(gpu_id%2)
    #     bd = b.cpu()
    #     j = 1
    #     for pa in spl_a:
    #         if j%3 == 1:
    #             pa = pa.cuda(gpu_id)
    #             c = torch.matmul(pa,b)
    #         elif j%3 == 2:
    #             c = torch.matmul(pa,bd)
    #         elif j%3 == 0:
    #             pa = pa.cuda(gpu_id)
    #             c = torch.matmul(pa,b)
    #         j += 1
    #         c = c.cpu()
    #         torch.cuda.empty_cache()
    #         res.append(c)
    #     return torch.cat(res,dim)

    def forward(self,query,key,value,mask,gpu_id):
        bz = query.shape[0]
        value = value.cuda(gpu_id)
        self.wv = self.wv.cuda(gpu_id)

        self.wq = self.wq.cuda(gpu_id-1)
        self.wk = self.wk.cuda(gpu_id-1)

        # put it to the secondary GPU
        self.fc = self.fc.cuda(gpu_id%2)

        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)
        scale = torch.sqrt(torch.LongTensor([self.hid_dim_per_heads])).cuda(gpu_id%2)

        Q = Q.view(bz,-1,self.n_heads,self.hid_dim_per_heads).permute(0,2,1,3)
        K = K.view(bz,-1,self.n_heads,self.hid_dim_per_heads).permute(0,2,1,3)
        V = V.view(bz,-1,self.n_heads,self.hid_dim_per_heads).permute(0,2,1,3)

        attention = torch.matmul(Q,K.permute(0,1,3,2))

        #Split computation, put it to secondary GPU
        attention = attention.cuda(gpu_id%2)
        attention = attention/scale

        torch.cuda.empty_cache()
        mask = mask.cuda(gpu_id%2)
        attention = attention.masked_fill(mask==0,-10000)

        # back to the main GPUs
        attention = attention.cuda(gpu_id)
        attention = F.softmax(attention,dim=-1)

        attention = self.drop(attention)
        
        torch.cuda.empty_cache()
        attention = torch.matmul(attention,V)
        # attention = self._tensor_splite_mul(attention,V,gpu_id,self.n_heads//2)

        attention = attention.permute(0,2,1,3).contiguous()
        attention = attention.view(bz,-1,self.n_heads*self.hid_dim_per_heads)

        attention = self.fc(attention.cuda(gpu_id%2))
        return attention
    
# FeedForward
class FeedForward(nn.Module):
    def __init__(self,hid_dim,pf_dim,dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim,pf_dim)
        self.fc2 = nn.Linear(pf_dim,hid_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self,x,gpu_id):
        self.fc1 = self.fc1.cuda(gpu_id+1)
        self.fc2 = self.fc2.cuda(gpu_id+1)
        x = self.fc1(x)
        x = F.gelu(x)

        # put it to secondary GPU
        # x = x.cuda(gpu_id%2)
        torch.cuda.empty_cache()
        x = self.drop(x)
        x = x.double()
        x = torch.where(x<0.0,0.0,x)
        x = x.float()

        # back to main GPUs
        x = x.cuda(gpu_id+1)
        x = self.fc2(x)
        return x

# Ecoder block or Layer
class EncoderLayer(nn.Module):
    def __init__(self,hid_dim,n_heads,pf_dim,dropout):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.atten = MultiHeadAttention(hid_dim,n_heads,dropout)
        self.fd = FeedForward(hid_dim,pf_dim,dropout)
    
    def forward(self,x,mask_x,gpu_id):

        x = x.cuda(gpu_id-1)
        # mask_x = mask_x.cuda(gpu_id%2)
        self.ln1 = self.ln1.cuda(gpu_id-1)
        q = self.ln1(x)
        attention = self.atten(q,q,q,mask_x,gpu_id)
        attention = attention.cuda(gpu_id+1)
        x = x.cuda(gpu_id+1) + attention

        torch.cuda.empty_cache()
        
        self.ln2 = self.ln2.cuda(gpu_id+1)
        x = x + self.fd(self.ln2(x),gpu_id)
        return x
        
    

# Pre-training model
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.input_dim = config.input_dim
        self.tok_embed = nn.Embedding(config.input_dim,config.hid_dim)
        self.pos_embed = nn.Embedding(70000,config.hid_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(config.hid_dim,config.n_heads,config.pf_dim,config.dropout) for _ in range(config.n_layers)]
        )
        self.drop = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hid_dim,config.output_dim)
        self.pad_idx = config.pad_idx

    def _make_mask(self,x):
        mask_x = (x==self.pad_idx).unsqueeze(1)
        mask_tril = 1 - torch.tril(torch.ones(1,mask_x.shape[-1],mask_x.shape[-1]))
        mask_x = mask_x + mask_tril
        mask_x = mask_x > 0
        mask_x = (mask_x==1).unsqueeze(1)
        mask_x = mask_x.expand(-1,1,mask_x.shape[-1],mask_x.shape[-1])
        return mask_x
    
    def _embed(self,x):
        tok_emb = self.tok_embed(x)
        pos = torch.arange(0,x.shape[1]).unsqueeze(0).repeat(x.shape[0],1)
        pos_emb = self.pos_embed(pos)
        x = self.drop(pos_emb + tok_emb)
        return x

    def forward(self,x,flag=True):
        mask_x = self._make_mask(x)
        x = self._embed(x)

        i = 3
        # jump to next
        j = 0
        rnd = [0,1,1,1,1,1,1,1,1,1]
        if flag:
            for layer in self.layers:
                # choose to jump
                if j == 1:
                    continue
                x = layer(x,mask_x,i)
                i = i%6 + 3
                # update 
                j = random.choice(rnd)
        else:
            for layer in self.layers:
                x = layer(x,mask_x,i)
                i = i%6 + 3
            
        # put it to secondary GPU
        x = x.cuda(0)
        self.fc = self.fc.cuda(0)
        x = self.fc(x)
        return x