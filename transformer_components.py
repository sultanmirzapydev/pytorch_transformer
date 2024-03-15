import torch
from torch import nn
from .utils import clones, init_mask

class TokenEmbedding(nn.Module):
	def __init__(self, hidden_size, vocab_size):
		super(TokenEmbedding, self).__init__()
		self.token_embed = nn.Embedding(vocab_size, hidden_size, padding_idx = 0,)
		self.hidden_size = torch.tensor(hidden_size, requires_grad=False)

	def forward(self, x):
		return self.token_embed(x) * torch.rsqrt(self.hidden_size)


class PositionalEncoding(nn.Module):
	def __init__(self, hidden_size, dropout, max_len,  base = 10000.0):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		position_embed = torch.zeros(max_len, hidden_size) 
		token_positions = torch.arange(0, max_len).unsqueeze(1) 
		div_term = 1/(base**(torch.arange(0,hidden_size,2)/hidden_size))
		position_embed[:,0::2] = torch.sin(token_positions*div_term)
		#position_embed[:,1::2] = torch.cos(token_positions*div_term) 
		#position_embed = position_embed.unsqueeze(0)
		self.register_buffer("position_embed", position_embed) 

	def forward(self, x):
		x = x + self.position_embed[ : x.size(1)].requires_grad_(False)
		return self.dropout(x)

class MultiHeadAttention(nn.Module):
	def __init__(self, hidden_size, num_head, dropout=0.1):
		super(MultiHeadAttention, self).__init__()
		self.head_dim = hidden_size//num_head
		self.num_head = num_head
		self.linears  = clones(nn.Linear(hidden_size, hidden_size), 4)
		self.dropout  = nn.Dropout(p=dropout) 
		
	def forward(self, pre_query, pre_key, pre_value,  mask=None):
		print(self.head_dim, self.num_head)
		query,key,value = [linear(x).view( x.shape[0], -1, self.num_head, self.head_dim).transpose(1, 2) for linear,
						   x in zip(self.linears,(pre_query, pre_key, pre_value))]
		
		attention_score = self.attention(query,key,value, mask, dropout=self.dropout)
		query, key, value = [None]*3
		x = attention_score.transpose(1, 2).contiguous().view(pre_query.shape[0], -1, self.num_head * self.head_dim)
		return self.linears[-1](x)

	@staticmethod
	def attention(query, key, value, mask=None, dropout = None):
		d_k     = torch.tensor(query.shape[-1], requires_grad=False)
		pre_softmax = torch.matmul(query, key.transpose(-2,-1))/torch.rsqrt(d_k)
		if mask != None:
			 pre_softmax = pre_softmax.masked_fill(mask==0, -9e15)
		scores = pre_softmax.softmax(dim=-1)
		scores = dropout(scores)
		return torch.matmul(scores, value)


class FeedForward(nn.Module):
	def __init__(self, mid_dim, hidden_size, dropout):
		super(FeedForward, self).__init__()
		self.fst_linear = nn.Linear(hidden_size, mid_dim)
		self.sec_linear = nn.Linear(mid_dim, hidden_size)
		self.dropout    = nn.Dropout(dropout)

	def forward(self, x):
		return self.sec_linear(self.dropout(self.fst_linear(x).relu()))



class LayerNorm(nn.Module):
	def __init__(self, h_dim):
		super(LayerNorm, self).__init__()
		self.weights = nn.Parameter(torch.ones(h_dim))
		self.bias    = nn.Parameter(torch.zeros(h_dim))

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std  = x.std(-1, keepdim=True)
		return self.weights * (x-mean) / (std+1e-6) + self.bias 
