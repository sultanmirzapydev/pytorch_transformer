import torch
from torch import nn
from .utils import clones
from .transformer_components import MultiHeadAttention, FeedForward, LayerNorm, TokenEmbedding, \
	 PositionalEncoding

class EncoderLayer(nn.Module):
	def __init__(self, hidden_size,num_head,  mid_dim, dropout):
		super(EncoderLayer,self).__init__()
		self.multi_head_attn = MultiHeadAttention( hidden_size, num_head, dropout)
		self.feed_forward    = FeedForward(mid_dim, hidden_size, dropout)
		self.layer_norms      = clones(LayerNorm(hidden_size),2)
		self.dropout         = nn.Dropout(p=dropout)
	
	def residual_connection(self, res, module, mask=None):
		if module == "mha":
			x = self.layer_norms[0](res)
			x = self.dropout(self.multi_head_attn(x,x,x, mask))
		if module == "ff":
			x = self.layer_norms[1](res)
			x = self.dropout(self.feed_forward(x))
		return res + x


	def forward(self, x, mask=None):
		x = self.residual_connection(x, "mha", mask)
		return  self.residual_connection(x, "ff") 

class Encoder(nn.Module):
	def __init__(self, embeddings,num_head, hidden_size,  max_seq_len, mid_dim, num_layer, dropout=0.1):
		super(Encoder, self).__init__()  
		self.embeddings = embeddings 
		self.position_embed = PositionalEncoding(hidden_size, dropout, max_seq_len)
		self.layer_norm = LayerNorm(hidden_size)
		
		self.layers = clones(EncoderLayer(hidden_size,num_head,  mid_dim, dropout), num_layer)
	def forward(self, x_BS, mask):
		x_BSH = self.embeddings(x_BS) 
		x_BSH = self.position_embed(x_BSH)
		for layer in self.layers:
			x_BSH = layer(x_BSH, mask)
		return self.layer_norm(x_BSH)
