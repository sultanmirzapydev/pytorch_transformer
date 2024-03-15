import torch
from torch import nn
from .encoder import Encoder
from .utils import clones, init_mask
from .transformer_components import MultiHeadAttention, FeedForward, LayerNorm, TokenEmbedding, \
	 PositionalEncoding



class DecoderLayer(nn.Module):
	def __init__(self, num_head, hidden_size, mid_dim, dropout):
		super(DecoderLayer, self).__init__()
		self.multi_head_causal_attn = MultiHeadAttention(hidden_size, num_head, dropout)
		self.multi_head_cross_attn = MultiHeadAttention(hidden_size, num_head, dropout)
		self.feed_forward    = FeedForward(mid_dim, hidden_size, dropout)
		self.layer_norms      = clones(LayerNorm(hidden_size), 3)
		self.dropout         = nn.Dropout(p=dropout)

	def residual_connection(self, res, module, mem=None, mask=None):
		if module == "causal-mha":
			dec_q_length = x.shape[1]
			x = self.layer_norms[0](res)
			mask = init_mask(seq_len= dec_q_length, mask_type='causal')
			x = self.multi_head_causal_attn(x,x,x, mask)
		elif module == "cross-mha":
			x = self.layer_norms[1](res) 
			x = self.multi_head_cross_attn(x, mem,mem, mask)
		elif module == "ff":
			x = self.layer_norms[2](res) 
			x = self.feed_forward(x)
		return res + x


	def forward(self, mem, x, mask=None):
		x = self.residual_connection(x, "causal-mha",)
		x = self.residual_connection(x, "cross-mha", mem, mask)
		return self.residual_connection(x, "ff")



class Decoder(nn.Module):
	def __init__(self,embeddings, num_head, hidden_size,  max_seq_len, mid_dim, num_layer, dropout=0.1):
		super(Decoder, self).__init__()
		self.embeddings = embeddings
		self.position_embed = PositionalEncoding(hidden_size, dropout, max_seq_len)
		self.layer_norm = LayerNorm(hidden_size)
		self.decoder_layers = clones(DecoderLayer(num_head, hidden_size, mid_dim, dropout), num_layer)
	def forward(self, mem, x, mask=None):
		x = self.embeddings(x)
		x = self.position_embed(x)
		for layer in self.decoder_layers:
			x = layer(mem, x, mask)
		return self.layer_norm(x)

		


