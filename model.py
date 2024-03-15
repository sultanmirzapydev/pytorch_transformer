import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .utils import TopLayer

class Transformer(nn.Module):
	def __init__(self, config, vocab_size):
		super(Transformer, self).__init__()
		self.token_embedd = nn.Embedding( vocab_size, config['hidden_size'], padding_idx=0)  
		self.encoder = Encoder(self.token_embedd, **config['encoder'])
		self.decoder = Decoder( self.token_embedd, **config['decoder'])
		self.last_layer = TopLayer(config['hidden_size'], vocab_size)

	def forward(self, data):
		pass




