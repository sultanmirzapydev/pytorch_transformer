from torch.nn import ModuleList
from torch import nn
import torch 
from copy import deepcopy

def clones(module, n):
	return ModuleList([deepcopy(module) for _ in range(n)])

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def init_mask(batch_size=None, seq_len=None, mask_type=None,  q_list=None, k_list=None):
	if mask_type == "self" or mask_type == "cross":
		mask_temp = torch.ones((seq_len, seq_len))
		
		for i in range(batch_size):
			ith_mask = mask_temp[q_list[i]:, k_list[i]:] = 0
			mask_temp.stack(ith_mask.unsqueeze(0), dim=0) 

		return mask_temp[1:]

	if mask_type == "causal":
		return torch.tril(torch.ones((seq_len, seq_len)), diagonal=-1)


class TopLayer(nn.Module):
	def __init__(self, hidden_size, vocab_size):
		super(TopLayer, self).__init__()
		self.last_layer = nn.Linear(hidden_size, vocab_size)

	def forward(self, x):
		return self.last_layer(x)
