from datasets import load_dataset
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

class DatasetLoader:
	def __init__(self):
		
		self.checkpoint = "t5-small"
		self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
		self.set_special_tokens()

	def set_special_tokens(self):
		self.tokenizer.add_tokens(['<EN/>','<DE/>','<EOS/>'], special_tokens=True)
		#special_tokens_dict = {"english": '<en/>', "german": '<de/>'}
		#self.tokenizer.add_special_tokens(special_tokens_dict)

	def get_token_id(self, token):
		return self.tokenizer.convert_tokens_to_ids([token])

	def __call__(self, path, name=None, batch_size=None, max_length=None, dec_length=None, split=None):
		raw_dataset = load_dataset(path, name=name, split=split) 
		return raw_dataset.map(self.pre_process, batched=True, batch_size=batch_size, fn_kwargs={'max_length':max_length, 'dec_length':dec_length})
		
	
	def pre_process(self, examples, max_length=None, dec_length=None):
		src_lng = 'de'
		tgt_lng = 'en'
    
		english_set, german_set = [],[] 
		prompt = "translate Germany to English:<DE/>"
		for example in examples['translation']:
			german_set.append(prompt + example[src_lng])
			english_set.append('<EN/>' + example[tgt_lng] + '<EOS/>' )
   
			
		encoded_data = lambda which_set, length: self.tokenizer(which_set, padding='max_length', truncation='longest_first', add_special_tokens=False, max_length= length)
		src_ids = encoded_data(german_set, max_length) 
    
		tgt_ids = encoded_data(english_set, dec_length if dec_length else max_length)
    
		return {'dec_length':dec_length if dec_length else max_length, 'src_ids':src_ids['input_ids'], 'src_mask':src_ids['attention_mask'], 'tgt_ids': tgt_ids['input_ids'], 'tgt_mask': tgt_ids['attention_mask']}

