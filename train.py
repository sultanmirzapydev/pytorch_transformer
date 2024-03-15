import torch
import yaml
from torch import nn
from tqdm import trange, tqdm
from .model import Transformer
#from transformers import BertTokenizer, AdamW , get_linear_schedule_with_warmup
from .utils import dotdict
#from .prepare_dataset import DatasetLoader


if __name__ == "__main__":
	path = "C:\\Users\\LENOVO\\Downloads\\do_it\\paper_implementations\\transformer\\config.yaml"
	with open(path, 'r') as f:
		config = yaml.safe_load(f)
	f.close()
	config = dotdict(config)
	init_trainer = Trainer(config.TrainerConfig) 
	init_trainer.train( config.DataConfig, **config.Train)


class Trainer:
	def __init__(self, TrainerConfig):
		self.data_loader = DatasetLoader()
		self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
		self.model = Transformer(TrainerConfig['model'], 500)
		self.optimizer = AdamW(self.model.parameters(),  lr=TrainerConfig['adam']['lr_rate'], betas=(0.9, 0.98), eps=1e-9) 
		

	def configure_optimizer(self, train_len, epochs):
		self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps= int(0.1 * len(train_len))*epochs,
		 num_training_steps= epochs)
		
	def train(self, DataConfig, epochs, batch_size, model_path): 
		train_set = self.data_loader(DataConfig.base, split=datasets.Split.TRAIN)
		test_set = self.data_loader(DataConfig.base, split=datasets.Split.TEST)
		
		self.configure_optimizer(train_set.num_rows, epochs)
		for eph in trange(epochs):
			training_loss = 0
			for bth_data in tqdm(batched(train_set, batch_size)): 
				memory = self.model.encoder(bth_data['src_ids'], bth_data['src_mask'])
				logits = self.model.decoder(memory, torch.tensor(data['tgt_ids'])[:,:-1], mask=data['src_mask'])
				preds = self.model.last_layer(logits)
				labels = torch.tensor(bth_data['tgt_ids'])[:,1:]
				loss = criterion(preds, labels)
				training_loss += loss.item()

				loss.backward()
				self.optimizer.step()
				self.scheduler.step()
			model.eval()
			validation_loss = 0
			with torch.no_grad():
				for bth_data in tqdm(batched(val_set, batch_size)):
					memory = self.model.encoder(bth_data['src_ids'], bth_data['src_mask'])
					logits = self.model.decoder(memory, torch.tensor(data['tgt_ids'])[:,:-1], mask=data['src_mask'])
					preds = self.model.last_layer(logits)
					labels = torch.tensor(bth_data['tgt_ids'])[:,1:]
					loss = criterion(preds, labels) 
					valid_loss += loss.item()
			print(f"training loss for epoch no. {eph} is {training_loss/len(train_set)}")
			print(f"###################################################################")
			print(f"validation loss after epoch no. {eph} is  {validation_loss/len(val_set)}")
			model.train()
			
		torch.save(self.model.state_dict(), model_path)
		print(f"model saved at {model_path}")
			


class Inference:
	def __init__(self, weight_path, max_len):
		self.weight_path = weight_path
		self.max_len = max_len
		self.data_loader = DatasetLoader()
		self.model = Transformer(num_head, batch_size, hidden_size, len(self.data_loader.tokenizer), max_len, mid_dim, enc_num_layer, dropout=0.1)
		self.load_params(weight_path)
	def load_params(self, path):
		self.model.load_state_dict(torch_load(path))
	def __call__(self, src_text):
		model.eval()
		with torch.inference_mode():
			encode_data = lambda which_set: self.data_loader.tokenizer(which_set, padding='max_length', truncation='longest_first', add_special_tokens=False, max_length= self.max_len)
			src = encode_data("translate Germany to English:<DE/>"+src_text)
			memory = self.model.encoder(src['input_ids'].unsqueeze(0), src['attention_mask']) 
			en_token_id = self.data_loader.tokenizer.convert_token_to_id('<EN/>')
			eos_token_id = self.data_loader.tokenizer.convert_token_to_id('<EOS/>')
			generated_tokens = [en_token_id]
			while self.max_len+50>len(generated_tokens) or generated_tokens[-1] != eos_token_id:
				logits = self.model.decoder(memory, generated_tokens.unsqueeze(0), mask=src['attention_mask'])
				pred = self.model.last_layer(logits)
				pred = nn.softmax(pred[0, -1, :], dim=-1)
				next_token_id = torch.argmax(pred).item()
				generated_tokens.append(next_token_id)
			
			output_text = self.data_loader.tokenizer.decode(generated_tokens, skip_special_tokens=True)
			return output_text






		




