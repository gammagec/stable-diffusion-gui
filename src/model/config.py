import os
import yaml

class Config:	
	def __init__(self):
		self.ckpt_loc = ''
		self.out_dir = ''
		self.config_file = ''
		self.embeddings = []

	def load(self):
		config_path = os.path.join(".", "config.yaml")
		if not os.path.exists(config_path):
			print("No config.yaml file found")
			return
		file = open(config_path)		
		config = yaml.load(file, Loader=yaml.UnsafeLoader)			
		self.ckpt_loc = config.ckpt_loc
		self.out_dir = config.out_dir
		self.config_file = config.config
		self.embeddings = config.embeddings
		print('loaded config')
		print(f'ckpt: {self.ckpt_loc}')
		print(f'out_dir: {self.out_dir}')
		print(f'config: {self.config_file}')