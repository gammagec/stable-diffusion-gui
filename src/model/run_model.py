import os
import yaml
from src.common.subject import Subject

class RunModel:
	name = 'run_model'

	def __init__(self, app_context):
		self.run = None		
		self.channels = 4
		self.width = 512
		self.height = 512
		self.ckpt = ''
		self.config = ''
		self.ddim_eta = 0.0
		self.ddim_steps = 50
		self.device = 'cuda'
		self.f = 8
		self.fixed_code = False
		self.from_file = ''
		self.half = True
		self.init_img = ''
		self.n_iter = 10
		self.n_rows = 0
		self.n_samples = 1
		self.outdir = ''
		self.plms = False
		self.precision = 'autocast'
		self.prompt = 'test prompt'
		self.scale = 7.5
		self.seed = 42
		self.session_name = 'test'
		self.skip_grid = True
		self.skip_save = False
		self.strength = 0.3
		self.small_batch = False
		self.runs_model = app_context.runs_model
		self.update_run_model_subject = Subject('update_run_model')		
		self.selection_model = app_context.selection_model

		self.selection_model.run_selected.register(self, lambda: self.on_run_selected())

	def on_run_selected(self):
		run = self.selection_model.selected_run
		self.run = run
		if (run != None):
			path = os.path.join(self.runs_model.session_path, run)
			path = os.path.join(path, "config.yaml")
			print(f'loading run info for {path}')
			if os.path.exists(path):
				with open(path) as file:				
					config = yaml.load(file, Loader=yaml.UnsafeLoader)	
					self.channels = config.C
					self.width = config.W
					self.height = config.H
					self.ckpt = config.ckpt
					self.config = config.config
					self.ddim_eta = config.ddim_eta
					self.ddim_steps = config.ddim_steps
					self.device = config.device
					self.f = config.f
					self.fixed_code = config.fixed_code
					self.from_file = config.from_file
					self.half = config.half
					self.init_img = config.init_img
					self.n_iter = config.n_iter
					self.n_rows = config.n_rows
					self.n_samples = config.n_samples
					self.outdir = config.outdir
					self.plms = config.plms
					self.precision = config.precision
					self.prompt = config.prompt
					self.scale = config.scale
					self.seed = config.seed
					self.session_name = config.session_name
					self.skip_grid = config.skip_grid
					self.skip_save = config.skip_save
					self.strength = config.strength
					self.small_batch = config.small_batch
					
		self.update_run_model_subject.dispatch()