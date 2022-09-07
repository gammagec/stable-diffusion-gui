import gc
import torch
from torch import autocast
import random
import os

from src.common.namespace import Namespace
from src.common.subject import Subject
from src.common.value_subject import ValueSubject
from scripts.run import run, load_model

class ParamsModel:
	name = 'params_model'

	def __init__(self, app_context):
		self.seed = 42	
		self.ddim_steps = 50
		self.n_samples = 1
		self.n_iter = 1
		self.ddim_eta = 0
		self.prompt = 'a gorilla drinking a soda'
		self.width = 512
		self.height = 512
		self.channels = 4
		self.downsampling = 8
		self.scale = 7.5
		self.init_image = ValueSubject('init_image', None)
		self.strength = 0.3		
		self.selection_model = app_context.selection_model
		self.config = app_context.config

		self.runs_model = app_context.runs_model
		self.model = None
		self.image_model = app_context.image_model
		self.seed_changed = Subject('seed-changed')
		self.config = app_context.config

		self.image_model.copy_seed_value.register(self, lambda: self.on_copy_seed())
		self.image_model.use_image_value.register(self, lambda: self.on_use_image())
		self.model_loaded = ValueSubject('model_loaded', False)		

	def on_copy_seed(self):
		img = self.selection_model.selected_image
		img = img[11:-4]
		self.seed = img
		self.seed_changed.dispatch()

	def on_use_image(self):
		path = os.path.join(self.config.out_dir)
		path = os.path.join(path, 'sessions')
		path = os.path.join(path, self.selection_model.selected_session.get_value())
		path = os.path.join(path, self.selection_model.selected_run)
		path = os.path.join(path, self.selection_model.selected_image)
		self.init_image.set_value(path)

	def set_random_seed(self):
		self.seed = random.randint(0, 4294960000)

	def load_model(self):
		print('loading model, be patient')
		self.model = load_model(
			self.config.config_file, self.config.ckpt_loc, self.config.embeddings,
			half = False)
		print('done loading model')
		self.model_loaded.set_value(True)

	def on_run(self):
		gc.collect()
		torch.cuda.empty_cache()

		run_folder = False
		if run_folder:	
			init_images = []
			path = self.init_image.get_value()		
			if os.path.exists(path) and os.path.isdir(path):
				for f in os.scandir(path):
					if not f.is_dir():							
						run(self.model, 
							'txt2img' if self.init_image.get_value() == None else 'img2img', Namespace(
							device = "cuda",
							fixed_out = True,
							seed = self.seed,
							ckpt = self.config.ckpt_loc,
							config = self.config.config_file,
							from_file = "",
							ddim_steps = self.ddim_steps,
							small_batch = False,
							fixed_code = False,
							n_samples = self.n_samples,
							n_iter = self.n_iter,
							prompt = self.prompt,
							skip_grid = True,
							skip_save = False,
							ddim_eta = self.ddim_eta,
							H = self.width,
							W = self.height,
							C = self.channels,
							f = self.downsampling,
							n_rows = 0,
							scale = self.scale,
							precision = 'autocast', # or autocast
							plms = False,
							init_img = f.path,
							strength = self.strength,
							session_name = self.selection_model.selected_session.get_value(),
							half = False,
						), lambda: self.after_run())
		else:
			run(self.model, 'txt2img' if self.init_image.get_value() == None else 'img2img', 
				Namespace(
					device = "cuda",
					outdir = self.config.out_dir,
					seed = self.seed,
					ckpt = self.config.ckpt_loc,
					config = self.config.config_file,
					fixed_out = False,
					from_file = "",
					ddim_steps = self.ddim_steps,
					small_batch = False,
					fixed_code = False,
					n_samples = self.n_samples,
					n_iter = self.n_iter,
					prompt = self.prompt,
					skip_grid = True,
					skip_save = False,
					ddim_eta = self.ddim_eta,
					H = self.width,
					W = self.height,
					C = self.channels,
					f = self.downsampling,
					n_rows = 0,
					scale = self.scale,
					precision = 'autocast', # or autocast
					plms = False,
					init_img = self.init_image.get_value(),
					strength = self.strength,
					session_name = self.selection_model.selected_session.get_value(),
					half = False,
			), lambda: self.after_run())

	def after_run(self):
		self.runs_model.after_new_run()
		print('done')