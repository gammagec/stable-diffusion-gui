from tkinter import StringVar, IntVar

from src.common.equals_observer import EqualsObserver
from src.common.not_observer import NotObserver
from src.common.and_observer import AndObserver

from src.common.bindings import bind_var_to_value_observer

class ParamsViewModel(object):

	def __init__(self, model, app_context):
		self.model = model

		self.seed = StringVar()
		self.ddim_steps = StringVar()
		self.n_samples = StringVar()
		self.n_iter = StringVar()
		self.width = StringVar()
		self.height = StringVar()
		self.channels = StringVar()
		self.downsampling = StringVar()
		self.scale = StringVar()
		self.use_image_prompt = IntVar()
		self.image_prompt = StringVar()
		self.strength = StringVar()
		self.prompt = StringVar()
		self.enable_load_model_button = IntVar(value = True)

		self.run_enabled = AndObserver('',
			NotObserver('', EqualsObserver('', app_context.selection_model.selected_session, None)),
			model.model_loaded)

		bind_var_to_value_observer(
			self, self.enable_load_model_button, NotObserver('', model.model_loaded))

		model.seed_changed.register(self, lambda: self.seed_changed())
		model.init_image.register(self, lambda val: self.on_update_init_image(val))	
		self.update_from_model()

	def use_image_prompt_checked(self):		
		val = self.use_image_prompt.get()
		if not val:
			self.image_prompt.set('')
			self.update_init_image_to_model()

	def update_init_image_to_model(self):
		val = self.image_prompt.get()
		self.model.init_image.set_value(val if val != '' else None)

	def run_clicked(self):
		self.run()

	def run_random_clicked(self):
		self.set_random_seed()
		self.run()

	def run(self):			
		self.params_to_model()
		self.model.on_run()

	def show_clicked(self):
		self.update_init_image_to_model()		

	def load_model_clicked(self):
		self.model.load_model()

	def set_random_seed(self):
		self.model.set_random_seed()
		self.seed.set(str(self.model.seed))

	def on_update_init_image(self, val):
		self.image_prompt.set(val if val != None else '')

	def seed_changed(self):
		self.seed.set(self.model.seed)

	def params_to_model(self):
		self.model.seed = int(self.seed.get())
		self.model.ddim_steps = int(self.ddim_steps.get())
		self.model.n_samples = int(self.n_samples.get())
		self.model.n_iter = int(self.n_iter.get())
		self.model.ddim_eta = 0
		self.model.prompt = self.prompt.get()
		self.model.width = int(self.width.get())
		self.model.height = int(self.height.get())
		self.model.channels = int(self.channels.get())
		self.model.downsampling = int(self.downsampling.get())
		self.model.scale = float(self.scale.get())
		prompt_val = self.image_prompt.get()
		print(f'prompt val {prompt_val}')
		self.model.init_image.set_value(
			self.image_prompt.get() if self.use_image_prompt and prompt_val != '' else None)
		self.model.strength = float(self.strength.get())

	def update_from_model(self):
		self.seed.set(str(self.model.seed))
		self.ddim_steps.set(str(self.model.ddim_steps))		
		self.n_samples.set(str(self.model.n_samples))
		self.n_iter.set(str(self.model.n_iter))
		self.width.set(str(self.model.width))
		self.height.set(str(self.model.height))		
		self.channels.set(str(self.model.channels))
		self.downsampling.set(str(self.model.downsampling))
		self.scale.set(str(self.model.scale))		
		self.update_init_image_to_model()		
		self.strength.set(str(self.model.strength))					