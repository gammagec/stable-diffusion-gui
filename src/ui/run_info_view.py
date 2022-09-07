from tkinter import WORD, scrolledtext, Label, Frame, INSERT, END

class RunInfoView:
	name = 'run_info_view'

	def __init__(self, parent, model, app_context):
		self.model = model
		self.frame = Frame(parent)
		label = Label(self.frame, text = "Selected Run")
		label.pack()		
		self.text = scrolledtext.ScrolledText(self.frame, 
			wrap = WORD, width = 40, height = 10, font = ("Times New Roman", 10))
		self.text.configure(state = 'disabled')
		self.text.pack()
		self.frame.pack()
		self.model.update_run_model_subject.register(self, lambda: self.update_run())

	def update_run(self):		
		self.text.configure(state = 'normal')
		self.text.delete(1.0, END)
		model = self.model
		if(model.run != None):
			self.text.insert(INSERT, 
				f'prompt: {model.prompt}\n'
				f'size: {model.width}x{model.height}\n'
				f'ddim_eta: {model.ddim_eta}, ddim_steps: {model.ddim_steps}\n'
				f'n_iter: {model.n_iter}, n_samples: {model.n_samples}\n'
				f'ckpt: {model.ckpt}\n'
				f'config: {model.config}\n'
				f'device: {model.device}, fixed_code: {model.fixed_code}\n'
				f'from_file: {model.from_file}\n'
				f'f: {model.f}, half: {model.half}\n'
				f'init_img: {model.init_img}\n'
				f'n_rows: {model.n_rows}\n'
				f'outdir: {model.outdir}\n'
				f'plms: {model.plms}\n'
				f'precision: {model.precision}\n'
				f'scale: {model.scale}\n'
				f'seed: {model.seed}\n'
				f'session_name: {model.session_name}\n'
				f'skip_grid: {model.skip_grid}, skip_save: {model.skip_save}\n'
				f'small_batch: {model.small_batch}\n'
				f'strength: {model.strength}\n'
			)		
		self.text.configure(state = 'disabled')