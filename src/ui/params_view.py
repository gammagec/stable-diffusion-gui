from tkinter import WORD, LEFT, Frame, Entry, Label, scrolledtext, Checkbutton
from tkinter import LEFT, Button, END, DISABLED, NORMAL, W, E

from src.common.bindings import bind_enabled_to_value_observer, bind_scrolledtext_to_stringvar
from src.common.bindings import bind_enabled_to_intvar

from src.ui.view import View

class ParamsView(View):
	name = 'params_view'

	def __init__(self, parent, model, view_model):
		super().__init__(parent)
		self.model = model		
		self.view_model = view_model

	def create(self):
		super().create()
		frame = super().get_frame()
		label = Label(frame, text = "Params")
		label.pack()	

		grid = Frame(frame)
		row = 0

		Label(grid, text = "Prompt:").grid(row = row, column = 0, columnspan = 3, sticky = W)
		row += 1

		prompt = scrolledtext.ScrolledText(
			grid, wrap = WORD, width = 40, height = 5, font = ("Times New Roman", 10))		
		bind_scrolledtext_to_stringvar(prompt, self.view_model.prompt)		
		prompt.grid(row = row, column = 0, columnspan = 3)
		row += 1

		Checkbutton(grid, 
			text = 'Use Image Prompt', variable = self.view_model.use_image_prompt, 
			onvalue = 1, offvalue = 0,
			command = lambda: self.view_model.use_image_prompt_checked()
			).grid(row = row, column = 0, columnspan = 3, sticky = W)
		row += 1

		image_prompt_label = Label(grid, text = 'Image Prompt:'
			).grid(row = row, column = 0, sticky = E)
		image_prompt_entry = Entry(grid, textvariable = self.view_model.image_prompt,
			state = DISABLED)
		image_prompt_entry.grid(row = row, column = 1, columnspan = 2)
		bind_enabled_to_intvar(image_prompt_entry, self.view_model.use_image_prompt)		
		row += 1
		
		strength_label = Label(grid, text = 'Strength:'
			).grid(row = row, column = 0, sticky = E)
		strength_entry = Entry(grid, textvariable = self.view_model.strength,
			state = DISABLED)
		strength_entry.grid(row = row, column = 1)
		bind_enabled_to_intvar(strength_entry, self.view_model.use_image_prompt)		
		strength_button = Button(grid, text = 'Show', 
			command = lambda: self.view_model.show_clicked(),
			state = DISABLED)
		strength_button.grid(row = row, column = 2)
		bind_enabled_to_intvar(strength_button, self.view_model.use_image_prompt)		
		row += 1		

		Label(grid, text = "Seed:").grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.seed).grid(row = row, column = 1)		
		Button(grid, 
			text = 'Random Seed',
			command = lambda: self.view_model.set_random_seed()
			).grid(row = row, column = 2)		
		row += 1

		Label(grid, text = 'DDIM Steps:').grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.ddim_steps).grid(row = row, column = 1)
		row += 1

		Label(grid, text = '# Samples:').grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.n_samples).grid(row = row, column = 1)
		load_model_button = Button(grid, text = 'Load Model',
			command = lambda: self.view_model.load_model_clicked())
		load_model_button.grid(row = row, column = 2)
		bind_enabled_to_intvar(load_model_button, self.view_model.enable_load_model_button)		
		row += 1

		Label(grid, text = '# Iterations:').grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.n_iter).grid(row = row, column = 1)
		row += 1

		Label(grid, text = 'Width:').grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.width).grid(row = row, column = 1)
		run_button = Button(grid, text = 'Run', 
			command = lambda: self.view_model.run_clicked(),
			state = DISABLED)	
		run_button.grid(row = row, column = 2)
		bind_enabled_to_value_observer(run_button, self.view_model.run_enabled, run_button)	
		row += 1

		Label(grid, text = 'Height:').grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.height).grid(row = row, column = 1)
		run_random_button = Button(grid, text = 'Run Random Seed', 
			command = lambda: self.view_model.run_random_clicked(),
			state = DISABLED)
		run_random_button.grid(row = row, column = 2)					
		bind_enabled_to_value_observer(run_random_button, 
			self.view_model.run_enabled, 
			run_random_button)			
		row += 1

		Label(grid, text = 'Channels:').grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.channels).grid(row = row, column = 1)
		row += 1

		Label(grid, text = 'Downsampling:').grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.downsampling).grid(row = row, column = 1)
		row += 1

		Label(grid, text = 'Scale:').grid(row = row, column = 0, sticky = E)
		Entry(grid, textvariable = self.view_model.scale).grid(row = row, column = 1)
		row += 1				

		grid.pack()

		
	
