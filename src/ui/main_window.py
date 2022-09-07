import tkinter as tk
from tkinter import Tk, Frame, LEFT, X

from src.ui.sessions_view import SessionsView
from src.ui.runs_view import RunsView
from src.ui.run_info_view import RunInfoView
from src.ui.images_view import ImagesView
from src.ui.image_view import ImageView
from src.ui.params_view import ParamsView
from src.ui.selected_image_view import SelectedImageView
from src.view_model.params_view_model import ParamsViewModel
from src.view_model.sessions_view_model import SessionsViewModel

class MainLayout:
	name = 'main_layout'

	def __init__(self, parent):
		self.frame = Frame(parent)
		self.left = Frame(self.frame)
		self.left.pack(side = LEFT)
		self.middle = Frame(self.frame)
		self.middle.pack(side = LEFT)
		self.right = Frame(self.frame)
		self.right.pack(side = LEFT)
		self.frame.pack(fill = X)

class MainWindow:
	name = 'main_window'
	
	def __init__(self, app_context):
		self.root = tk.Tk()
		self.root.title('Stable Diffusion GUI')
		self.root.geometry("1324x1200")
		layout = MainLayout(self.root)
		
		sessions_view = SessionsView(layout.left, 
			SessionsViewModel(app_context.sessions_model, app_context))
		sessions_view.create()

		runs_view = RunsView(layout.left, app_context.runs_model)
		run_info_view = RunInfoView(layout.left, app_context.run_model, app_context)
		images_view = ImagesView(layout.left, app_context.images_model, app_context)

		input_image_view = ImageView(layout.middle, app_context.input_image_model, 'Input Image')		
		input_image_view.create()

		image_view = SelectedImageView(layout.middle, app_context.image_model)		
		image_view.create()

		params_view = ParamsView(layout.right,
			ParamsViewModel(app_context.params_model, app_context))		
		params_view.create()

	def start(self):
		self.root.mainloop()