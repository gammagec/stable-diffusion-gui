import os
from src.common.subject import Subject

class ImagesModel:
	name = 'images_model'

	def __init__(self, app_context):
		self.images = []
		self.image = ''
		self.runs_model = app_context.runs_model
		self.images_path = ''
		self.message_service = app_context.message_service
		self.update_images_subject = Subject('update images')
		self.selection_model = app_context.selection_model

		self.selection_model.run_selected.register(self, lambda: self.on_run_selected())

	def set_image(self, image):		
		self.selection_model.set_selected_image(image)

	def on_run_selected(self):
		print('images model for run')
		run = self.selection_model.selected_run		
		self.images = []
		if (run != None):
			self.images_path = os.path.join(self.runs_model.session_path, run)
			if os.path.exists(self.images_path) and os.path.isdir(self.images_path):
				for f in os.scandir(self.images_path):
					if not f.is_dir() and f.name.endswith('.png'):	
						self.images.append(f.name)
		print(f'loaded images for {run}')
		self.update_images_subject.dispatch()	