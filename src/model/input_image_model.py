from PIL import ImageTk

from src.common.subject import Subject
from src.model.image_model import ImageModel

class InputImageModel(ImageModel):
	name = 'input_image_model'

	def __init__(self, app_context):
		super().__init__(app_context)		
		self.params_model = app_context.params_model
		self.params_model.init_image.register(self, lambda val: self.load_image())

	def get_image_path(self):
		return self.params_model.init_image.get_value()