import os
from PIL import Image, ImageTk
from io import BytesIO
import win32clipboard
from src.common.subject import Subject
from src.common.value_subject import ValueSubject

class ImageModel(object):
	name = 'image_model'

	def __init__(self, app_context):
		self.selection_model = app_context.selection_model
		self.image_loaded = ValueSubject('image_loaded', False)
		self.images_model = app_context.images_model
		self.image = None
		self.copy_seed_value = Subject('copy-seed-value')
		self.use_image_value = Subject('use-image-value')

		self.selection_model.image_selected.register(self, lambda: self.load_image())

	def open(self):
		path = self.get_image_path()
		if (path != None):						
			print(f'opening image {path}')
			os.startfile(path)

	def get_image_path(self):
		img_name = self.selection_model.selected_image
		if not img_name:
			return None
		path = os.path.join(self.images_model.images_path, img_name)
		return path

	def copy_seed(self):		
		self.copy_seed_value.dispatch()

	def use_image(self):
		self.use_image_value.dispatch()
	
	def copy(self):
		path = self.get_image_path()		
		if (path != None):						
			print(f'copying image {path}')
			image = Image.open(path)
			output = BytesIO()
			image.convert("RGB").save(output, "BMP")
			data = output.getvalue()[14:]
			output.close()
			win32clipboard.OpenClipboard()
			win32clipboard.EmptyClipboard()
			win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
			win32clipboard.CloseClipboard()

	def load_image(self):
		path = self.get_image_path()
		print(f'load image for {path}')
		if (path == None):
			self.image = None
			self.image_loaded.set_value(False)
		else:			
			print(f'loading image {path}')
			self.image = ImageTk.PhotoImage(file = path)
			self.image_loaded.set_value(True)