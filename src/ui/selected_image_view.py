from tkinter import Label, Frame, Canvas, Button, NW, LEFT

from src.ui.image_view import ImageView

class SelectedImageView(ImageView):
	name = 'selected_image_view'

	def __init__(self, parent, model):
		super().__init__(parent, model, 'Selected Image')
		self.model = model

	def create(self):
		super().create()
		use_image = Button(super().get_button_frame(), 
			text = "Use Image", command = lambda: self.model.use_image())
		use_image.pack(side = LEFT)
		copy_seed = Button(super().get_button_frame(), 
			text = "Copy Seed", command = lambda: self.model.copy_seed())
		copy_seed.pack(side = LEFT)							