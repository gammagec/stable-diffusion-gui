from tkinter import Label, Frame, Canvas, Button, NW, LEFT, DISABLED, NORMAL

from src.ui.view import View

class ImageView(View):
	name = 'image_view'

	def __init__(self, parent, model, title):
		super().__init__(parent)
		self.title = title
		self.model = model	

		self.model.image_loaded.register(self, lambda loaded: self.on_image_loaded(loaded))

	def create(self):
		super().create()
		frame = super().get_frame()
		label = Label(frame, text = self.title)
		label.pack()
		self.canvas = Canvas(frame, width = 512, height = 512)
		self.canvas.pack()
		self.b_frame = Frame(frame)
		self.open_button = Button(
			self.b_frame, 
			text = "Open", 
			state = DISABLED,
			command = lambda: self.model.open())
		self.open_button.pack(side = LEFT)
		self.copy_button = Button(
			self.b_frame, 
			text = "Copy Image",
			state = DISABLED,
			command = lambda: self.model.copy())
		self.copy_button.pack(side = LEFT)		
		self.b_frame.pack()					

	def get_button_frame(self):
		return self.b_frame

	def on_image_loaded(self, loaded):
		if (loaded):
			self.canvas.create_image(0, 0, anchor = NW, image = self.model.image)
			self.open_button['state'] = NORMAL
			self.copy_button['state'] = NORMAL
			print('image loaded')
		else:
			self.canvas.delete('all')
			self.open_button['state'] = DISABLED
			self.copy_button['state'] = DISABLED
			