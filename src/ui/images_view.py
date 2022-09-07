from tkinter import Frame, Label, END, X

from src.ui.list_box import create_list_box

class ImagesView:
	name = 'images_view'

	def __init__(self, parent, model, app_context):
		self.model = model
		self.frame = Frame(parent)
		label = Label(self.frame, text = "Images")
		label.pack()		
		self.images_list = create_list_box(self.frame, lambda evt: self.on_image_select(evt))
		self.images_list.pack(fill = X)		
		self.frame.pack()
		model.update_images_subject.register(self, lambda: self.update_images())

	def update_images(self):
		print('updating images')
		self.images_list.delete(0, END)
		for name in self.model.images:
			self.images_list.insert(END, name)		
		if (self.images_list.size() > 0):
			self.images_list.select_set(0)
			self.images_list.event_generate("<<ListboxSelect>>")
		else:
			self.model.set_image(None)		

	def on_image_select(self, evt):
		image = self.images_list.get(self.images_list.curselection())
		self.model.set_image(image)