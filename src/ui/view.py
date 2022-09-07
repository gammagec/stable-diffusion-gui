from tkinter import Frame

class View(object):
	name = 'image_view'

	def __init__(self, parent):
		self.parent = parent		

	def create(self):
		self.frame = Frame(self.parent)
		self.frame.pack()

	def get_frame(self):
		return self.frame