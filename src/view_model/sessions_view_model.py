
from tkinter import StringVar, simpledialog

class SessionsViewModel(object):

	def __init__(self, model, app_context):
		self.model = model
		self.list_items = StringVar()

		model.session_names.register(self, lambda val: self.list_items.set(val))		
		self.list_items.set(model.session_names.get_value())

	def set_view(self, view):
		self.view = view

	def on_session_clicked(self, name):				
		self.model.set_session(name)

	def on_new_session_clicked(self):
		answer = self.view.get_new_session_name()
		self.model.create_session(answer)
		self.view.select_last()		

	def on_delete_session_clicked(self):
		if (self.model.get_selected_session_image_count() > 0):
			answer = self.view.confirm_session_delete()			
			if answer:
				self.model.delete_selected_session()
				self.view.clear_selection()
		else:
			self.model.delete_selected_session()