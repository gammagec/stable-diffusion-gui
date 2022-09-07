from tkinter import Frame, Label, Button, X, END, simpledialog, messagebox, LEFT

from src.ui.list_box import create_list_box
from src.ui.view import View

class SessionsView(View):
	name = 'session_view'
	
	def __init__(self, parent, view_model):
		super().__init__(parent)
		self.view_model = view_model	
		view_model.set_view(self)			

	def create(self):
		super().create()
		frame = super().get_frame()
		label = Label(frame, text = "Sessions")
		label.pack()
		self.sessions_list = create_list_box(frame, 			
			lambda evt: self.view_model.on_session_clicked(self.sessions_list.get(self.sessions_list.curselection())),
			listvariable = self.view_model.list_items)
		self.sessions_list.pack(fill = X)

		button_frame = Frame(frame)
		new_button = Button(button_frame, text = "+", 
			command = lambda: self.view_model.on_new_session_clicked())
		new_button.pack(side = LEFT)
		delete_button = Button(button_frame, text = "-", 
			command = lambda: self.view_model.on_delete_session_clicked())
		delete_button.pack(side = LEFT)					
		button_frame.pack()

	def get_new_session_name(self):
		return simpledialog.askstring("Input", "New session name?", parent = self.frame)

	def confirm_session_delete(self):
		return messagebox.askyesno(
				"Confirmation", 
				"There are runs in this session, are you sure you want to delete it?", 
				parent = self.frame)

	def select_last(self):
		self.clear_selection()
		self.sessions_list.selection_set(END)
		self.sessions_list.event_generate('<<ListboxSelect>>')

	def clear_selection(self):
		self.sessions_list.selection_clear(0, END)
		self.sessions_list.event_generate('<<ListboxSelect>>')