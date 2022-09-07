from tkinter import Frame, Label, Button, X, END, simpledialog

from src.ui.list_box import create_list_box

class SessionsView:
	name = 'session_view'
	
	def __init__(self, parent, model):
		self.model = model
		self.frame = Frame(parent)
		label = Label(self.frame, text = "Sessions")
		label.pack()
		self.sessions_list = create_list_box(self.frame, lambda evt: self.on_session_select(evt))
		self.sessions_list.pack(fill = X)
		new_button = Button(self.frame, text = "+", command = lambda: self.on_new_session())
		new_button.pack()
		delete_button = Button(self.frame, text = "-", command = lambda: self.on_delete_session())
		delete_button.pack()
		self.frame.pack()			
		model.update_sessions_subject.register(self, lambda: self.update_sessions())

	def update_sessions(self):
		print('updating sessions')
		self.sessions_list.delete(0, END)
		for name in self.model.session_names:
			self.sessions_list.insert(END, name)

	def on_session_select(self, evt):
		session = self.sessions_list.get(self.sessions_list.curselection())		
		self.model.set_session(session)

	def on_new_session(self):
		answer = simpledialog.askstring("Input", "New session name?", parent = self.frame)
		self.model.create_session(answer)
		self.sessions_list.select_set(END)
		self.sessions_list.event_generate('<<ListboxSelect>>')

	def on_delete_session(self):
		if (self.model.get_selected_session_image_count() > 0):
			answer = messagebox.askyesno(
				"Confirmation", 
				"There are runs in this session, are you sure you want to delete it?", 
				parent = self.frame)
			if answer:
				self.model.delete_selected_session()
		else:
			self.model.delete_selected_session()