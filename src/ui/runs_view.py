from tkinter import Label, Frame, X, END

from src.ui.list_box import create_list_box

class RunsView:
	name = 'runs_view'

	def __init__(self, parent, model):
		self.model = model
		self.frame = Frame(parent)
		label = Label(self.frame, text = "Runs")
		label.pack()
		self.runs_list = create_list_box(self.frame, lambda evt: model.on_run_select(self.get_selected_run()))
		self.runs_list.pack(fill = X)		
		self.frame.pack()
		model.update_runs_subject.register(self, lambda:self.update_runs())

	def update_runs(self):
		self.runs_list.delete(0, END)
		for name in self.model.runs:
			self.runs_list.insert(END, name)
		if (self.runs_list.size() > 0):
			self.runs_list.select_set(0)
			self.runs_list.event_generate("<<ListboxSelect>>")
		else:
			self.model.on_run_select(None)		

	def get_selected_run(self):
		return self.runs_list.get(self.runs_list.curselection()) 	