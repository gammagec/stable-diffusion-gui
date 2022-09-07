from tkinter import Listbox, SINGLE, END

def create_list_box(parent, select_command):
	listbox = Listbox(parent, selectmode = SINGLE, exportselection = 0)
	listbox.bind('<<ListboxSelect>>', select_command)

	def on_arrow_up(evt):
		index = listbox.curselection()[0] - 1
		if 0 <= index < evt.widget.size():
			listbox.selection_clear(0, END)
			listbox.select_set(index)
			select_command(evt)

	def on_arrow_down(evt):
		index = listbox.curselection()[0] + 1
		if 0 <= index < evt.widget.size():
			listbox.selection_clear(0, END)
			listbox.select_set(index)
			select_command(evt)

	listbox.bind('<Down>', on_arrow_down)
	listbox.bind('<Up>', on_arrow_up)
	return listbox