from src.common.value_subject import ValueSubject

class NotObserver(ValueSubject):
	def __init__(self, name, subject):		
		super().__init__(name, not subject.get_value())
		self.subject = subject

		subject.register(self, lambda value: self.on_change())

	def on_change(self):
		self.set_value(not self.subject.get_value())