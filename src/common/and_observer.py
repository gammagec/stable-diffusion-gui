from src.common.value_subject import ValueSubject

class AndObserver(ValueSubject):
	def __init__(self, name, subject1, subject2):		
		super().__init__(name, subject1.get_value() and subject2.get_value())
		self.subject1 = subject1
		self.subject2 = subject2

		subject1.register(self, lambda value: self.on_change())
		subject2.register(self, lambda value: self.on_change())

	def on_change(self):
		self.set_value(self.subject1.get_value() and self.subject2.get_value())