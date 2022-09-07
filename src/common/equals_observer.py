from src.common.value_subject import ValueSubject

class EqualsObserver(ValueSubject):
	def __init__(self, name, subject, expect):		
		super().__init__(name, subject.get_value() == expect)
		self.subject = subject
		self.expect = expect

		subject.register(self, lambda value: self.on_change())

	def on_change(self):
		self.set_value(self.subject.get_value() == self.expect)