from src.common.subject import Subject

class ValueSubject(Subject):
	def __init__(self, name, initial_value):		
		super().__init__(name)
		self.value = initial_value

	def set_value(self, value):
		self.value = value
		self.dispatch()

	def get_value(self):
		return self.value

	def dispatch(self):
		for subscriber, callback in self.subscribers.items():
			sub_name =  subscriber.name if hasattr(subscriber, 'name') else 'unknown'
			print(f'{self.name} dispatching to {sub_name} with value {self.value}')
			callback(self.value)	