class Subject(object):
	def __init__(self, name):
		self.name = name
		self.subscribers = dict()

	def register(self, who, callback = None):
		self.subscribers[who] = callback

	def unregister(self, who):
		del self.subscribers[who]

	def dispatch(self):		
		for subscriber, callback in self.subscribers.items():
			sub_name = subscriber.name if hasattr(subscriber, 'name') else 'unknown'
			print(f'{self.name} dispatching to {sub_name}')
			callback()	