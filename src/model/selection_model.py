from src.common.subject import Subject
from src.common.value_subject import ValueSubject

class SelectionModel:
	name = 'selection_model'

	def __init__(self, app_context):
		self.selected_session = ValueSubject('selected_session', None)
		self.selected_run = None
		self.selected_image = None
		self.run_selected = Subject('run_selected')
		self.image_selected = Subject('image_selected')

	def set_selected_run(self, run):
		print(f'run set {run}')
		self.selected_run = run
		self.run_selected.dispatch()	

	def set_selected_session(self, session):
		print(f'session set {session}')		
		self.selected_session.set_value(session)

	def set_selected_image(self, image):
		print(f'image set {image}')
		self.selected_image = image
		self.image_selected.dispatch()