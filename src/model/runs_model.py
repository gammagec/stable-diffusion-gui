import os
from src.common.subject import Subject

class RunsModel:
	name = 'runs_model'	

	def __init__(self, app_context):					
		self.runs = []
		self.sessions_model = app_context.sessions_model
		self.message_service = app_context.message_service		
		self.update_runs_subject = Subject('update_runs')
		self.session_path = ''		
		self.selection_model = app_context.selection_model
		self.selection_model.selected_session.register(self, 
			lambda session: self.selected_session_updated(session))
		
	def selected_session_updated(self, session):
		self.runs = []		
		if session != None:
			self.session_path = os.path.join(self.sessions_model.sessions_path, session)			
			if os.path.exists(self.session_path) and os.path.isdir(self.session_path):			
				for f in os.scandir(self.session_path):
					if f.is_dir():	
						self.runs.append(f.name)
			self.runs.sort(key = int, reverse = True)
			print(f'loaded runs for {session}')
		self.update_runs_subject.dispatch()	

	def on_run_select(self, run):			
		print(f'run selected {run}')
		self.selection_model.set_selected_run(run)		

	def after_new_run(self):
		self.selected_session_updated(self.selection_model.selected_session.get_value())