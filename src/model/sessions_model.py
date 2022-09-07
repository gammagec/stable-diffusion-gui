import os
import shutil

from src.common.subject import Subject
from src.common.value_subject import ValueSubject

class SessionsModel:
	name = 'sessions_model'

	def __init__(self, app_context):								
		self.config = app_context.config
		self.message_service = app_context.message_service
		self.session_names = ValueSubject('session-names', [])
		self.sessions_path = ''				
		self.selection_model = app_context.selection_model

	def load(self):
		print(f'loading sessions from {self.config.out_dir}')
		session_names = []		
		self.sessions_path = os.path.join(self.config.out_dir, "sessions")				
		if os.path.exists(self.sessions_path) and os.path.isdir(self.sessions_path):
			for f in os.scandir(self.sessions_path):
				if f.is_dir():	
					session_names.append(f.name)
		self.session_names.set_value(session_names)

	def set_session(self, session):
		print(f'set session: {session}')		
		self.selection_model.set_selected_session(session)

	def create_session(self, name):		
		session_path = os.path.join(self.sessions_path, name)
		if os.path.exists(session_path):
			self.message_service.error(f'session {name} already exists')
			return
		os.makedirs(session_path, exist_ok = False)
		session_names = self.session_names.get_value()
		session_names.append(name)
		self.session_names.set_value(session_names)

	def get_selected_session_image_count(self):
		name = self.selection_model.selected_session.get_value()
		path = os.path.join(self.sessions_path, name)
		count = sum(1 for _ in os.scandir(path))		
		return count

	def delete_selected_session(self):				
		name = self.selection_model.selected_session.get_value()
		path = os.path.join(self.sessions_path, name)
		print(f'deleting session {path}')
		shutil.rmtree(path)
		self.load()
		self.selection_model.set_selected_session(None)	