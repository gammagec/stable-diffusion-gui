from src.model.sessions_model import SessionsModel
from src.model.config import Config
from src.model.selection_model import SelectionModel
from src.model.runs_model import RunsModel
from src.model.images_model import ImagesModel
from src.model.image_model import ImageModel
from src.model.run_model import RunModel
from src.model.input_image_model import InputImageModel
from src.model.params_model import ParamsModel

from src.services.message_service import MessageService

class AppContext:	
	name = 'app_context'

	def __init__(self):
		self.config = Config()
		self.message_service = MessageService()
		self.selection_model = SelectionModel(self)
		self.sessions_model = SessionsModel(self)	
		self.runs_model = RunsModel(self)	
		self.run_model = RunModel(self)		
		self.images_model = ImagesModel(self)		
		self.image_model = ImageModel(self)			
		self.params_model = ParamsModel(self)		
		self.input_image_model = InputImageModel(self)	