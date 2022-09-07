from src.common.namespace import Namespace

from src.model.app_context import AppContext
from src.ui.main_window import MainWindow

def main():
	app_context = AppContext()
	app_context.config.load()
	app_context.sessions_model.load()
	main_window = MainWindow(app_context)
	main_window.start()

if __name__ == "__main__":
	main()