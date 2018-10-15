import inspect
import os

ROOT_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/'

DATA_PATH = ROOT_PATH + 'data/'
MODEL_PATH = DATA_PATH + 'saved_models/'
EXPORT_PATH = DATA_PATH + 'exports/'
OBJ_FILES_PATH = DATA_PATH + 'saved_objs/'

for folder_path in [DATA_PATH, MODEL_PATH, EXPORT_PATH, OBJ_FILES_PATH]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
