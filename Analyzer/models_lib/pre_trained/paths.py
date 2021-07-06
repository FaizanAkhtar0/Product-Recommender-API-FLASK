import os
import glob


def return_path():
    return glob.glob(os.path.dirname(__file__) + "/*.save")


