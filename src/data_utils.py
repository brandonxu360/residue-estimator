import os
from typing import List
from config import DATA_DIR

def get_jpg_filenames(root: str) -> List[str]:
    """
    Walk the file tree given the root and return the list of jpg files.
    """
    jpgs = []
    for _, _, filenames in os.walk("."):
        for filename in filenames:
            if filename.endswith(".jpg"):
                print(filename)
                jpgs.append(filename)
        
        return jpgs

print(get_jpg_filenames(DATA_DIR))



    
