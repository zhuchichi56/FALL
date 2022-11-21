import os.path
from pathlib import Path
import shutil


from config import INTERMEDIATE_DATA_DIR
#  用来清空生成的中间数据的
if __name__ == "__main__":
    dirs = ['train_intermediate', 'test_intermediate', 'val_intermediate']

    file_path1 = Path(os.path.join(INTERMEDIATE_DATA_DIR, dirs[0], 'processed'))
    file_path2 = Path(os.path.join(INTERMEDIATE_DATA_DIR, dirs[1], 'processed'))
    file_path3 = Path(os.path.join(INTERMEDIATE_DATA_DIR, dirs[2], 'processed'))
    try:
        shutil.rmtree(file_path1)
    except OSError as e:
        print("Error: %s : %s" % (file_path1, e.strerror))
    try:
        shutil.rmtree(file_path2)
    except OSError as e:
        print("Error: %s : %s" % (file_path2, e.strerror))
    try:
        shutil.rmtree(file_path3)
    except OSError as e:
        print("Error: %s : %s" % (file_path3, e.strerror))
