import os
import shutil

import numpy as np
path=r"D:\项目\image\pCT"
path_list=os.listdir(path)
index_1=np.random.choice(path_list,400,replace=False)
for i in index_1:
    new_path="D:/项目/image_test/pCT/"+i
    old_path=path+"/"+i
    # shutil.move(old_path, new_path)
