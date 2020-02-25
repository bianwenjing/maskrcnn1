from PIL import Image
import numpy as np

depth_i = Image.open('/home/wenjing/storage/ScanNetv2/test_depth/scene0070_00/depth/1.png')  # (1296,968)
depth_i = np.array(depth_i)
print(np.unique(depth_i))