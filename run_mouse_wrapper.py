
import os, sys
from matching import *

_, datapath, dataset, mouse, cpus = sys.argv

mousePath = os.path.join(datapath,dataset,mouse)

paths = [os.path.join(mousePath,sessionPath,'OnACID_results.hdf5') for sessionPath in os.listdir(mousePath) if 'Session' in sessionPath]
paths.sort()
print('mousePath:',mousePath)
print('paths:',paths)

match = matching(paths,mousePath)
match.run_matching()