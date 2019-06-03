FOLDER = '../rsc/gsea/c5bp/'
OUTPUT = '../rsc/gsea/c5bp'

###########################################################
###########################################################

import os
import numpy as np

file_list = os.listdir(FOLDER)
d = dict()
for f in file_list:
	with open(os.path.join(FOLDER,f),'r') as ff:
		for line in ff:
			line = line.strip('\n').strip('\r').split('\t')
			name = line[0]
			line.pop(1)
			line.pop(0)
			d[name] = line

np.save(OUTPUT,d)