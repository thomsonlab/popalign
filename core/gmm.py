#####################
# PopAlign
# Caltech 2019
#####################

from sklearn import mixture as smix
from multiprocessing import Pool
import numpy as np

def build_single_gmm(C, *args,**kwargs):
	np.random.seed()
	gmm = smix.GaussianMixture(*args,**kwargs)
	return gmm.fit(C)

def build_repeat(C, range_, nreps=1, ncores=None):
	# set training and validation sets
	m, n = C.shape
	idx = np.random.choice(np.arange(m), int(m*0.2), replace=False)
	neg = np.setdiff1d(np.arange(m), idx)
	Ctrain = C[idx,:]
	Cvalid = C[neg,:]

	# fit GMMs in parallel
	ks = np.repeat(range_,nreps)
	with Pool(processes=ncores) as p:
		q = p.starmap(build_single_gmm, [(Ctrain,k) for k in ks])

	# score each model with the validation set
	scores = [gmm.score(Cvalid) for gmm in q]

	return q, scores
	#return q[np.argmax(scores)]