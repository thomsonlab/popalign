#####################
# PopAlign
# Caltech 2018
#####################
import numpy as np

def KL(mu1, cov1, mu2, cov2):
	k = len(mu1)
	return 0.5*(np.trace(np.linalg.inv(cov2).dot(cov1))+
		(mu2-mu1).T.dot(np.linalg.inv(cov2)).dot(mu2-mu1)-k+
		np.log(np.linalg.det(cov2)/np.linalg.det(cov1)))

def JeffreyDiv(mu1, cov1, mu2, cov2):
	return np.log10(0.5*KL(mu1, cov1, mu2, cov2)+0.5*KL(mu2, cov2, mu1, cov1))

def align_components(gmmtest, gmmref):
	ntest = gmmtest.n_components
	nref = gmmref.n_components
	arr = np.zeros((ntest,3))

	for i in range(ntest):
		scores = [JeffreyDiv(gmmtest.means_[i],
					gmmtest.covariances_[i],
					gmmref.means_[j],
					gmmref.covariances_[j]) for j in range(nref)]
		imax = np.argmin(scores)
		arr[i,:] = [i, imax, scores[imax]]
	return arr