# Paul Rivaud
# Caltech
# paulrivaud.info@gmail.com

import os
import csv
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import io as sio
from scipy import sparse as ss
from scipy import optimize as so
from scipy import stats
from scipy.stats import linregress
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import sklearn.mixture as smix
from sklearn import cluster as sc
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
import adjustText
import ipywidgets as widgets
from ipywidgets import interact, interactive, Layout, HBox, VBox, fixed
from textwrap import wrap
import plotly
from plotly.graph_objs import graph_objs as go
from plotly.offline import iplot

'''
Misc functions
'''
def mkdir(str):
	'''
	Create directories recursively from `str`

	Parameters
	----------
	str : str
		Path to an output folder
	'''
	try:
		os.makedirs(str)
	except:
		pass

def cat_data(pop, name):
	'''
	Aggregates data from each sample in `pop` given an attribute `name`

	Parameters
	----------
	pop : dict
		Popalign object
	name : str
		Name of the attribute. Can be M, M_norm or C
	'''
	if name not in ['M', 'M_norm', 'C']:
		raise Exception('name must be one of M, M_norm or C')
	if name in ['M', 'M_norm']:
		tmp = ss.hstack([pop['samples'][x][name] for x in pop['order']])
	elif name in ['C']:
		tmp = np.vstack([pop['samples'][x][name] for x in pop['order']])
	return tmp

def show_samples(pop):
	'''
	Print the names of the samples loaded in `pop`

	Parameters
	----------
	pop : dict
		Popalign object
	'''
	print('Loaded samples are:')
	for x in pop['samples']:
		print(x)

def nr_nc(n):
	'''
	Compute the optimum number of rows and columns to plot `n` plots in a grid

	Parameters
	----------
	n : int
		Number of items from which to derive numbers of rows and columns
	'''
	sr = np.sqrt(n)
	nr = int(np.ceil(sr))
	nc = int(np.rint(sr))
	return nr, nc

def intraclass_var(X,t):
	'''
	Compute the intraclass variance after splitting `X` into two based on threshold value `t`

	Parameters
	----------
	X : array
		Array of values to split
	t : float
		Threshold value
	'''
	N = len(X)
	# idxi is an array of indices of elements belonging to class i
	idx1 = np.where(X<=t)[0]
	idx2 = np.where(X>t)[0]

	# wi is the weight of class i
	w1 = len(idx1)/N
	w2 = len(idx2)/N

	#vari is the variance of class i
	var1 = np.var(X[idx1])
	var2 = np.var(X[idx2])

	# return weighted intra class variance
	return w1*var1 + w2*var2

def otsu(X, nbins=50):
	'''
	Implementation of Otsu's method
	Find optimal threshold to split
	a vector `X` into two classes

	Parameters
	----------
	X : array
		Array of values to split
	nbins : int
		Number of bins. Will define the threshold values to try
	'''
	N = len(X)
	thresholds = np.linspace(min(X), max(X), nbins+1)
	
	# compute the intraclass variance for each possible threshold in parallel
	with Pool(None) as p:
		q = p.starmap(intraclass_var, [(X,t) for t in thresholds[:-1]])
		
	# pick threshold that minimizes the intraclass variance
	return thresholds[np.argmin(q)]

'''
Load functions
'''
def load_genes(genes):
	'''
	Load the gene names from a file

	Parameters
	----------
	genes : str
		Path to a .tsv 10X gene file
	'''
	return np.array([row[1].upper() for row in csv.reader(open(genes), delimiter="\t")])

def load_samples(samples, genes, outputfolder='output'):
	'''
	Load data from a dictionary and gene labels from a file

	Parameters
	----------
	samples : dict
		Dictionary of sample names (keys) and paths to their respective matrix files (values)
	genes : str
		Path to a .tsv 10X gene file
	outputfolder : str
		Path (or name) of the output folder to create
	'''
	obj = {}
	obj['samples'] = {}
	order = []
	for x in samples:
		x = str(x)
		# create entry and load sparse matrix
		obj['samples'][x] = {}
		obj['samples'][x]['M'] = sio.mmread(samples[x]).tocsc()
		order.append(x)
	# save list of sample names to always call them in the same order for consistency
	obj['order'] = order
	obj['nsamples'] = len(order)
	# define name of output folder to save results
	obj['output'] = outputfolder
	# load and store genes
	obj['genes'] = load_genes(genes)
	return obj

def check_cols(s,cols):
	'''
	Check that column label `s` is in list `cols`
	
	Parameters
	----------
	s : str
		Name to check for
	cols : list
		List of strings
	'''
	if s not in cols:
		raise Exception('Columns of meta data must include: %s' % s)

def load_screen(matrix, barcodes, metafile, genes, outputfolder='output'):
	'''
	Load data from a screen experiment and genes from a file

	Parameters
	----------
	matrix : str
		Path to a sparse matrix
	barcodes : str
		Path to a .tsv 10X barcodes file
	metafile : str
		Path to a metadata file. Must contains `cell_barcodes` and `sample_id` columns
	outputfolder : str
		Path (or name) of the output folder to create
	'''
	obj = {}
	obj['samples'] = {}

	# load main matrix
	M = sio.mmread(matrix).tocsc()

	# load associated barcodes
	obj['barcodes'] = np.array([row[0] for row in csv.reader(open(barcodes), delimiter="\t")])
	# store index of each barcode in a dictionary to quickly retrieve indices for a list of barcodes
	obj['bc_idx'] = {}
	for i, bc in enumerate(obj['barcodes']):
		obj['bc_idx'][bc] = i

	# load metadata file
	obj['meta'] = pd.read_csv(metafile, header=0)
	cols = obj['meta'].columns.values
	check_cols('cell_barcode', cols)
	check_cols('sample_id', cols)

	# if additional columns exist in the metadata file, save those under `conditions`
	if len(obj['meta'].columns.values)>2:
		tmp = obj['meta'].drop_duplicates(subset='sample_id')
		tmp.index = tmp.sample_id.values
		tmp = tmp.drop(columns=['cell_barcode','sample_id'])
		obj['conditions'] = tmp
	
	order = []
	# go through the sample_id values to split the data and store it for each individual sample
	for i in obj['meta']['sample_id'].dropna().unique():
		x = str(i)
		obj['samples'][x] = {}
		# get the cell barcodes for sample defined by sample_id
		sample_bcs = obj['meta'][obj['meta'].sample_id == i].cell_barcode.values
		# retrieve list of matching indices
		idx = [obj['bc_idx'][bc] for bc in sample_bcs]
		# extract matching data from M
		obj['samples'][x]['M'] = M[:,idx]
		order.append(x)

	# save list of sample names to always call them in the same order for consistency
	obj['order'] = order
	obj['nsamples'] = len(order)
	obj['output'] = outputfolder
	obj['genes'] = load_genes(genes)
	return obj

'''
Normalization and filtering functions
'''
def col_norm(M):
	'''
	Perform column normalization on matrix `M`

	Parameters
	----------
	M : sparse matrix
		matrix to be column normalized
	'''
	sums = np.array(M.sum(axis=0)).flatten() # compute sums of all columns (cells)
	M.data = M.data.astype(float) # convert type from int to float prior to division
	
	for i in range(len(M.indptr)-1): # for each column i
		rr = range(M.indptr[i], M.indptr[i+1]) # get range rr
		M.data[rr] = M.data[rr]/sums[i] # divide data values by matching column sum

def factor(M, normfactor):
	'''
	Multiply the values of `M` by `normfactor`

	Parameters
	----------
	M : sparse matrix
		matrix to multiply
	normfactor : int
		normalization factor to use	
	'''
	return M.data*normfactor

def comparison_factor(tmp, f, ogmean):
	'''
	Compare the original mean to the mean of the data multiplied by a factor `f`

	Parameters
	----------
	tmp : sparse matrix
		Matrix to try a normalization factor
	f : int
		Normalization factor to try
	ogmean : float
		Original mean of the aggregated data
	'''
	tmp.data = tmp.data*f # multiply matrix values by factor
	return np.abs(tmp.mean()-ogmean) # return the absolute value of the difference between the new mean and the original one

def norm_factor(pop):
	'''
	Find a normalization factor that minimizes the difference between the original mean of the data and the mean after column normalization and factorization

	Parameters
	----------
	pop : dict
		Popalign object
	'''
	M = cat_data(pop, 'M') # Aggregatre data from all samples

	ogmean = pop['original_mean'] # Retrive original mean of the data prior to normalization
	factorlist = [1,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000] # list of factors to try

	with Pool(None) as p:
		q = p.starmap(comparison_factor, [(M.copy(), f, ogmean) for f in factorlist]) # try different factors
	normfactor = factorlist[np.argmin(q)] # pick the factor that minimizes the difference between the new mean and the original one 

	with Pool(None) as p:
		q = p.starmap(factor, [(pop['samples'][x]['M'], normfactor) for x in pop['order']]) #Multiply data values by picked factor in parallel
	for i,x in enumerate(pop['order']):
		pop['samples'][x]['M'].data = q[i] # update data values for each sample

def normalize(pop):
	'''
	Normalize the samples of object `pop` and applies a normalization factor

	Parameters
	----------
	pop : dict
		Popalign object
	'''
	if 'normed' in pop:
		print('Data already column normalized')
	else:
		M = cat_data(pop, 'M') # aggregate data
		pop['original_mean'] = M.mean() # store original mean of the data

		print('Performing column normalization')
		for x in pop['order']:
			col_norm(pop['samples'][x]['M']) #column normalize data of sample `x`
		
		print('Finding best normalization factor')
		norm_factor(pop) # Apply normalization factor
		pop['normed'] = True

def mu_sigma(M, pop):
	'''
	Compute mu and sigma values of each gene and return the logged values

	Parameters
	----------
	M : sparse matrix
		Matrix for which genes mu and sigma values will be computed
	pop : dict
		Popalign object
	'''
	mean, var = mean_variance_axis(M, axis=1) # get genes means and variances from M
	std = np.sqrt(var) # compute standard deviations from variances
	cv = np.divide(std, mean, out=np.zeros_like(std), where=mean!=0) # compute coefficient of variation

	# indices of genes that are present in more than .1% of the cells
	MCSR = M.tocsr() # MCSR is the row oriented version of M
	presence = np.array([MCSR.indptr[i+1]-MCSR.indptr[i] for i in range(M.shape[0])]) # count how many cells have non zeros expression for each gene
	presence_idx = np.where(presence>M.shape[1]*0.001)[0] # get indices of genes that are expressed in more than .1% of the cells
	MCSR = None

	nzidx = np.nonzero(mean)[0] # indices of genes with non-zero mean
	nzidx = np.intersect1d(nzidx, presence_idx) # get intersection of genes with non-zero mean and genes present in more than .1% of the cells

	nzcv = cv[nzidx] # select the matching cvs
	nzmean = mean[nzidx] # select the matching means
	lognzcv = np.log10(nzcv) # log10
	lognzmean = np.log10(nzmean) # log10

	pop['nzidx'] = nzidx # store index for future gene selection and filtering
	return lognzcv, lognzmean

def filter(pop):
	'''
	Filter genes from data in `pop`. Discards Ribosomal genes that start with RPS or RPL

	Parameters
	----------
	pop :dict
		Popalign object
	'''
	gene_idx = pop['filter_idx'] # get indices of genes to keep
	genes = pop['genes'] # get all genes names
	tmp = []
	print('Removing ribosomal genes')
	for i in gene_idx:
		g = genes[i]
		if g.startswith('RPS') or g.startswith('RPL'):
			pass
		else:
			tmp.append(i) # only append gene index if gene name doesn't star with RPS or RPL
	gene_idx = np.array(tmp)

	print('Filtering genes ang logging data')
	for x in pop['order']: # for each sample x
		M_norm = pop['samples'][x]['M'][gene_idx,:] # filter genes
		M_norm.data = np.log(M_norm.data+1) # log the data
		pop['samples'][x]['M_norm'] = M_norm # store the filtered logged data
		pop['samples'][x]['M'].data = np.log(pop['samples'][x]['M'].data+1) # log the non gene-filtered matrix 

	filtered_genes = [genes[i] for i in gene_idx] # get gene names of filtered genes
	pop['filtered_genes'] = filtered_genes # store name list
	pop['filtered_genes_set'] = set(filtered_genes)

def plot_mean_cv(pop, offset):
	'''
	Generate a gene filtering plot

	Parameters
	----------
	pop : dict
		Popalign object
	offset : float
		Offset value to slide filtering line
	'''
	plt.rcParams['figure.figsize'] = [8, 6]

	lognzcv = pop['genefiltering']['lognzcv'] # get cv values
	lognzmean = pop['genefiltering']['lognzmean'] # get mean values
	slope = pop['genefiltering']['slope'] # get line slope
	intercept = pop['genefiltering']['intercept'] # get line intercept

	adjusted_intercept = intercept+np.log10(offset) # slide filtering line with offset
	selection_idx = np.where(lognzcv>lognzmean*slope+adjusted_intercept)[0] # get indices of genes above the filtering line
	print('%d genes selected' % len(selection_idx))

	plt.scatter(lognzmean, lognzcv, s=1) # plot all genes
	plt.scatter(lognzmean[selection_idx], lognzcv[selection_idx], c='maroon', s=1) # plot genes above line with different color
	plt.plot(lognzmean,lognzmean*slope+adjusted_intercept, c='darkorange') # plot filtering line
	plt.xlabel('log10(mean)')
	plt.ylabel('log10(cv)')
	plt.title('%d genes selected' % len(selection_idx))

	dname = 'qc'
	mkdir(os.path.join(pop['output'], dname))
	plt.savefig(os.path.join(pop['output'], dname, 'gene_filtering.png'), dpi=200)
	
	pop['filter_idx'] = pop['nzidx'][selection_idx]

def plot_gene_filter(pop, offset=1):
	M = cat_data(pop, 'M') # get column normalized, factored data
	if 'genefiltering' not in pop:
		lognzcv, lognzmean = mu_sigma(M, pop)
		slope, intercept, r_value, p_value, std_err = linregress(lognzmean, lognzcv)
		pop['genefiltering'] = {}
		pop['genefiltering']['lognzcv'] = lognzcv
		pop['genefiltering']['lognzmean'] = lognzmean
		pop['genefiltering']['slope'] = slope
		pop['genefiltering']['intercept'] = intercept
	plot_mean_cv(pop, offset)

def gene_filter(pop):
	'''
	Plot and interactively select genes

	Parameters
	----------
	pop : dict
		Popalign object
	'''

	fig = plt.figure() # create matplotlib figure
	ax = fig.add_subplot(111) # and axes
	fig.subplots_adjust(bottom=0.25) # leave room at the bottom for slider and button

	M = cat_data(pop, 'M') # aggregate data from samples
	lognzcv, lognzmean = mu_sigma(M, pop) # get logged mean and cv values for the genes in M
	slope, intercept, r_value, p_value, std_err = linregress(lognzmean, lognzcv) # fit regression line

	offset = 1 # default is no offset, log10(1) is 0
	updatedintercept = intercept+np.log10(offset) # update the intercept
	xlims = [min(lognzmean), max(lognzmean)] # gene line x axis limits
	y = [slope*v+updatedintercept for v in xlims] # calculate line y axis values

	selection_idx = np.where(lognzcv>lognzmean*slope+updatedintercept)[0] # update selected genes (above the regression line)
	pop['filter_idx'] = pop['nzidx'][selection_idx] # store the index of the selected genes

	c = np.array(['#1f77b4']*len(lognzmean)) # create color vector (hex value is matplotlib's blue)
	c[selection_idx] = 'maroon' # update the color of the selected genes
	points = plt.scatter(lognzmean, lognzcv, s=1, c=c) # plot genes
	[line] = plt.plot(xlims, y, c='orange') # plots regression line
	plt.title('%d genes selected' % len(selection_idx))
	plt.xlabel('log10(mean)')
	plt.ylabel('log10(cv)')

	text_box = AnchoredText('Less variable', frameon=True, loc=3, pad=0.5) # create information box
	plt.setp(text_box.patch, facecolor='white', alpha=0.5)
	ax.add_artist(text_box)
	text_box = AnchoredText('More variable', frameon=True, loc=1, pad=0.5) # create information box
	plt.setp(text_box.patch, facecolor='white', alpha=0.5)
	ax.add_artist(text_box)

	amp_slider_ax  = fig.add_axes([0.7, 0.10, 0.2, 0.04]) # Define new axes area and draw a slider in it
	offset_slider = Slider(amp_slider_ax, 'Offset slider', 0.5, 2, valinit=1) # create offset slider

	def sliders_on_changed(offset): # when the slider changes
		updatedintercept = intercept+np.log10(offset)  #get new intercept from offset value
		y = [slope*v+updatedintercept for v in xlims] #update y values
		line.set_ydata(y) # set y values for the regression line
		
		selection_idx = np.where(lognzcv>lognzmean*slope+updatedintercept)[0] # update selected genes
		pop['filter_idx'] = pop['nzidx'][selection_idx] # update indices of selected genes
		
		ax.set_title('%d genes selected' % len(selection_idx)) # update title with number of selected genes
		
		c = np.array(['#1f77b4']*len(lognzmean))
		c[selection_idx] = 'maroon'
		points.set_facecolor(c) # update colors
		
		fig.canvas.draw_idle() # update figure
		
	offset_slider.on_changed(sliders_on_changed) # link update function to slider

	filter_button_ax = fig.add_axes([0.7, 0.025, 0.2, 0.05]) # nez axes for button
	filter_button = Button(filter_button_ax, 'Click to filter', color='white', hovercolor='lightgrey') # Button to launch filtering

	def filter_button_on_clicked(mouse_event): # when the button is clicked
		plt.close() # close plot with current selection of genes (stored in pop object)
		filter(pop) # launch filtering of the data
		
	filter_button.on_clicked(filter_button_on_clicked) # link button update function to button
	plt.show()

def removeRBC(pop, species):
	'''
	Remove red blood cells from data

	Parameters
	----------
	pop : dict
		Popalign object
	species : str
		name of experiment species. Can be human or mouse

	'''
	if species == 'mouse':
		genes = ['HBB-BT','HBB-BS' ,'HBA-A1','HBA-A2'] # mouse RBC gene list
	elif species == 'human':
		genes = ['HBB', 'HBA1', 'HBA2'] # human RBC gene list
	else:
		raise Exception('Wrong species (must be mouse or human')
	
	gidx = [np.where(pop['genes']==g)[0][0] for g in genes] # get indices of genes in current gene list
	cellssums = []
	for x in pop['order']: # for each sample
		cellssums.append(np.array(pop['samples'][x]['M'][gidx,:].sum(axis=0)).flatten()) # compute cell sums for the specific genes

	T = otsu(np.concatenate(cellssums)) # find optimal threshold to seperate the low sums from the high sums

	for i,x in enumerate(pop['order']): # for each sample
		idx = np.where(cellssums[i]<=T)[0] # get indices of cells with sums inferior to T
		pop['samples'][x]['M'] = pop['samples'][x]['M'][:,idx] # select cells
		pop['samples'][x]['M_norm'] = pop['samples'][x]['M_norm'][:,idx] # select cells
		print(x, '%d cells kept out of %d' % (len(idx), len(cellssums[i])))

'''
Gene Set Enrichment Analysis functions
'''
def load_dict(dic):
	'''
	Load numpy object

	Parameters
	----------
	dict : str
		Path to a .npy dictionary file

	'''
	return np.load(dic, allow_pickle=True).item()

def sf(k, size_total, n, N):
	'''
	Calculate value of survival function

	Parameters
	----------
	k : int
		Overlap between the gene list and the gene set
	size_total : int
		Total number of genes
	n : int
		Number of genes in the gene set
	N : int
		Number of genes in the gene list
	'''
	return stats.hypergeom.sf(k,size_total,n,N)

def enrichment_analysis(d,genelist,size_total):
	'''
	Find the top gene sets for a given list of genes `genelist`

	Parameters
	----------
	d : dict
		Dictionary of gene set names (key) and gene sets (values)
	genelist : list
		A list of genes to compare to the gene sets in `d`
	size_total : int
		Total number of genes
	'''
	N = len(genelist) # get number of genes in the gene list
	keys = np.array(list(d.keys())) # get a list of gene set names
	with Pool(None) as p:
		q = np.array(p.starmap(sf, [(len(set(d[key]) & set(genelist)), size_total, len(d[key]), N) for key in keys])) # For each gene set, compute the p-value of the overlap with the gene list
	return keys[np.argsort(q)[:3]] # return the most significant gene set names

def gsea(pop, geneset='c5bp'):
	'''
	Perform GSEA on the feature vectors of the feature space

	Parameters
	----------
	pop : dict
		Popalign object
	genes : str
		Name of the gene set dictionary to use. File should be present in `../rsc/gsea/`

	'''
	print('Starting gene set enrichment analysis')
	size_total = len(pop['genes']) # get total number of genes
	pop['feat_labels'] = dict() #create entry to store the top labels for each feature


	currpath = os.path.abspath(os.path.dirname(__file__))
	d = load_dict(os.path.join(currpath, "gsea/%s.npy" % geneset))
	genes = pop['filtered_genes'] # load list of filtered genes

	W = pop['W'] # get feature space W
	stds = np.array(W.std(axis=0)).flatten() # compute stds 
	stdfactor = 2 

	for i in range(pop['nfeats']): # for each feature i
		print('GSEA progress: %d of %d' % ((i+1), pop['nfeats']))
		idx = np.where(np.array(W[:,i]).flatten() > stdfactor*stds[i])[0] # get indices of genes that are above stdfactor times the standard deviation of feature i
		genelist = [genes[j] for j in idx] # gene matching gene names
		pop['feat_labels'][i] = enrichment_analysis(d, genelist, size_total) # for that list of genes, run GSEA

	pop['top_feat_labels'] = [pop['feat_labels'][i][0] for i in range(pop['nfeats'])] # store the top gene set of each feature

'''
Dimensionality reduction functions
'''
def minibatchkmeans(m, k):
	'''
	Perform sklearn's mini batch kmeans

	Parameters
	----------
	m : sparse matrix
		Normalized data
	k : int
		Number of clusters
	'''
	model = sc.MiniBatchKMeans(n_clusters=k) # prepare k means model
	model.fit(m) # fit with data
	return model.cluster_centers_.T, model.predict(m)

def oNMF(X, k, n_iter=500, verbose=1, residual=1e-4, tof=1e-4):
	'''
	Run orthogonal nonnegative matrix factorization

	Parameters
	----------
	X : sparse matrix
		Normalized data
	k : int
		Number of features
	n_iters : int
		Maximum number of iterations
	verbose : boolean
		Print iterations numbers if 1 (or True)
	residual : float
		Algorithm converged if the reconstruction error is below the residual values
	tof : float
		Tolerance of the stopping condition
	'''
	r, c = X.shape #r number of features(genes), c number of samples (cells)
	A, inx = minibatchkmeans(X.T, k) # Initialize the features (centers of the kmeans clusters)
	orthogonal = [1,0] # orthogonality constraints

	Y = ss.csc_matrix((np.ones(c), (inx, range(c))), shape=(k,c)).todense()
	Y = Y+0.2
	if np.sum(orthogonal) == 2:
		S = A.T.dot(X.dot(Y.T))
	else:
		S = np.eye(k)

	X=X.todense()
	XfitPrevious = np.inf
	for i in range(n_iter):
		if orthogonal[0]==1:
			A=np.multiply(A,(X.dot(Y.T.dot(S.T)))/(A.dot(A.T.dot(X.dot(Y.T.dot(S.T))))))
		else:
			A=np.multiply(A,(X.dot(Y.T))/(A.dot(Y.dot(Y.T))))
		A = np.nan_to_num(A)
		A = np.maximum(A,np.spacing(1))

		if orthogonal[1]==1:
			Y=np.multiply(Y,(S.T.dot(A.T.dot(X)))/(S.T.dot(A.T.dot(X.dot(Y.T.dot(Y))))))
		else:
			Y=np.multiply(Y,(A.T.dot(X))/(A.T.dot(A.dot(Y))))
		Y = np.nan_to_num(Y)
		Y = np.maximum(Y,np.spacing(1))

		if np.sum(orthogonal) == 2:
			S=np.multiply(S,(A.T.dot(X.dot(Y.T)))/(A.T.dot(A.dot(S.dot(Y.dot(Y.T))))))
			S=np.maximum(S,np.spacing(1))
		
		if np.mod(i,100) == 0 or i==n_iter-1:
			if verbose:
				print('......... Iteration #%d' % i)
			XfitThis = A.dot(S.dot(Y))
			fitRes = np.linalg.norm(XfitPrevious-XfitThis, ord='fro')
			XfitPrevious=XfitThis
			curRes = np.linalg.norm(X-XfitThis,ord='fro')
			if tof>=fitRes or residual>=curRes or i==n_iter-1:
				print('Orthogonal NMF performed with %d iterations\n' % (i+1))
				break
	return A, Y # return feature space and projection for the cells in X

def nnls(W, V):
	'''
	Project `V` onto `W` with non negative least squares

	Parameters
	----------
	W : array
		Feature space
	V : vector
		Cell gene expression vector
	'''
	return so.nnls(W, V)[0]

def reconstruction_errors(M_norm, q):
	'''
	Compute the mean square errors between the original data `M_norm`
	and reconstructed matrices using projection spaces from list `q`

	Parameters
	----------
	M_norm : sparse matrix
		Normalized data
	q : list
		List of feature space arrays to try
	'''
	errors = []
	projs = []
	D = M_norm.toarray() # dense matrix to compute error
	for j in range(len(q)): # For each feature space j in q
		print('Progress: %d of %d' % ((j+1), len(q)))
		Wj = q[j] # Retrieve feature space j from q
		with Pool(None) as p:
			Hj = p.starmap(nnls, [(Wj, M_norm[:,i].toarray().flatten()) for i in range(M_norm.shape[1])]) # project each cell i of normalized data onto the current W 
		Hj = np.vstack(Hj) # Hj is projected data onto Wj
		projs.append(Hj) # store projection
		Dj = Wj.dot(Hj.T) # compute reconstructed data: Dj = Wj.Hj
		errors.append(mean_squared_error(D, Dj)) # compute mean squared error between original data D and reconstructed data Dj
	return errors, projs

def split_proj(pop, proj):
	'''
	Slice a projection matrix and store each sample's projection

	Parameters
	----------
	pop : dict
		Popalign object
	proj : array
		Projected data in feature space (cells x features)
	'''
	start = 0
	end = 0
	for x in pop['order']: # for each sample in pop
		n = pop['samples'][x]['M'].shape[1] # number of cells
		end += n
		pop['samples'][x]['C'] = proj[start:end,:] # store matching projected data
		start = end

def save_top_genes_features(pop, stds, stdfactor):
	'''
	Save top gene names per feature

	Parameters
	----------
	pop : dict
		Popalign object
	stds : list
		List of the feature standard deviations
	stdfactor : int
		Factor to select genes. For a feature i, genes above stdfactor*stds[i] will be selected
	'''
	out = []
	W = pop['W'] # grab feature space from pop
	filtered_genes = pop['filtered_genes'] # grab names of filtered genes
	for i in range(pop['nfeats']): # for each feature i
		gidx = np.where(np.array(W[:,i]).flatten() > stdfactor*stds[i])[0] # get index of top genes
		gs = [filtered_genes[j] for j in gidx] # get matching names
		out.append(gs) # store in list

	dname = 'qc'
	with open(os.path.join(pop['output'], dname, 'top_genes_per_feat.txt'), 'w') as fout: # create file
		for i in range(pop['nfeats']): # dump selected genes for each feature i
			fout.write('Feature %d:\n' %i)
			fout.write('\n'.join(out[i]))
			fout.write('\n\n')

def plot_top_genes_features(pop):
	"""
	Plot heatmap top genes per feature ~ features

	Parameters
	----------
	pop : dict
		Popalign object
	"""
	W = pop['W'] # grab feature space
	stds = np.array(W.std(axis=0)).flatten() # get the feature standard deviations
	stdfactor = 2 # factor to select genes
	genes_idx = np.array([])
	for i,s in enumerate(stds): # for each feature i and its respective standard deviations
		a = np.where(np.array(W[:,i]).flatten() > stdfactor*stds[i])[0] # get indices of top genes
		sub = np.array(W[a,i]).flatten() # grab coefficients of top genes for feature i
		a = a[np.argsort(sub)[::-1]] # sort subset by coefficent
		genes_idx = np.concatenate([genes_idx, a]) # concatenate gene indices
	genes_idx = genes_idx.astype(int)

	# make list unique (a gene can't appear twice in the list for two different features)
	s = set()
	l = []
	for x in genes_idx:
		if x not in s:
			s.add(x)
			l.append(x)
	genes_idx = np.array(l)

	mtx = W[genes_idx, :] # subset feature space with the top genes for the features
	ax = plt.imshow(mtx, cmap='magma', aspect='auto') # create heatmap
	xlbls = [pop['top_feat_labels'][i] for i in range(pop['nfeats'])]
	plt.xticks(np.arange(pop['nfeats']),xlbls,rotation=90)
	plt.ylabel('Genes')
	plt.yticks([])
	plt.xlabel('Features')

	dname = 'qc'
	mkdir(os.path.join(pop['output'], dname)) # create subfolder
	plt.savefig(os.path.join(pop['output'], dname, 'topgenes_features.pdf'), bbox_inches = "tight")
	plt.close()

	save_top_genes_features(pop, stds, stdfactor)

def plot_reconstruction(pop):
	"""
	Data reconstruction

	Parameters
	----------
	pop : dict
		Popalign object
	"""
	M_norm = cat_data(pop, 'M_norm') # grab normalized data
	W = pop['W'] # grab feature space
	proj = cat_data(pop, 'C') # grab projected data
	max_ = np.max(M_norm)
	idx = np.random.choice(M_norm.shape[1], 200, replace=False) # select 200 cells randomly
	mtx = M_norm[:,idx].toarray() # subset original data
	rcstr = W.dot(proj[idx,:].T) # subset the projected data and cast it onto W to reconstruct the data

	nbins = 50 # number of bins to bin the data value space
	rr = np.linspace(0, max_, nbins)
	step = (rr[1]-rr[0])/2

	x = []
	y = []
	stdinf = []
	stdsup = []
	for i in range(1,nbins): # for each bin 
		min_ = rr[i-1] # get min range
		max_ = rr[i] # get max range
		idx = (mtx>=min_) & (mtx<max_) # get the indices of the data values in that value range
		entries = mtx[idx] # get entries
		matching_entries = rcstr[idx] # get matching entries from the reconstructed matrix
		x.append(min_+step) # set x coordinate for the plot
		a = np.average(matching_entries) # compute average for the matching entries
		y.append(a)
		std = np.sqrt(np.sum(np.power(matching_entries-entries, 2))/(len(entries)-1)) # compute standard deviation of the difference between entries and matching reconstructed entries
		stdinf.append(a-std)
		stdsup.append(a+std)

	plt.plot(x, y, c='black', linewidth=1)
	plt.fill_between(x, stdinf, stdsup, alpha=0.1, color='black')
	plt.xlabel('Original data')
	plt.ylabel('Reconstructed data')
	plt.tick_params(labelsize='small')
	plt.title('Reconstructed average')
	dname = 'qc'
	mkdir(os.path.join(pop['output'], dname))
	plt.savefig(os.path.join(pop['output'], dname, 'reconstructed_data.pdf'))

def scale_W(W):
	'''
	Divide each feature vector of `W` by its L2-norm

	Parameters
	----------
	W : array
		Feature space
	'''
	norms = [np.linalg.norm(np.array(W[:,i]).flatten()) for i in range(W.shape[1])] # compute the L2-norm of each feature
	return np.divide(W,norms) # divide each feature by its respective L2-norm

def onmf(pop, ncells=2000, nfeats=[5,7,9], nreps=3, niter=300):
	'''
	Compute feature spaces and minimize the reconstruction error
	to pick a final feature space
	Perform Gene Set Enrichment Analysis

	Parameters
	----------
	pop : dict
		Popalign object
	ncells : int
		Number of cells to use
	nfeats : int or list of ints
		Number(s) of features to use
	nreps : int
		Number of repetitions to perform for each k in nfeats
	niter : int
		Maximum number of iterations to perform for each instance of the algoirthm
	'''
	M_norm = cat_data(pop, 'M_norm') # grab normalized data
	idx = np.random.choice(M_norm.shape[1], ncells, replace=False) # randomly select ncells cells

	print('Computing W matrices')
	with Pool(None) as p:
		q = p.starmap(oNMF, [(M_norm[:,idx], x, niter) for x in np.repeat(nfeats,nreps)]) # run ONMF in parallel for each possible k nreps times

	q = [scale_W(q[i][0]) for i in range(len(q))] # scale the different feature spaces
	print('Computing reconstruction errors')
	errors, projs = reconstruction_errors(M_norm, q) # compute the reconstruction errors for all the different spaces in q and a list of the different data projections for each feature spaces
	
	print('Retrieving W with lowest error')
	idx_best = np.argmin(errors) # get idx of lowest error
	pop['W'] = q[idx_best] # retrieve matching W
	pop['nfeats'] = pop['W'].shape[1] # store number of features in best W
	proj = projs[idx_best] # retrieve matching projection
	pop['reg_covar'] = max(np.linalg.eig(np.cov(proj.T) )[0])/100 # store a regularization value for GMM covariance matrices

	gsea(pop) # run GSEA on feature space
	split_proj(pop, proj) # split projected data and store it for each individual sample
	plot_top_genes_features(pop) # plot a heatmap of top genes for W
	#plot_reconstruction(pop) # plot reconstruction data

def pca(pop, fromspace='genes'):
	'''
	Run PCA on the samples data

	Parameters
	----------
	pop : dict
		Popalign object
	fromspace : str
		What data to use. If 'genes', normalized filtered data is used, if 'features', projected data is used.
	'''
	n_components = 2
	pca = PCA(n_components=n_components,
		copy=True, 
		whiten=False, 
		svd_solver='auto', 
		tol=0.0, 
		iterated_power='auto', 
		random_state=None) # generate sklearn PCA model

	if fromspace == 'genes':
		M_norm = cat_data(pop, 'M_norm')
		pcaproj = pca.fit_transform(M_norm.toarray().T) # fit PCA model with all the cells
	elif fromspace == 'features':
		C = cat_data(pop, 'C')
		pcaproj = pca.fit_transform(C) # fit PCA model with all the cells

	start = 0
	end = 0
	for x in pop['order']: # for each sample x
		n = pop['samples'][x]['M'].shape[1] # number of cells
		end += n
		pop['samples'][x]['pcaproj'] = pcaproj[start:end,:] # store PCA projection for x
		start = end

	# pcaproj (cells, principal components) matrix is the data projected in PCA space
	pop['pca'] = dict()
	pop['pca']['components'] = pca.components_.T # store PCA space (PC1, PC2)
	pop['pca']['proj'] = pcaproj # store entire PCA projection
	pop['pca']['maxes'] = pcaproj.max(axis=0) # store PCA projection space limits
	pop['pca']['mines'] = pcaproj.min(axis=0) # store PCA projection space limits
	pop['pca']['lims_ext'] = 0.25
	pop['pca']['fromspace'] = fromspace

def matplotlib_to_plotly(cmap, pl_entries):
	'''
	Convert Matplotlib colormap `cmap` to a Plotly colorscale

	Parameters
	----------
	cmap : matplotlib cmap
		A matplotlib cmap
	pl_entries : int
		Number of entries to create for the colorscale
	'''
	h = 1.0/(pl_entries-1)
	pl_colorscale = []
	for k in range(pl_entries):
		C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
		pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
	return pl_colorscale

def plotfeaturesfunc(pop, labeloptions, coloroptions, C, samplenums, samplelbls, colorscale, x, y, z, color, size, opacity):	
	'''
	Plot a 3D scatter plot in feature space
	Update view with widgets values

	Parameters
	----------
	pop : dict
		Popalign object
	labeloptions : list
		List of feature labels
	coloroptions : list
		List of color labels
	samplenums : list
		List of sample numbers (used to color)
	samplelbls : list
		List of sample labels
	colorscale : pl_colorscale
		Plotly colorscale
	x : str
		Name of feature to use for x axis
	y : str
		Name of feature to use for y axis
	z : str
		Name of feature to use for z axis
	color : str
		Color option
	Size : int
		Size of the scatter plot points
	Opacity : float
		Opacity of the scatter plot points
	'''
	# get indices of features from labels
	ix = labeloptions.index(x)
	iy = labeloptions.index(y)
	iz = labeloptions.index(z)
	
	# define color vectors based on color value from dropdown
	if color==coloroptions[0]:
		c='#1f77b4'
	elif color==coloroptions[1]:
		c=samplenums
	elif color==coloroptions[2]:
		gmm = pop['gmm']
		c = gmm.predict(C)
		
	# create trace
	data = [go.Scatter3d(
		x=C[:,ix],
		y=C[:,iy],
		z=C[:,iz],
		text=samplelbls,
		hoverinfo='text',
		mode='markers',
		opacity=1,
		marker=dict(
			size=size,
			color=c,
			opacity=opacity,
			colorscale=colorscale
		),
	)]
	
	# create layout
	layout = go.Layout(
		height=600,
		showlegend=False,
		scene = dict(
			xaxis = dict(
				title='<br>'.join(wrap(x,20))
			),
			yaxis = dict(
				title='<br>'.join(wrap(y,20))
			),
			zaxis = dict(
				title='<br>'.join(wrap(z,20))
			)
		),
		font=dict(
					size=5
		)
	)
	fig = go.Figure(data=data,layout=layout)
	iplot(fig)
	
def plotfeatures(pop):
	'''
	Plot a 3D scatter plot in feature space interactively

	Parameters
	----------
	pop : dict
		Popalign object
	'''

	# create various widgets	
	labeloptions = pop['top_feat_labels']
	x = widgets.Dropdown(
		options=labeloptions,
		value=labeloptions[0],
		description='x:',
	)

	y = widgets.Dropdown(
		options=labeloptions,
		value=labeloptions[1],
		description='y:',
	)
		
	z = widgets.Dropdown(
		options=labeloptions,
		value=labeloptions[2],
		description='z:',
	)

	if 'gmm' in pop:
		coloroptions=['None', 'Samples', 'GMM subpopulations']
	else:
		coloroptions=['None', 'Samples']
		
	color = widgets.Dropdown(
		options=coloroptions,
		value=coloroptions[0],
		description='color:',
	)

	size=widgets.IntSlider(
		value=2,
		min=1,
		max=10,
		step=1,
		continuous_update=False,
		readout=True,
		readout_format='d'
	)

	opacity=widgets.FloatSlider(
		value=1,
		min=0.1,
		max=1,
		step=0.1,
		continuous_update=False,
		readout=True,
		readout_format='.1f',
	)

	C = cat_data(pop, 'C')
	samplenums = np.concatenate([[i]*pop['samples'][x]['C'].shape[0] for i,x in enumerate(pop['order'])])
	samplelbls = np.concatenate([[x]*pop['samples'][x]['C'].shape[0] for x in pop['order']])

	tab20 = matplotlib.cm.get_cmap('tab20')
	colorscale = matplotlib_to_plotly(tab20, 20)

	w = interactive(plotfeaturesfunc,
		pop=fixed(pop), 
		labeloptions=fixed(labeloptions), 
		coloroptions=fixed(coloroptions), 
		C=fixed(C), 
		samplenums=fixed(samplenums),
		samplelbls=fixed(samplelbls),
		colorscale=fixed(colorscale),
		x=x, 
		y=y, 
		z=z, 
		color=color, 
		size=size, 
		opacity=opacity)
	display(w)
	
'''
GMM functions
'''
def default_types():
	'''
	Return a default dictionary of cell types (key) and gene lists (value) pairs
	'''
	types = {
		'Monocytes' : [
			'CD14',
			'CD33',
			'LYZ',
			'FCER1G',
			'LGALS3',
			'CSF1R',
			'ITGAX',
			'ITGAM',
			'CD86',
			'HLA-DRB1'],
		'Dendritic cells' : [
			'LAD1',
			'LAMP3',
			'TSPAN13',
			'CLIC2',
			'FLT3'],
		'B-cells' : [
			'MS4A1',
			'CD19',
			'CD79A'],
		'T-helpers' : [
			'TNF',
			'TNFRSF18',
			'IFNG',
			'IL2RA',
			'BATF'],
		'T cells' : [
			'CD27',
			'CD69',
			'CD2',
			'CD3D',
			'CXCR3',
			'CCL5',
			'IL7R',
			'CXCL8',
			'GZMK'],
		'Natural Killers' : [
			'NKG7',
			'GNLY',
			'PRF1',
			'FCGR3A',
			'NCAM1',
			'TYROBP']
	}
	return types

def typer_func(gmm, prediction, M, genes, types):
	'''
	!!!! Need to work on a normalized matrix. types x components

	Type the components of a Gaussian Mixture Model `gmm`

	Parameters
	----------
	gmm : sklearn.mixture.GaussianMixture
		Mixture model
	prediction : vector
		Vector of cell to mixture assignments. Length of the sample's number of cells.
		If the ith value is 3, cell i is considered to be part of the model's mixture component number 3
	M : sparse matrix
		The column normalized logged data, not gene filtered
	genes : vector
		Array of genes
	types : dict
		Dictionary of cell types (keys) and gene lists (values)
	'''
	if types == None:
		types = default_types()

	typelist = list(types.keys()) # create a list of cell types
	finaltypes = [] # to store top type for each component

	for i in range(gmm.n_components): # for each component number i of the gmm
		idx = np.where(prediction==i)[0] # get the cell indices that match component i
		sub = M[:,idx] # get the matching cells
		avgs = [] # empty 
		for j, group in enumerate(typelist): # for each cell type
			subgenes = types[group] # get the gene list
			subgenes = [g for g in subgenes if g in genes] # only keep known genes
			if len(subgenes) == 0:
				raise Exception('No valid genes in group: %s' % group)
			gidx = [np.where(genes==g)[0][0] for g in subgenes] # get the gene indices
			avgs.append(sub[gidx,:].mean(axis=1).mean()) # get the average value 
		finaltypes.append(typelist[np.argmax(avgs)]) # retrieve the cell type with the largest average
	return finaltypes

def render_model(pop, gmm, C, pcaproj, name, mean_labels=None):
	'''
	Render a model as a density heatmap

	Parameters
	----------
	pop : dict
		Popalign object
	gmm : sklearn.mixture.GaussianMixture
		Mixture model
	C : array
		Feature projected data
	pcaproj : array
		Data projected in pca space
	name : str
		Sample name
	mean_labels : list
		List of cell type labels. If None, the labels are the component number
	'''
	plt.rcParams['figure.figsize'] = [7, 6]
	pcacomps = pop['pca']['components'] # get the pca space
	cmap = 'jet' # define colormap
	w_factor = 3000 # factor to get the mean point sizes (multiplied by the mean weights)
	alpha = 0.2
	mean_color = 'black'
	cbarmin = -15
	cbarmax = -3
	lims_ext = pop['pca']['lims_ext']
	nbins = 200
	x = 0
	y = 1
	row_idx = np.array([x, y])
	col_idx = np.array([x, y])
	maxes = pop['pca']['maxes']
	mines = pop['pca']['mines']
	xlim = (mines[x], maxes[x])
	ylim = (mines[y], maxes[y])
	x_ext = (xlim[1]-xlim[0])*lims_ext
	y_ext = (ylim[1]-ylim[0])*lims_ext
	xlim = (xlim[0]-x_ext, xlim[1]+x_ext)
	ylim = (ylim[0]-y_ext, ylim[1]+y_ext)
	x1 = np.linspace(xlim[0], xlim[1], nbins)
	x2 = np.linspace(ylim[0], ylim[1], nbins)
	X, Y = np.meshgrid(x1, x2)
	pos = np.dstack((X, Y))

	mean_proj = np.zeros((gmm.n_components, 2)) # to project the means in PC1/PC2 space
	sample_density = np.zeros(X.shape)
	w = gmm.weights_ # get the model component weights

	if mean_labels==None:
		mean_labels = list(range(gmm.n_components))
	
	prediction = gmm.predict(C) # get the cells component assignments
	for k in range(gmm.n_components):
		idx = np.where(prediction == k)[0] # get the cell indices for component k
		sub = pcaproj[idx,:] # get the pca projected data for these cells
		mean = sub.mean(axis=0) # compute the mean
		cov = np.cov(sub.T) # compute the covariance matrix
		mean_proj[k,:] = mean # get the mean projected coordinates
		sample_density += w[k]*(np.reshape(mvn.pdf(pos,mean=mean_proj[k].T,cov=cov),X.shape)) # compute the density

	sample_density = np.log(sample_density) # log density
	pp = plt.pcolor(x1, x2, sample_density, cmap=cmap, vmin=cbarmin, vmax=cbarmax) # plot density
	plt.scatter(x=mean_proj[:,0], y=mean_proj[:,1], s=w_factor*w, alpha=alpha, c=mean_color) # plot means
	texts=[]
	for i,txt in enumerate(mean_labels):
		texts.append(plt.text(mean_proj[i,0], mean_proj[i,1], txt)) # plot mean labels (or numbers)
	adjustText.adjust_text(texts)

	pp.set_edgecolor('face')
	cb = plt.colorbar(pp)
	cb.set_alpha(1)
	cb.draw_all()
	cb.set_label('Weighted log probability')
	plt.xlabel('PC%d' % (x+1))
	plt.ylabel('PC%d' % (y+1))
	plt.tick_params(labelsize='small')
	plt.title('Model rendering\n%s' % name)

	dname = 'renderings'
	mkdir(os.path.join(pop['output'], dname))
	name = name.replace('/','')
	plt.savefig(os.path.join(pop['output'], dname, 'model_rendering_%s.png' % name), dpi=200)
	plt.close()
	return sample_density

def grid_rendering(pop, q):
	'''
	Generate a grid plot of gmm renderings

	Parameters
	----------
	pop : dict
		Popalign object
	q : list
		list of sample density arrays
	'''
	plt.rcParams['figure.figsize'] = [7, 6]
	pcacomps = pop['pca']['components']
	cmap = 'jet'
	cbarmin = -15
	cbarmax = -3
	lims_ext = pop['pca']['lims_ext']
	nbins = 200
	x = 0
	y = 1
	row_idx = np.array([x, y])
	col_idx = np.array([x, y])
	maxes = pop['pca']['maxes']
	mines = pop['pca']['mines']
	xlim = (mines[x], maxes[x])
	ylim = (mines[y], maxes[y])
	x_ext = (xlim[1]-xlim[0])*lims_ext
	y_ext = (ylim[1]-ylim[0])*lims_ext
	xlim = (xlim[0]-x_ext, xlim[1]+x_ext)
	ylim = (ylim[0]-y_ext, ylim[1]+y_ext)
	x1 = np.linspace(xlim[0], xlim[1], nbins)
	x2 = np.linspace(ylim[0], ylim[1], nbins)

	nr, nc = nr_nc(len(pop['order']))
	for i, name in enumerate(pop['order']):
		ii = i+1
		plt.subplot(nr, nc, ii)
		pp = plt.pcolor(x1, x2, q[i], cmap=cmap, vmin=cbarmin, vmax=cbarmax)
		pp.set_edgecolor('face')
		plt.xticks([])
		plt.yticks([])
		plt.title(name)

		if i % nc == 0:
			plt.ylabel('PC%d' % (y+1))
		if i >= len(pop['order'])-nc:
			plt.xlabel('PC%d' % (x+1))

	plt.suptitle('Model renderings\n%s' % name)
	dname = 'renderings'
	mkdir(os.path.join(pop['output'], dname))
	name = name.replace('/','')
	plt.savefig(os.path.join(pop['output'], dname, 'model_rendering_%s.png' % 'allsamples'), dpi=200)
	plt.close()

def render_models(pop, mode='grouped'):
	'''
	Parameters
	----------
	pop : dict
		Popalign object
	mode : str
		One of grouped, individual or unique.
		Grouped will render the models individually and together in a separate grid
		Inidividual will only render the models individually
		Unique will render the data's unique model
	'''
	if mode == 'grouped':
		with Pool(None) as p:
			q = p.starmap(render_model, [(pop, pop['samples'][x]['gmm'], pop['samples'][x]['C'], pop['samples'][x]['pcaproj'], x, pop['samples'][x]['gmm_types']) for x in pop['order']])
		grid_rendering(pop, q)

	elif mode == 'individual':
		with Pool(None) as p:
			q = p.starmap(render_model, [(pop, pop['samples'][x]['gmm'], pop['samples'][x]['C'], pop['samples'][x]['pcaproj'], x, pop['samples'][x]['gmm_types']) for x in pop['order']])

	elif mode == 'unique':
		sd = render_model(pop, pop['gmm'], cat_data(pop, 'C'), pop['pca']['proj'], 'uniquegmm', pop['gmm_types'])

def build_single_GMM(k, C, reg_covar):
	'''
	Fit a gaussian mixture model

	Parameters
	----------
	k : int
		Number of components
	C : array
		Feature data
	reg_covar : float
		Regularization of the covariance matrix
	'''
	np.random.seed()
	gmm = smix.GaussianMixture(
		n_components=k,
		covariance_type='full',
		tol=0.001,
		reg_covar=reg_covar,
		max_iter=10000,
		n_init=10,
		init_params='kmeans',
		weights_init=None,
		means_init=None,
		precisions_init=None,
		random_state=None,
		warm_start=False,
		verbose=0,
		verbose_interval=10) # create model
	return gmm.fit(C) # Fit the data

def build_gmms(pop, ks=(5,20), nreps=3, reg_covar=False, rendering='grouped', types=None):
	'''
	Build a Gaussian Mixture Model on feature projected data for each sample

	Parameters
	----------
	pop : dict
		Popalign object
	ks : int or tuple
		Number or range of components to use
	nreps : int
		number of replicates to build for each k in `ks`
	reg_covar : boolean
		If True, the regularization value will be computed from the feature data
		If False, 1e-6 default value is used
	rendering : str
		One of groupd, individual or unique
	types : dict, str or None
		Dictionary of cell types.
		If None, a default PBMC cell types dictionary is provided
	'''
	if isinstance(ks, tuple): # if ks is tuple
		ks = np.arange(ks[0], ks[1]) # create array of ks
	if isinstance(ks, int): # if int
		ks = [ks] # # make it a list

	if 'pca' not in pop:
		pca(pop) # build pca space if necessary

	for i,x in enumerate(pop['order']): # for each sample x
		print('Building model for %s (%d of %d)' % (x, (i+1), len(pop['order'])))
		C = pop['samples'][x]['C'] # get sample feature data
		M = pop['samples'][x]['M'] # get sample gene data
		m = C.shape[0] # number of cells
		n = int(m*0.7) # number of cells for the training set (70%)
		idx = np.random.choice(m, n, replace=False) # get n random cell indices
		not_idx = np.setdiff1d(range(m), idx) # get the validation set indices
		Ctrain = C[idx,:] # subset to get the training sdt
		Cvalid = C[not_idx,:] # subset to get the validation set

		if reg_covar == True:
			reg_covar = pop['reg_covar'] # retrieve reg value from pop object
		else:
			reg_covar = 1e-06 # default value
		with Pool(7) as p: # build all the models in parallel
			q = p.starmap(build_single_GMM, [(k, Ctrain, reg_covar) for k in np.repeat(ks, nreps)])

		# We minimize the BIC score of the validation set
		# to pick the best fitted gmm
		BIC = [gmm.bic(Cvalid) for gmm in q] # compute the BIC for each model with the validation set
		gmm = q[np.argmin(BIC)] # best gmm is the one that minimizes the BIC
		pop['samples'][x]['gmm'] = gmm # store gmm

		pop['samples'][x]['gmm_types'] = typer_func(gmm=gmm, prediction=gmm.predict(C), M=M, genes=pop['genes'], types=types)

	print('Rendering models')
	render_models(pop, mode=rendering) # render the models

def build_unique_gmm(pop, ks=(5,20), nreps=3, reg_covar=False, types=None):
	'''
	Build a unique Gaussian Mixture Model on the feature projected data

	Parameters
	----------
	pop : dict
		Popalign object
	ks : int or tuple
		Number or range of components to use
	nreps : int
		number of replicates to build for each k in `ks`
	reg_covar : boolean
		If True, the regularization value will be computed from the feature data
		If False, 1e-6 default value is used
	types : dict, str or None
		Dictionary of cell types.
		If None, a default PBMC cell types dictionary is provided
	'''
	if 'pca' not in pop:
		pca(pop) # build pca space if necessary

	if isinstance(ks, tuple): # if ks is tuple
		ks = np.arange(ks[0], ks[1]) # create array of ks
	if isinstance(ks, int): # if int
		ks = [ks] # # make it a list

	C = cat_data(pop, 'C') # get feature data
	M = cat_data(pop, 'M') # get gene data
	m = C.shape[0] # get training and validation sets ready
	n = int(m*0.7)
	idx = np.random.choice(m, n, replace=False)
	not_idx = np.setdiff1d(range(m), idx)
	Ctrain = C[idx,:]
	Cvalid = C[not_idx,:]

	if reg_covar == True:
		reg_covar = pop['reg_covar']
	else:
		reg_covar = 1e-06
	with Pool(7) as p:
			q = p.starmap(build_single_GMM, [(k, Ctrain, reg_covar) for k in np.repeat(ks, nreps)])
	
	# We minimize the BIC score of the validation set
	# to pick the best fitted gmm
	BIC = [gmm.bic(Cvalid) for gmm in q]
	gmm = q[np.argmin(BIC)]
	pop['gmm'] = gmm

	pop['gmm_types'] = typer_func(gmm=gmm, prediction=gmm.predict(C), M=M, genes=pop['genes'], types=types)
	render_models(pop, mode='unique') # render model

'''
Align functions
'''
def KL(mu1, cov1, mu2, cov2):
	'''
	Compute the Kullback Leibler divergence between two densities

	Parameters
	----------
	mu1 : vector
		Mean value in feature space
	cov1 : array
		Covariance matrix
	mu2 : vector
		Mean value in feature space
	cov2 : array
		Covariance matrix
	'''
	k = len(mu1)
	return 0.5*(np.trace(np.linalg.inv(cov2).dot(cov1))+
		(mu2-mu1).T.dot(np.linalg.inv(cov2)).dot(mu2-mu1)-k+
		np.log(np.linalg.det(cov2)/np.linalg.det(cov1)))

def JeffreyDiv(mu1, cov1, mu2, cov2):
	'''
	Compute the Jeffrey Divergence between two densities

	Parameters
	----------
	mu1 : vector
		Mean value in feature space
	cov1 : array
		Covariance matrix
	mu2 : vector
		Mean value in feature space
	cov2 : array
		Covariance matrix
	'''
	return np.log10(0.5*KL(mu1, cov1, mu2, cov2)+0.5*KL(mu2, cov2, mu1, cov1))

def plot_deltas(pop):
	'''
	Genere delta mu and delta w plots for the computed alignments

	Parameters
	----------
	pop : dict
		Popalign object
	'''
	for x in pop['order']: # for each sample x
		pop['samples'][x]['means_genes'] = pop['samples'][x]['gmm'].means_.dot(pop['W'].T) # project feature means into gene space

	dname = 'deltas'
	mkdir(os.path.join(pop['output'], dname))

	ref = pop['ref'] # get reference sample name
	for i, lbl in enumerate(pop['samples'][ref]['gmm_types']): # for each component i of the reference model
		samplelbls = []
		delta_mus = []
		delta_ws = []
		mu_ref = pop['samples'][ref]['means_genes'][i] # get the mean i value
		w_ref = pop['samples'][ref]['gmm'].weights_[i] # get the weight i value

		for x in pop['order']: # for each sample x
			if x != ref: # if x is not the reference
				arr = pop['samples'][x]['alignments'] # get the alignments between x and the reference
				try:
					irow = np.where(arr[:,1] == i) # try to get the row where the ref comp number matches i
					itest = int(arr[irow, 0]) # get test comp number from row
					mu_test = pop['samples'][x]['means_genes'][itest] # get the test comp mean value
					w_test = pop['samples'][x]['gmm'].weights_[itest] # get the test comp weight value
					samplelbls.append(x) # store test sample label x
					delta_mus.append(np.linalg.norm([np.array(mu_test).flatten() - np.array(mu_ref).flatten()], ord='fro')) # store delta mu
					delta_ws.append((w_test - w_ref)*100) # store delta w
				except:
					pass

		if len(delta_mus)>0:
			x = np.arange(len(delta_mus)) # create x coordinates 

			# plot delta mus
			markerline, stemlines, baseline = plt.stem(x, delta_mus, markerfmt='ko', use_line_collection=True)
			plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
			plt.setp(stemlines, 'linestyle', 'dotted')

			plt.xticks(x, samplelbls, rotation=90)
			plt.ylabel('Δ\u03BC')
			plt.title('Reference sample %s\nComponent %d: %s' %(ref, i,lbl))

			lbl = lbl.replace('/','')
			plt.savefig(os.path.join(pop['output'], dname, 'delta_mu_comp%d_%s.png' % (i,lbl)), 
						dpi=200, bbox_inches='tight')
			plt.close()

			#plot delta ws
			markerline, stemlines, baseline = plt.stem(x, delta_ws, markerfmt='ko', use_line_collection=True)
			plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
			plt.setp(stemlines, 'linestyle', 'dotted')

			plt.xticks(x, samplelbls, rotation=90)
			plt.ylabel('Δ\u03C9 (%)')
			plt.title('Reference sample %s\nComponent %d: %s' %(ref, i,lbl))

			lbl = lbl.replace('/','')
			plt.savefig(os.path.join(pop['output'], dname, 'delta_w_comp%d_%s.png' % (i,lbl)), 
						dpi=200, bbox_inches='tight')
			plt.close()

def align(pop, ref=None, method='conservative'):
	'''
	Align the commponents of each sample's model to the components of a reference model

	Parameters
	----------
	pop : dict
		Popalign object
	ref : str
		Name of reference sample
	method : str
		Method to perform the alignment
		If conservative, the reference component and the test component have to be each other's best match to align
		If aligntest, the closest reference component is found for each test component
		If alignref, the closest test component is found for each test component
	'''
	if ref == None:
		raise Exception('Please provide sample id of reference')
	elif ref not in pop['samples']:
		raise Exception('Provided reference not in sample list.\nYou can print the list of available samples with show_samples()')

	pop['ref'] = ref # store reference sample name

	refgmm = pop['samples'][ref]['gmm'] # get reference gmm
	for x in pop['order']: # for each sample x
		if x != ref: # if sample is not ref
			testgmm = pop['samples'][x]['gmm'] # get test gmm

			ltest = testgmm.n_components # get test number of components
			lref = refgmm.n_components # get ref number of components
			arr = np.zeros((ltest, lref)) # create empty array to store all pairwise JD values

			for i in range(ltest):
				mutest = testgmm.means_[i,:]
				covtest = testgmm.covariances_[i]
				for j in range(lref):
					muref = refgmm.means_[j,:]
					covref = refgmm.covariances_[j]
					arr[i, j] = JeffreyDiv(mutest, covtest, muref, covref) # compute all pairwise JD values

			if method == 'aligntest':
				minsidx = np.argmin(arr, axis=1) # get idx of closest ref mixture for each test mixture
				mins = np.min(arr, axis=1) # get min divergence values
				res = np.zeros((ltest, 3))
				for i in range(ltest):
					res[i,:] = np.array([i, minsidx[i], mins[i]])

			elif method == 'alignref':
				minsidx = np.argmin(arr, axis=0) # get idx of closest ref mixture for each test mixture
				mins = np.min(arr, axis=0) # get min divergence values
				res = np.zeros((lref, 3))
				for i in range(lref):
					res[i,:] = np.array([minsidx[i], i, mins[i]])

			elif method == 'conservative':
				minstest = [[i,x] for i,x in enumerate(np.argmin(arr,axis=1))]
				minsref = [[x,i] for i,x in enumerate(np.argmin(arr,axis=0))]
				mins = np.min(arr, axis=1)
				minsidx = np.argmin(arr, axis=1)
				idx = []

				for i,row in enumerate(minstest):
					if row in minsref:
						idx.append(i)
				res = np.zeros((len(idx), 3))
				for ii, i in enumerate(idx):
					res[ii,:] = np.array([i, minsidx[i], mins[i]])

			pop['samples'][x]['alignments'] = res

	plot_deltas(pop) # generate plot mu and delta w plots

'''
Entropy functions
'''
def plot_entropies(pop):
	'''
	Plot entropy values for each single gaussin density

	Parameters
	----------
	pop : dict
		Popalign object
	'''
	xvals = []
	yvals = []
	xlbls = []
	lbls = []
	for i, x in enumerate(pop['order']): # for each sample
		xlbls.append(i)
		lbls.append(x)
		E = pop['samples'][x]['entropy'] # get list of entropy values
		xvals += [i]*len(E) # add x coords
		yvals += E # add y coords
	plt.scatter(xvals, yvals) # plot
	plt.xticks(xlbls, lbls, rotation=90)
	plt.ylabel('Entropy')
	plt.title('Entropies of single gaussian densities')
	
	dname = 'entropy'
	mkdir(os.path.join(pop['output'], dname))
	plt.savefig(os.path.join(pop['output'], dname, 'models_entropy.png'), dpi=200, bbox_inches='tight')

def single_entropy(N, S):
	'''
	Compute the entropy of a single gaussian density

	Parameters
	----------
	N : int
		Number of features
	S : array
		Covariance matrix
	'''
	return 0.5*np.log(((2*np.pi*np.exp(1))**N) * np.linalg.det(S));

def entropy(pop):
	'''
	For each sample in `pop`, compute the entropy for the gaussian densities of their Gaussian Mixture Model
	and the mixture of gaussians upper bound

	Parameters
	----------
	pop : dict
		Popalign object
	'''
	N = pop['nfeats'] # get number of features
	for x in pop['order']: # for each sample x
		gmm = pop['samples'][x]['gmm'] # get gmm
		w = gmm.weights_ # get weights of components
		entropy = [single_entropy(N, gmm.covariances_[i]) for i in range(gmm.n_components)] # compute entropy for each gaussian density
		pop['samples'][x]['entropy'] = entropy # store entropy
		pop['samples'][x]['upperbound'] = np.sum([w[i]*(-np.log(w[i]) + entropy[i]) for i in range(gmm.n_components)]) # compute sample's entropy upperbound

	plot_entropies(pop) # generate entropy plots

'''
Rank functions
'''
def rank(pop, ref=None, k=100, niter=200, mincells=50):
	'''
	Generate a ranking plot of the samples against a reference model

	Parameters
	----------
	pop : dict
		Popalign object
	ref : str
		Reference sample name
	k : int
		Number of random cells to use
	niter : int
		Number of iterations to perform
	mincells : int
		If a sample has less than `mincells` cells, is discarded
	'''
	# For a given sample S, k random cells are scored against the reference
	# model. This process is repeated niter times and the results are 
	# shown as boxplots
	scores = []
	lbls = []
	gmmctrl = pop['samples'][ref]['gmm']

	for x in pop['order']:
		C = pop['samples'][x]['C']
		m,n = C.shape
		if m > mincells:
			# nk is actual number of cells used
			# (based on # cells in sample)
			# if k < # cells in sample, nk is k (unchanged)
			# else if k >= # cells in sample, nk is changed to #cells in sample minus 5
			if k<m:
				nk = k
			else:
				nk = m-5
			# Score nk different random cells, niter times
			# keep track of scores, labels and drug classes
			#gmmtest = pop['samples'][x]['gmm']
			for _ in range(niter):
				idx = np.random.choice(m, nk, replace=False)
				sub = C[idx,:]
				#scores.append(gmmctrl.score(sub) / gmmtest.score(sub)) # for Log Likelihood Ratio LLR
				scores.append(gmmctrl.score(sub))
				lbls.append(x)
		else:
			print('Not enough cells for samples: %s' % x)
					
	# create dataframe from scores, labels and drug classes
	# find sample order based on score means
	df = pd.DataFrame({'scores': scores, 'labels': lbls})
	df.scores = df.scores - df.scores.max()
	df2 = pd.DataFrame({col:vals["scores"] for col, vals in df.groupby("labels")})
	means = df2.mean().sort_values()
	lblorder = means.index.values.tolist()

	plt.rcParams['figure.figsize'] = [10,5]
	x = range(len(lblorder))

	# find min and max of the control samples scores
	# against control model. Then plot grey bar to
	# emphasize how these control sample score
	min_ = df[df.labels == ref].scores.min()
	max_ = df[df.labels == ref].scores.max()
	
	# create boxplot using the computed order based on score means
	ax = sns.boxplot(x="labels", y="scores", data=df, order=lblorder, palette='tab20')
	x = plt.xlim()
	plt.fill_between(x, min_, max_, alpha=0.1, color='black')
	# adjusting plot labels
	x = range(len(lblorder))
	plt.xticks(x, lblorder, rotation=90)
	plt.xlabel('Samples')
	plt.ylabel('Log-likelihood scores')
	plt.title('Likelihood scores against reference model (%s)' % ref)
	plt.tight_layout()
	dname = 'ranking'
	mkdir(os.path.join(pop['output'], dname))
	plt.savefig(os.path.join(pop['output'], dname, 'rankings_boxplot.pdf'))

	plt.close()
	# create stripplot using the computed order based on score means
	ax = sns.stripplot(x="labels", y="scores", data=df, order=lblorder, palette='tab20', size=2)
	x = plt.xlim()
	plt.fill_between(x, min_, max_, alpha=0.1, color='black')
	# adjusting plot labels
	x = range(len(lblorder))
	plt.xticks(x, lblorder, rotation=90)
	plt.xlabel('Samples')
	plt.ylabel('Log-likelihood scores')
	plt.title('Likelihood scores against reference model (%s)' % ref)
	#plt.title('Sample scores against %s sample\n(For each sample: %d random cells %d times)' % (ref, k, niter))
	plt.tight_layout()
	dname = 'ranking'
	mkdir(os.path.join(pop['output'], dname))
	plt.savefig(os.path.join(pop['output'], dname, 'rankings_stripplot.pdf'))

'''
Query functions
'''

def plot_query(pop):
	'''
	Plot samples x components matrix (how many % of cells of sample i in component j)
	
	Parameters
	----------

	pop: dict
		Popalign object
	'''
			
	# for each sample
	# get sample cells and get respective gmm component predictions
	# get percentage of cells per component
	gmm = pop['gmm']
	N = len(pop['order'])
	arr = np.zeros((N, gmm.n_components))
	for i,x in enumerate(pop['order']):
		C = pop['samples'][x]['C']
		prediction = gmm.predict(C)
		unique, counts = np.unique(prediction, return_counts=True)
		l = len(prediction)
		counts = np.array([x/l for x in counts])
		d = dict(zip(unique, counts))
		for j in range(gmm.n_components):
			try:
				arr[i,j] = d[j]
			except:
				pass
			
	# create Pandas dataframe
	#components = np.arange(gmm.n_components)
	components = [pop['gmm_types'][i] for i in np.arange(gmm.n_components)]
	df = pd.DataFrame(data=arr, columns=components, index=pop['order'])

	# cluster rows and columns based on their respective correlation matrix
	g = sns.clustermap(df, 
		metric='correlation', 
		method='complete', 
		cmap='bone_r',
		figsize=(10,10)
	)

	# rotate xlabels
	plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

	# adjusting labels
	g.ax_heatmap.set_xlabel('GMM components')
	plt.suptitle('Samples proportions in GMM components (%)', size=20)
	plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

	dname = 'query'
	mkdir(os.path.join(pop['output'], dname))
	plt.savefig(os.path.join(pop['output'], dname, 'query_plot.pdf'), bbox_inches='tight')

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")