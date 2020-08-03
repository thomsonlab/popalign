# Paul Rivaud
# paulrivaud.info@gmail.com
# Sisi Chen
# sisichen@caltech.edu
# Caltech

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
from scipy.stats import norm
from scipy.stats import rankdata
from scipy.cluster import hierarchy as shc
from scipy.spatial import distance as scd
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn import preprocessing as sp
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import mixture as smix
from sklearn.utils import validation
from sklearn.manifold import TSNE
import fastcluster as fc
from sklearn import cluster as sc
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.offsetbox import AnchoredText
from matplotlib import gridspec
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
import adjustText
import ipywidgets as widgets
from ipywidgets import interact, interactive, Layout, HBox, VBox, fixed
from textwrap import wrap
import plotly
#from plotly.graph_objs import graph_objs as go
import plotly.graph_objs as go
from plotly.offline import iplot
import umap.umap_ as umap
import time
import shutil


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
	if name not in ['M', 'M_norm', 'C', 'pcaproj', 'cell_type']:
		raise Exception('name must be one of M, M_norm, C, pcaproj, cell_type')
	if name in ['M', 'M_norm']:
		tmp = ss.hstack([pop['samples'][x][name] for x in pop['order']])
	if name in ['cell_type']:
		tmp = np.hstack([pop['samples'][x]['cell_type'] for x in pop['order']]).tolist()
	elif name in ['C','pcaproj']:
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

def otsu(pop, X, nbins=50):
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
	with Pool(pop['ncores']) as p:
		q = p.starmap(intraclass_var, [(X,t) for t in thresholds[:-1]])
		
	# pick threshold that minimizes the intraclass variance
	return thresholds[np.argmin(q)]

def print_ncells(pop):
	'''
	Display number of cells of each sample

	Parameters
	----------
	pop : dict
		Popalign object
	'''
	total = 0
	for x in pop['order']:
		tmp = pop['samples'][x]['M'].shape[1]
		total += tmp
		print(x, '\t', tmp)
	print('Total number of cells loaded: %d' % total)

'''
Load functions
'''
def load_genes(genes):
	'''
	Load the gene names from a file

	Parameters
	----------
	genes : str
		Path to a gene file
	'''
	try:
		genes = np.array([row[1].upper() for row in csv.reader(open(genes), delimiter="\t")]) # 10X format
	except:
		genes = np.array([row[0].upper() for row in csv.reader(open(genes), delimiter="\t")]) # base format with one gene name per row
	return genes

def load_samples(samples, controlstring=None, genes=None, outputfolder='output', existing_obj=None):
	'''
	Load data from a dictionary and gene labels from a file

	Parameters
	----------
	samples : dict
		Dictionary of sample names (keys) and paths to their respective matrix files (values)
	controlstring: string
		String containing common name across all control samples so that we can pull them out easily	
	genes : str
		Path to a .tsv 10X gene file. Optional if existing_obj is provided
	outputfolder : str
		Path (or name) of the output folder to create
	existing_obj : dict, optional
		Object previously returned by either load_samples() or load_screen(). New samples will be added to that object
	'''
	if (genes == None) & (existing_obj == None):
		raise Exception('Please specify path to gene file')

	if existing_obj == None:
		obj = {}
		obj['samples'] = {}
		obj['order'] = []
		obj['controlstring'] = controlstring
		obj['genes'] = load_genes(genes) # load and store genes
		obj['ncores'] = None
	else:
		obj = existing_obj
	for x in samples:
		x = str(x)
		# create entry and load sparse matrix
		obj['samples'][x] = {}
		obj['samples'][x]['M'] = sio.mmread(samples[x]).tocsc()
		obj['order'].append(x) # add sample name to list to always call them in the same order for consistency	
	
	# save start and end of cell indices of sample relative to other samples
	start = 0
	end = 0
	for x in obj['order']:
		n = obj['samples'][x]['M'].shape[1]
		end = start+n
		obj['samples'][x]['indices'] = (start,end)
		start = end

	obj['nsamples'] = len(obj['order']) # save number of samples
	obj['output'] = outputfolder # define name of output folder to save results
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
		raise Exception('%s not a valid column. Must be one of:' % s, cols.tolist())

def load_multiplexed(matrix, barcodes, metafile, controlstring=None, genes=None, outputfolder='output', existing_obj=None, only=[], col=None, value=None):
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
	controlstring: string
		String containing common name across all control samples so that we can pull them out easily	
	genes : str
		Path to a .tsv 10X gene file. Optional if existing_obj is provided
	outputfolder : str, optional
		Path (or name) of the output folder to create
	existing_obj : dict, optional
		Object previously returned by either load_samples() or load_screen(). New samples will be added to that object
	only : list, optional
		List of sample names to load (other samples with names not in list will not be loaded)
	col : str, optional
		Name of a specific column in the meta data to use
	value : str or int, optional
		Value in the specified meta data column `col` to use to filter samples to load
	'''
	if (genes == None) & (existing_obj == None):
		raise Exception('Please specify path to gene file')

	if existing_obj == None:
		obj = {}
		obj['samples'] = {}
		obj['order'] = []
		obj['controlstring'] = controlstring
		obj['genes'] = load_genes(genes) # load and store genes
		obj['ncores'] = None
	else:
		obj = existing_obj

	# check if meta data has the minimum requirements
	meta = pd.read_csv(metafile, header=0) # load metadata file
	cols = meta.columns.values
	check_cols('cell_barcode', cols)
	check_cols('sample_id', cols)
	if col != None:
		check_cols(col, cols)
		if value not in meta[col].unique():
			raise Exception('Provided value not in column %s' % col) 
	
	# Find sample names that obey the metadata filter 
	if (value != None) and (col != None):
		tmp_only = meta[meta[col]==value]['sample_id'].dropna().unique()
	elif (value != None) and (col == None):
		raise Exception('col and value arguments must be specified together, or both equal to None')
	elif (value == None) and (col != None):
		raise Exception('col and value arguments must be specified together, or both equal to None')
	else:
		tmp_only =  meta['sample_id'].dropna().unique() # get unique list of sample names

	if only == []: # if no specific sample specified
		only = tmp_only # get unique list of sample names

	M = sio.mmread(matrix).tocsc() # load main matrix
	barcodes = np.array([row[0] for row in csv.reader(open(barcodes), delimiter="\t")]) # load associated barcodes
	bc_idx = {} # store index of each barcode in a dictionary to quickly retrieve indices for a list of barcodes
	for i, bc in enumerate(barcodes):
		bc_idx[bc] = i

	accum_idx = [] # accumulate index values for subsetted samples
	for i in only: # go through the sample_id values to split the data and store it for each individual sample
		x = str(i)
		if x != 'unknown':
			obj['samples'][x] = {} # create entry for sample x
			sample_bcs = meta[meta.sample_id == i].cell_barcode.values # get the cell barcodes for sample defined by sample_id
			idx = [bc_idx[bc] for bc in sample_bcs] # retrieve list of matching indices
			obj['samples'][x]['M'] = M[:,idx] # extract matching data from M
			obj['order'].append(x) # save list of sample names to always call them in the same order for consistency
			accum_idx = accum_idx + idx 
			# store cell type information in samples 
			if 'cell_type' in cols: 
				obj['samples'][x]['cell_type'] = meta.loc[idx].cell_type.tolist() # extract supplied cell types from metadata

	# Trim the meta data file to only contain the filtered samples
	currmeta = meta.loc[accum_idx]
	obj['meta'] = currmeta

	# save start and end of cell indices of sample relative to other samples
	if existing_obj != None:
		start = existing_obj['ncells']
	else:
		start = 0
	end = 0
	for x in obj['order']:
		n = obj['samples'][x]['M'].shape[1]
		end = start+n
		obj['samples'][x]['indices'] = (start,end)
		start = end
	obj['ncells'] = end
	obj['nsamples'] = len(obj['order']) 
	obj['output'] = outputfolder

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

def scale_factor(pop, ncells):
	'''
	Find a scaling factor that minimizes the difference between the original mean of the data and the mean after column normalization and factorization

	Parameters
	----------
	pop : dict
		Popalign object
	'''
	M = cat_data(pop, 'M') # Aggregatre data from all samples
	if ncells != None:
		if ncells<M.shape[1]:
			idx = np.random.choice(M.shape[1], ncells, replace=False) # select 200 cells randomly
			M = M[:,idx]

	ogmean = pop['original_mean'] # Retrive original mean of the data prior to normalization
	factorlist = [500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000] # list of factors to try

	with Pool(pop['ncores']) as p:
		q = p.starmap(comparison_factor, [(M.copy(), f, ogmean) for f in factorlist]) # try different factors
	scalingfactor = factorlist[np.argmin(q)] # pick the factor that minimizes the difference between the new mean and the original one 
	pop['scalingfactor'] = scalingfactor
	with Pool(pop['ncores']) as p:
		q = p.starmap(factor, [(pop['samples'][x]['M'], scalingfactor) for x in pop['order']]) #Multiply data values by picked factor in parallel
	for i,x in enumerate(pop['order']):
		pop['samples'][x]['M'].data = q[i] # update data values for each sample

def normalize(pop, scaling_factor=None, ncells=None):
	'''
	Normalize the samples of object `pop` and applies a normalization factor

	Parameters
	----------
	pop : dict
		Popalign object
	scaling_factor : int or None, optional
		Number used to scale the data values. If None, that factor is computed automatically.
	ncells : int or None
		Number of cells to randomly subsample to try different normalization factors to use less memory. If None, all cells are used.
	
	Output
	----------
	pop['normed']: bool
		Indicates if data has been rescaled and logged
	pop['original_mean']: list, float
		Gene means before normalizing
	'''
	if 'normed' in pop:
		if pop['normed']==True:
			print('Data already column normalized')
	else:
		M = cat_data(pop, 'M') # aggregate data
		pop['original_mean'] = M.mean() # store original mean of the data

		print('Performing column normalization')
		for x in pop['order']: # for each sample x
			col_norm(pop['samples'][x]['M']) #column normalize data of sample `x`
		
		if scaling_factor != None:
			with Pool(pop['ncores']) as p:
				q = p.starmap(factor, [(pop['samples'][x]['M'], scaling_factor) for x in pop['order']]) #Multiply data values by factor in parallel
			for i,x in enumerate(pop['order']):
				pop['samples'][x]['M'].data = q[i] # update data values for each sample
		else:
			print('Finding best scaling factor')
			scale_factor(pop, ncells) # Apply normalization factor
			print('Best scaling factor beta is: ' + str(pop['scalingfactor']))
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
	Outputs: 
	----------
	Adds the following fields to the pop object: 
	pop['nzidx'] : list, int
		indices of nonzero genes
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

def filter(pop, remove_ribsomal=True, remove_mitochondrial=True):
	'''
	Filter genes from data in `pop`. Discards Ribosomal genes that start with RPS or RPL

	Parameters
	----------
	pop :dict
		Popalign object
	remove_ribsomal : bool
		Whether to remove the ribosomal genes. Default: True
	remove_mitochondrial : bool
		Whether to remove the mitochondrial geneset. Default: True
	Outputs: 
	----------
	Adds the following fields to the pop object: 
	pop['filtered_genes'] : list
		Names of filtered genes
	pop['filtered_genes_set'] : set
		Names of filtered genes

	'''
	gene_idx = pop['filter_idx'] # get indices of genes to keep
	genes = pop['genes'] # get all genes names
	if remove_ribsomal or remove_mitochondrial:
		if remove_ribsomal and remove_mitochondrial:
			prefixes = ('RPS', 'RPL', 'MT-')
		elif remove_ribsomal:
			prefixes = ('RPS', 'RPL')
		elif remove_mitochondrial:
			prefixes = ('MT-')

		tmp = []
		print('Removing ribosomal and/or mitochondrial genes')
		for i in gene_idx:
			g = genes[i]
			if g.startswith(prefixes):
				pass
			else:
				tmp.append(i) # only append gene index if gene name doesn't star with RPS or RPL
		gene_idx = np.array(tmp)

	print('Filtering genes and logging data')
	for x in pop['order']: # for each sample x
		M_norm = pop['samples'][x]['M'][gene_idx,:] # filter genes
		M_norm.data = np.log(M_norm.data+1) # log the data
		pop['samples'][x]['M_norm'] = M_norm # store the filtered logged data
		pop['samples'][x]['M'].data = np.log(pop['samples'][x]['M'].data+1) # log the non gene-filtered matrix 

	filtered_genes = [genes[i] for i in gene_idx] # get gene names of filtered genes
	pop['filter_idx'] = gene_idx
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

	Outputs: 
	----------
	Adds the following fields to the pop object: 
	pop['filter_idx'] : list, int
		indices of filtered genes based on offset

	'''

	lognzcv = pop['genefiltering']['lognzcv'] # get cv values
	lognzmean = pop['genefiltering']['lognzmean'] # get mean values
	slope = pop['genefiltering']['slope'] # get line slope
	intercept = pop['genefiltering']['intercept'] # get line intercept

	adjusted_intercept = intercept+np.log10(offset) # slide filtering line with offset
	selection_idx = np.where(lognzcv>lognzmean*slope+adjusted_intercept)[0] # get indices of genes above the filtering line
	print('%d genes selected' % len(selection_idx))

	plt.figure(figsize=(8,6))
	plt.scatter(lognzmean, lognzcv, s=1) # plot all genes
	plt.scatter(lognzmean[selection_idx], lognzcv[selection_idx], c='maroon', s=1) # plot genes above line with different color
	plt.plot(lognzmean,lognzmean*slope+adjusted_intercept, c='darkorange') # plot filtering line
	plt.xlabel('log10(mean)')
	plt.ylabel('log10(cv)')
	plt.title('%d genes selected' % len(selection_idx))

	dname = 'qc'
	mkdir(os.path.join(pop['output'], dname))
	plt.savefig(os.path.join(pop['output'], dname, 'gene_filtering.pdf'), dpi=200)
	
	pop['filter_idx'] = pop['nzidx'][selection_idx]

def plot_gene_filter(pop, offset=1):
	'''
	Plot genes by their log(mean) and log(coefficient of variation) 
	and also filters genes based on specified offset. 

	Parameters: 
	----------
	offset: float
		Value (its log) will be added to the intercept of the linear fit to filter genes

	Outputs: 
	----------
	Adds the following fields to the pop object: 
	pop['filter_idx'] : list, int
		indices of filtered genes based on offset
		Added by plot_mean_cv
	pop['nzidx'] : list, int
		indices of nonzero genes
		Added by mu_sigma

	The following should not need to be accessed directly: 
	pop['genefiltering']['lognzcv'] 
	pop['genefiltering']['lognzmean'] 
	pop['genefiltering']['slope'] 
	pop['genefiltering']['intercept'] 

	'''
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

def gene_filter(pop, remove_ribsomal=True):
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
	# need to fix this to update all pieces of data
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

	T = otsu(pop, np.concatenate(cellssums)) # find optimal threshold to seperate the low sums from the high sums

	start = 0
	end = 0
	for i,x in enumerate(pop['order']): # for each sample
		idx = np.where(cellssums[i]<=T)[0] # get indices of cells with sums inferior to T
		pop['samples'][x]['M'] = pop['samples'][x]['M'][:,idx] # select cells
		pop['samples'][x]['M_norm'] = pop['samples'][x]['M_norm'][:,idx] # select cells
		end = start+len(idx) # update start and end cell indices
		pop['samples'][x]['indices'] = (start,end) # update start and end cell indices
		start = end # update start and end cell indices
		#print(x, '%d cells kept out of %d' % (len(idx), len(cellssums[i])))

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

def enrichment_analysis(pop,d,genelist,size_total,ngenesets):
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
	with Pool(pop['ncores']) as p:
		q = np.array(p.starmap(sf, [(len(set(d[key]) & set(genelist)), size_total, len(d[key]), N) for key in keys])) # For each gene set, compute the p-value of the overlap with the gene list
	return keys[np.argsort(q)[:ngenesets]] # return top 

def gsea(pop, geneset='c5bp', ngenesets=20):
	'''
	Perform GSEA on the feature vectors of the feature space

	Parameters
	----------
	pop : dict
		Popalign object
	genes : str
		Name of the gene set dictionary to use. File should be present in `../rsc/gsea/`
	ngenesets : int
		Number of top genesets to return
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
		print('GSEA progress: %d of %d' % ((i+1), pop['nfeats']), end='\r')
		idx = np.where(np.array(W[:,i]).flatten() > stdfactor*stds[i])[0] # get indices of genes that are above stdfactor times the standard deviation of feature i
		genelist = [genes[j] for j in idx] # gene matching gene names
		pop['feat_labels'][i] = enrichment_analysis(pop, d, genelist, size_total, ngenesets) # for that list of genes, run GSEA
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

def oNMF(X, k, n_iter=500, verbose=0, residual=1e-4, tof=1e-4):
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

def reconstruction_errors(pop, M_norm, q):
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
		print('Progress: %d of %d' % ((j+1), len(q)), end='\r')
		Wj = q[j] # Retrieve feature space j from q
		with Pool(pop['ncores']) as p:
			Hj = p.starmap(nnls, [(Wj, M_norm[:,i].toarray().flatten()) for i in range(M_norm.shape[1])]) # project each cell i of normalized data onto the current W 
		Hj = np.vstack(Hj) # Hj is projected data onto Wj
		projs.append(Hj) # store projection
		Dj = Wj.dot(Hj.T) # compute reconstructed data: Dj = Wj.Hj\
		curr_error = mean_squared_error(D, Dj)
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
	# Need to fix this function to accommodate other feature sets

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
		#for i in range(pop['nfeats']): # dump selected genes for each feature i
		for i, lbl in enumerate(pop['top_feat_labels']): # dump selected genes for each feature i
			print(i, lbl)
			fout.write('Feature %d, %s:\n' % (i,lbl))
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
	# proj = cat_data(pop, 'C') # grab projected data
	proj = get_cat_coeff(pop)
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

def plot_H(pop, method='complete', n=None):
	'''
	Plot the projection in feature space of the data

	Parameters
	----------
	pop : dict
		Popalign object
	method : str
		Hierarchical clustering method. Default is complete
	n : int or None
		Number of cells to randomly sample.
	'''
	C = cat_data(pop, 'C') # get feature data
	
	if n != None:
		if not isinstance(n, int): # check that n is an int
			raise Exception('n must be an int')
		if n<C.shape[0]: # if n is small than the number of cells in C
			idx = np.random.choice(C.shape[0], n, replace=False) # randomly select ncells cells
			C = C[idx,:] # subsample

	d = pairwise_distances(X=C,metric='correlation',n_jobs=-1) # pairwaise distance matrix
	np.fill_diagonal(d,0.) # make sure diagonal is not rounded to some small value
	d = scd.squareform(d,force='tovector',checks=False) # force matrix to vector form
	d = np.nan_to_num(d)
	z = fc.linkage(d,method=method) # create linkage from distance matrix
	z = z.clip(min=0)
	idx = shc.leaves_list(z) # get clustered ordered
	X = C[idx,:] # append ordered feature data
	X = X.T # concatenate the clustered matrices of all samples. Cells are columns, features are rows.
	
	plt.imshow(X, aspect='auto') # generate heatmap
	plt.yticks(np.arange(pop['nfeats']), pop['top_feat_labels'])
	plt.xticks([])
	plt.xlabel('Cells')
	if n == None:
		plt.title('%d cells from %d samples' % (X.shape[1], pop['nsamples']))
	else:
		plt.title('%d randomly selected cells from %d samples' % (X.shape[1], pop['nsamples']))

	dname = 'qc'
	mkdir(os.path.join(pop['output'], dname)) # create subfolder
	plt.savefig(os.path.join(pop['output'], dname, 'projection_cells.pdf'), bbox_inches = "tight", dpi=300)
	plt.close()

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
	maxncells = M_norm.shape[1] # get total number of cells
	if ncells > maxncells: # if ncells is larger than total number of cells
		ncells = maxncells # adjust number down
	idx = np.random.choice(M_norm.shape[1], ncells, replace=False) # randomly select ncells cells

	print('Computing W matrices')
	with Pool(pop['ncores']) as p:
		q = p.starmap(oNMF, [(M_norm[:,idx], x, niter) for x in np.repeat(nfeats,nreps)]) # run ONMF in parallel for each possible k nreps times

	q = [scale_W(q[i][0]) for i in range(len(q))] # scale the different feature spaces
	print('Computing reconstruction errors')
	errors, projs = reconstruction_errors(pop, M_norm, q) # compute the reconstruction errors for all the different spaces in q and a list of the different data projections for each feature spaces
	
	# choose the best out of all replicates
	mlist = np.repeat(nfeats,nreps)

	bestofreps = []
	for i in range(len(nfeats)): 
	    curridx = np.where(mlist==nfeats[i])[0]
	    currerrors = [errors[i] for i in curridx]
	    bestidx = np.argmin(currerrors)
	    bestofreps.append(curridx[bestidx])
	    
	# Now subselect only the best of the replicate feature sets
	q = [q[i] for i in bestofreps]
	errors = [errors[i] for i in bestofreps]
	projs = [projs[i] for i in bestofreps]


	# Store each of these feature sets into the pop object for posterity
	pop['onmf'] = {}
	pop['onmf']['q'] = q
	pop['onmf']['errors'] = errors
	pop['onmf']['projs'] = projs
	pop['onmf']['nfeats'] = nfeats

def choose_featureset(pop, m = [], alpha = 3, multiplier=3):
	'''
	Choose featureset from store oNMF calculations. Either user directly supplies a preferred m value or the 

	Parameters
	----------
	errors : list
		list of MSE errors from oNMF
	alpha : float
		power of polynomial
	multiplier : float
		multiplies constant C in f(m)

	'''
	# Unpack variables from pop object
	q = pop['onmf']['q']
	errors = pop['onmf']['errors']
	projs = pop['onmf']['projs']
	nfeats = pop['onmf']['nfeats']

	if m in nfeats: 
		print('Using oNMF featureset for specified m: ' + str(m))
		bestm = m
	else: 
		print('Retrieving oNMF featureset with lowest f(m): ')
		# idx_best = np.argmin(errors) # get idx of lowest error
		bestm = find_best_m(pop, alpha, multiplier)
		print('Featureset with ' + str(bestm) + ' features loaded')

	idx_best = np.argwhere(np.array(nfeats)==bestm)[0][0]
	# Store featureset and coefficients into the individual samples
	pop['W'] = q[idx_best] # retrieve matching W
	pop['nfeats'] = pop['W'].shape[1] # store number of features in best W
	proj = projs[idx_best] # retrieve matching projection
	# pop['reg_covar'] = max(np.linalg.eig(np.cov(proj.T))[0])/100 # store a regularization value for GMM covariance matrices

	gsea(pop) # run GSEA on feature space
	split_proj(pop, proj) # split projected data and store it for each individual sample
	plot_top_genes_features(pop) # plot a heatmap of top genes for W
	plot_H(pop, method='complete', n=2000)
	#plot_reconstruction(pop) # plot reconstruction data

def find_best_m(pop, alpha = 3, multiplier = 3): 
	'''
	Find the best number of features (m) given the MSE error and 
	parameters for polynomial cost function f(m)

	Parameters
	----------
	errors : list
		list of MSE errors from oNMF
	alpha : float
		power of polynomial
	multiplier : float
		multiplies constant C in f(m)

	Output
	----------
	bestm : int
		value of m (number of features) that minimizes cost function f(m)
	'''
	errors = pop['onmf']['errors']
	nfeats = pop['onmf']['nfeats']

	# First rescale MSE so it starts at 1
	errors_2 = np.divide(errors, np.max(errors))

	# set all powers and scaling factors to try
	alphas = [0.7, 0.9, 1, 2, 3, 4, 5]
	if alpha not in alphas: 
		alphas.append(alpha)
		alphas = np.sort(alphas).tolist()

	basescales = []
	r_univ= range(1,30)

	for i in range(len(alphas)):
		currscale = np.round(np.max(np.power(np.array(r_univ),alphas[i]))*250)
		basescales.append(currscale)

	# Assemble f(m) values by sweeping over j and alpha
	fm_df = pd.DataFrame()
	minvals = []
	jrange= range(1,8)
	if multiplier not in jrange: 
		jrange.append(multiplier)
		jrange = np.sort(jrange).tolist()

	for i in range(len(alphas)): 
		curralpha = alphas[i]
		currbasescale = basescales[i]
		currminvals = []
		for j in jrange:               
			currscale = j*1/currbasescale # increase j linearly
			#         currscale = np.power(2,j) * currbasescale # increase j by powers of 2
			curr_values = currscale*(np.power(np.array(nfeats),curralpha)) + np.array(errors)
			currmin = np.argwhere(curr_values==np.min(curr_values))[0][0]

			curr_df = {'vals': np.append(curr_values,np.min(curr_values)),
						'm': np.append(nfeats,currmin)}
			curr_df = pd.DataFrame(curr_df)
			curr_df['scale'] = currscale
			curr_df['a'] = curralpha
			curr_df['j'] = j
			curr_df['col'] = np.append(0*curr_values,1)
			fm_df = fm_df.append(curr_df, ignore_index=True)

			currminvals.append(nfeats[currmin])
		minvals.append(currminvals)

	# Find best m given supplied alpha and multiplier: 
	irow = alphas.index(alpha)
	icol = jrange.index(multiplier)
	idx_flat = irow*len(jrange) + icol
	bestm = minvals[irow][icol]

	# Plot MSE curve
	plt.scatter(nfeats,errors, marker=".", s=100)
	plt.plot(nfeats,errors, marker=".")
	plt.ylabel('m')
	plt.savefig(os.path.join(pop['output'], 'featurechoice_1_mse.pdf'), bbox_inches = "tight")
	plt.close()

	# Plot f(m) curves
	with sns.plotting_context('notebook',font_scale=1.7):
		g = sns.FacetGrid(fm_df, col="j",  row="a", hue="col")
		g = (g.map(plt.scatter, "m", "vals", marker=".", s=400))#.set(ylim=(0.03,0.055))
		g = (g.map(plt.plot, "m", "vals", marker="."))#.set(ylim=(0.03,0.055))

		# Highlight the curve that uses the chosen parameters
		axes = g.axes.flatten()
		ax = axes[idx_flat]
		for _, spine in ax.spines.items():
			spine.set_visible(True) # You have to first turn them on
			spine.set_color('magenta')
			spine.set_linewidth(2)

		plt.savefig(os.path.join(pop['output'], 'featurechoice_2_fm_curves.pdf'), bbox_inches = "tight")
		plt.close()

	# Plot phase portrait of argmin
	plt.figure(figsize=(5,7)) 
	heat_map = sns.heatmap(minvals, annot=True, cmap='viridis')
	plt.yticks(plt.yticks()[0], alphas)
	heat_map.set_ylim(len(minvals)+0.5, -0.5)
	plt.ylabel('alpha (a)')
	plt.xlabel('constant multiplier (j)')
	plt.yticks(rotation=0)
	plt.xticks(np.array(jrange)-0.5, jrange)
	# highlight the m value selected in the data
	heat_map.add_patch(plt.Rectangle((icol, irow), 1, 1, fill=False, edgecolor='magenta', lw=2))
	plt.savefig(os.path.join(pop['output'], 'featurechoice_3_phase_argmin_m.pdf'), bbox_inches = "tight")
	plt.close()

	return bestm

def pca(pop, n_components=2, fromspace='genes'):
	'''
	Run PCA on the samples data

	Parameters
	----------
	pop : dict
		Popalign object
	fromspace : str
		What data to use. If 'genes', normalized filtered data is used, if 'features', projected data is used.
	'''
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
	elif fromspace == 'onmf':
		C = cat_data(pop, 'C') # use onmf features 
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
	pop['pca']['mean'] = pca.mean_
	pop['pca']['explained_variance'] = pca.explained_variance_

def get_cat_coeff(pop):
	'''
	Get concatenated coefficients; either oNMF or pca as specified in pop
	Parameters
	----------
	pop : dict
		Popalign object

	'''
	if 'featuretype' not in pop.keys():
		raise Exception('pop[\'featuretype\'] was not found. Run build_gmms which sets the featuretype for the pop object')

	featuretype = pop['featuretype']
	if featuretype == 'pca':
		coeff = cat_data(pop,'pcaproj')
	elif featuretype == 'onmf': 
		coeff = cat_data(pop, 'C') # get feature data
	else:
		raise Exception('featuretype variable must be: \'pca\' or \'onmf\'')
	return coeff

def get_coeff(pop, name):
	'''
	Get coefficients for a specific sample; either oNMF or pca as specified in pop

	Parameters
	----------
	pop : dict
		Popalign object
	name : str
		name of sample
	'''
	if 'featuretype' not in pop.keys():
		raise Exception('pop[\'featuretype\'] was not found. Run build_gmms which sets the featuretype for the pop object')

	featuretype = pop['featuretype']
	if featuretype == 'pca':
		coeff = pop['samples'][name]['pcaproj']
	elif featuretype == 'onmf': 
		coeff = pop['samples'][name]['C']
	else:
		raise Exception('featuretype variable must be: \'pca\' or \'onmf\'')
	return coeff

def get_features(pop):
	'''
	Get feature vectors for pop object; either oNMF or pca as specified in pop

	Parameters
	----------
	pop : dict
		Popalign object
	'''

	if 'featuretype' not in pop.keys():
		raise Exception('pop[\'featuretype\'] was not found. Run build_gmms which sets the featuretype for the pop object')

	featuretype = pop['featuretype']
	if featuretype == 'pca':
		features = pop['pca']['components']
	elif featuretype == 'onmf': 
		features = pop['W']
	else:
		raise Exception('featuretype variable must be: \'pca\' or \'onmf\'')
	return features

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
		c = '#1f77b4'
	elif color==coloroptions[1]:
		c = samplenums
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

	# C = cat_data(pop, 'C')
	C = get_cat_coeff(pop)
	samplenums = np.concatenate([[i]*C.shape[0] for i,x in enumerate(pop['order'])])
	samplelbls = np.concatenate([[x]*C.shape[0] for x in pop['order']])

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
def default_pbmc_types():
	'''
	Return a default dictionary of cell types (key) and gene lists (value) pairs for PBMC cell types
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
	if types == 'defaultpbmc':
		types = default_pbmc_types() # get default types
	
	typeslist = list(types.keys()) # get list of types for consistency
	cols = range(gmm.n_components) # range of components
	finaltypes = [] # to store top type for each component

	markers = np.concatenate([types[x] for x in types]) # get entire list of genes from dict
	markers = [x for x in markers if x in genes] # make sure to only keep valid genes
	genes_idx = [np.where(genes==x)[0][0] for x in markers] # get matching indices

	M = M[genes_idx,:] # genes from initial M matrix
	M = sp.normalize(M, norm='max', axis=1) # scale rows by max

	arr = [] # empty list to store mean vectors for each type
	for t in typeslist: # for each cell type t
		l = types[t] # get matching list of genes
		lidx = [markers.index(g) for g in l if g in markers] # retrieve gene indices from marker list (only valid genes in markers list)
		sub = M[lidx,:] # pick only genes for cell type t
		submean = np.array(sub.mean(axis=0)).flatten() # compute mean for each cell
		arr.append(submean) # store mean vector
	arr = np.vstack(arr) # stack mean vectors
	calls = np.argmax(arr,axis=0) # for each cell, get the index of max value

	for i in cols: # for each component i
		idx = np.where(prediction==i) # get indices of cells matching component i
		sub = calls[idx] # get the indices of max means for those cells
		unique, counts = np.unique(sub, return_counts=True) # count how many counts for each argmax
		finaltypes.append(typeslist[unique[np.argmax(counts)]]) # append name of most prominant cell type in component i

	return finaltypes

def render_model(pop, name, figsizesingle):
	'''
	Render a model as a density heatmap

	Parameters
	----------
	pop : dict
		Popalign object
	name : str
		Sample name
	featuretype : str
		either 'pca' or 'NMF'
	figsizesingle : int
		size of figure
	'''
	# if name == 'unique_gmm':
	if name == 'global_gmm':
		gmm = pop['gmm']
		# C = cat_data(pop,'C')
		C = get_cat_coeff(pop)
		pcaproj = pop['pca']['proj']
		mean_labels = pop['gmm_types']
	else:
		gmm = pop['samples'][name]['gmm']
		C = get_coeff(pop, name)
		pcaproj = pop['samples'][name]['pcaproj']
		mean_labels = pop['samples'][name]['gmm_types']

	plt.figure(figsize=figsizesingle)
	pcacomps = pop['pca']['components'] # get the pca space
	cmap = 'viridis'
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
	mins = pop['pca']['mines']
	xlim = (mins[x], maxes[x])
	ylim = (mins[y], maxes[y])
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
	
	print(name)
	print(mean_labels)

	if mean_labels != [str(i) for i in range(gmm.n_components)]:
		mean_labels = ['%d: %s' % (i,lbl) for i,lbl in enumerate(mean_labels)]

	prediction = gmm.predict(C) # get the cells component assignments
	for k in range(gmm.n_components):
		idx = np.where(prediction == k)[0] # get the cell indices for component k
		if len(idx)==1:
			print('Sample %s: Component %d only contains one cell.' % (name, k))
		else: 
			sub = pcaproj[idx,0:2] # get the pca projected data for these cells
			mean = sub.mean(axis=0) # compute the mean
			cov = np.cov(sub.T) # compute the covariance matrix
			mean_proj[k,:] = mean # get the mean projected coordinates
			sample_density += w[k]*(np.reshape(mvn.pdf(pos,mean=mean_proj[k].T,cov=cov),X.shape)) # compute the density
	
	sample_density = np.log(sample_density) # log density
	
	pp = plt.pcolor(x1, x2, sample_density, cmap=cmap, vmin=cbarmin, vmax=cbarmax) # plot density
	plt.scatter(x=mean_proj[:,0], y=mean_proj[:,1], s=w_factor*w, alpha=alpha, c=mean_color) # plot means
	texts=[]
	for i,txt in enumerate(mean_labels):
		texts.append(plt.text(mean_proj[i,0], mean_proj[i,1], txt, color='white')) # plot mean labels (or numbers)
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
	plt.tight_layout()

	dname = 'renderings'
	mkdir(os.path.join(pop['output'], dname))
	name = name.replace('/','')
	plt.savefig(os.path.join(pop['output'], dname, 'model_rendering_%s.pdf' % name), dpi=200)
	plt.close()
	
	return sample_density

def grid_rendering(pop, q, figsize, samples):
	'''
	Generate a grid plot of gmm renderings

	Parameters
	----------
	pop : dict
		Popalign object
	q : list
		list of sample density arrays
	figsize : tuple, optional
		Size of the figure. Default is (10,10)
	samples : list
		Order in which to plot the sample renderings in the grid
	'''
	pcacomps = pop['pca']['components']
	cmap = 'jet'
	cmap = 'viridis'
	cbarmin = -15
	cbarmax = -3
	lims_ext = pop['pca']['lims_ext']
	nbins = 200
	x = 0
	y = 1
	row_idx = np.array([x, y])
	col_idx = np.array([x, y])
	maxes = pop['pca']['maxes']
	mins = pop['pca']['mines']
	xlim = (mins[x], maxes[x])
	ylim = (mins[y], maxes[y])
	x_ext = (xlim[1]-xlim[0])*lims_ext
	y_ext = (ylim[1]-ylim[0])*lims_ext
	xlim = (xlim[0]-x_ext, xlim[1]+x_ext)
	ylim = (ylim[0]-y_ext, ylim[1]+y_ext)
	x1 = np.linspace(xlim[0], xlim[1], nbins)
	x2 = np.linspace(ylim[0], ylim[1], nbins)

	nr, nc = nr_nc(len(samples))
	fig, axes = plt.subplots(nr,nc,figsize=figsize)
	axes = axes.flatten()
	for i, name in enumerate(samples):
		ax = axes[i]
		pp = ax.pcolor(x1, x2, q[i], cmap=cmap, vmin=cbarmin, vmax=cbarmax)
		pp.set_edgecolor('face')
		ax.set(xticks=[])
		ax.set(yticks=[])
		ax.set(title=name)

		if i % nc == 0:
			ax.set(ylabel='PC%d' % (y+1))
		if i >= len(pop['order'])-nc:
			ax.set(xlabel='PC%d' % (x+1))

	rr = len(axes)-len(samples) # count how many empty plots in grid
	for i in range(1,rr+1):
		ax = axes[-i]
		ax.axis('off') # clear empty axis from plot

	plt.suptitle('Model renderings')
	dname = 'renderings'
	mkdir(os.path.join(pop['output'], dname))
	name = name.replace('/','')
	plt.savefig(os.path.join(pop['output'], dname, 'model_rendering_%s.pdf' % 'allsamples'), dpi=200)
	plt.savefig(os.path.join(pop['output'], dname, 'model_rendering_%s.png' % 'allsamples'), dpi=200)
	plt.close()

	'''
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
	plt.savefig(os.path.join(pop['output'], dname, 'model_rendering_%s.pdf' % 'allsamples'), dpi=200)
	plt.close()
	'''

def render_models(pop, figsizegrouped, figsizesingle, samples, mode='grouped'):
	'''
	Parameters
	----------
	pop : dict
		Popalign object
	figsizegrouped : tuple
		Figure size for the grid rendering plotof all samples together
	figsizesingle : tuple
		Figure size of an individual sample rendering plot
	mode : str
		One of 'grouped', 'individual' or 'global'.
		Grouped will render the models individually and together in a separate grid
		Individual will only render the models individually
		Global will render the entire dataset's global model
	'''
	#if mode == 'unique':
		#sd = render_model(pop, 'unique_gmm', figsizesingle)
	if mode == 'global':
		sd = render_model(pop, 'global_gmm', figsizesingle)
	else:
		'''
		with Pool(pop['ncores']) as p:
			q = p.starmap(render_model, [(pop, x, figsizesingle) for x in samples])
		'''
		q = [render_model(pop, x, figsizesingle) for x in samples]
		if mode == 'grouped':
			if len(samples)>1:
				grid_rendering(pop, q, figsizegrouped, samples)
		return q

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

def build_gmms(pop, ks=(5,20), niters=3, training=0.7, nreplicates=0, reg_covar='auto', rendering='grouped', types=None, figsizegrouped=(20,20), figsizesingle=(5,5), only=None, featuretype = 'onmf'):
	'''
	Build a Gaussian Mixture Model on feature projected data for each sample

	Parameters
	----------
	pop : dict
		Popalign object
	ks : int or tuple
		Number or range of components to use
	niters : int
		number of replicates to build for each k in `ks`
	training : int or float
		If training is float, the value will be used a percentage to select cells for the training set. Must follow 0<value<1
		If training is int, that number of cells will be used for the training set.
	nreplicates : int
		Number of replicates to generate. These replicates model will be used to provide confidence intervals later in the analysis.
	reg_covar : str or float
		If 'auto', the regularization value will be computed from the feature data
		If float, value will be used as reg_covar parameter to build GMMs
	rendering : str
		Either 'grouped', 'individual' or 'global'
	types : dict, str or None
		Dictionary of cell types.
		If None, a default PBMC cell types dictionary is provided
	figsizegrouped : tuple, optional
		Size of the figure for the renderings together. Default is (20,20)
	figsizesingle : tuple, optional
		Size of the figure for each single sample rendering. Default is (5,5)
	only: list or str, optional
		Sample label or list of sample labels. Will force GMM construction for specified samples only. Defaults to None
	featuretype: str
		either 'pca' or 'onmf'
	'''

	if 'featuretype' in pop.keys():
		if featuretype != pop['featuretype']:
			raise Exception('featuretype supplied is not consistent with existing featuretype. Please reset stored gmms (reset_gmms) and try again.')
	else: 
		pop['featuretype'] = featuretype

	if isinstance(ks, tuple): # if ks is tuple
		ks = np.arange(ks[0], ks[1]) # create array of ks
	if isinstance(ks, int): # if int
		ks = [ks] # # make it a list

	if 'pca' not in pop:
		pca(pop) # build pca space if necessary

	if only != None:
		if isinstance(only,list):
			samples = only
		else:
			samples = [only]
	else:
		samples = pop['order']

	calc_reg_covar(pop)

	for i,x in enumerate(samples): # for each sample x
		print('Building model for %s (%d of %d)' % (x, (i+1), len(samples)))
		C = get_coeff(pop,x) # get sample feature data
		M = pop['samples'][x]['M'] # get sample gene data
		m = C.shape[0] # number of cells

		if (isinstance(training, int)) & (training<m) & (training > 1): # if training is int and smaller than number of cells
			n = training
		elif (isinstance(training, int)) & (training>=m): # if training is int and bigger than number of cells
			n = int(m*0.8) # since the number of training cells was larger than the number of cells in the sample, take 80%
		elif (isinstance(training, float)) & (0<training) & (training<1):
			n = int(m*training) # number of cells for the training set
		else:
			raise Exception('Value passed to training argument is invalid. Must be an int or a float between 0 and 1.')

		idx = np.random.choice(m, n, replace=False) # get n random cell indices
		not_idx = np.setdiff1d(range(m), idx) # get the validation set indices

		Ctrain = C[idx,:] # subset to get the training sdt
		Cvalid = C[not_idx,:] # subset to get the validation set

		if reg_covar == 'auto':
			reg_covar_param = pop['reg_covar'] # retrieve reg value from pop object that was computed from projection data
		else:
			reg_covar_param = reg_covar
		with Pool(pop['ncores']) as p: # build all the models in parallel
			q = p.starmap(build_single_GMM, [(k, Ctrain, reg_covar_param) for k in np.repeat(ks, niters)])

		# We minimize the BIC score of the validation set
		# to pick the best fitted gmm
		BIC = [gmm.bic(Cvalid) for gmm in q] # compute the BIC for each model with the validation set
		gmm = q[np.argmin(BIC)] # best gmm is the one that minimizes the BIC
		pop['samples'][x]['gmm'] = gmm # store gmm
		# pop['samples'][x]['gmm_means'] = np.array(gmm.means_.dot(pop['W'].T))
		pop['samples'][x]['gmm_means'] = get_gmm_means(pop,x,None)

		if types != None:
			try:
				pop['samples'][x]['gmm_types'] = typer_func(gmm=gmm, prediction=gmm.predict(C), M=M, genes=pop['genes'], types=types)
			except:
				print('Something went wrong while typing the GMM subopulations. Skipping subpopulation typing.')
				pop['samples'][x]['gmm_types'] = [str(ii) for ii in range(gmm.n_components)]
		else:
			pop['samples'][x]['gmm_types'] = [str(ii) for ii in range(gmm.n_components)]

		# Also store first gmm in ['replicates'][0]: 
		pop['samples'][x]['replicates'] = {}
		pop['samples'][x]['replicates'][0] = {}
		pop['samples'][x]['replicates'][0]['gmm'] = gmm # store gmm
		# pop['samples'][x]['replicates'][0]['gmm_means'] = np.array(gmm.means_.dot(pop['W'].T))
		pop['samples'][x]['replicates'][0]['gmm_means'] = get_gmm_means(pop, x, 0)

		# Create replicates
		pop['nreplicates'] = nreplicates # store number of replicates in pop object
		if nreplicates >=1: # if replicates are requested
			for j in range(1,nreplicates): # for each replicate number j
				idx = np.random.choice(m, n, replace=False) # get n random cell indices
				not_idx = np.setdiff1d(range(m), idx) # get the validation set indices
				Ctrain = C[idx,:] # subset to get the training sdt
				Cvalid = C[not_idx,:] # subset to get the validation set

				with Pool(pop['ncores']) as p: # build all the models in parallel
					q = p.starmap(build_single_GMM, [(k, Ctrain, reg_covar_param) for k in np.repeat(ks, niters)])
				# We minimize the BIC score of the validation set
				# to pick the best fitted gmm
				BIC = [gmm.bic(Cvalid) for gmm in q] # compute the BIC for each model with the validation set
				gmm = q[np.argmin(BIC)] # best gmm is the one that minimizes the BIC
				pop['samples'][x]['replicates'][j] = {}
				pop['samples'][x]['replicates'][j]['gmm'] = gmm # store replicate number j
				# pop['samples'][x]['replicates'][j]['gmm_means'] = gmm.means_.dot(pop['W'].T)
				pop['samples'][x]['replicates'][j]['gmm_means'] = get_gmm_means(pop, x, j)
				if types != None:
					pop['samples'][x]['replicates'][j]['gmm_types'] = typer_func(gmm=gmm, prediction=gmm.predict(C), M=M, genes=pop['genes'], types=types)
				else:
					pop['samples'][x]['replicates'][j]['gmm_types'] = [str(ii) for ii in range(gmm.n_components)]

	# print('Rendering models')
	# render_models(pop, figsizegrouped=figsizegrouped, figsizesingle=figsizesingle, samples=samples, mode=rendering) # render the models

#def build_unique_gmm(pop, ks=(5,20), niters=3, training=0.2, reg_covar=True, types=None, figsize=(6,5)):
def build_global_gmm(pop, ks=(5,20), niters=3, training=0.2, reg_covar=True, types=None, figsize=(6,5), featuretype = 'onmf'):
	'''
	Build a global Gaussian Mixture Model on the feature projected data for all samples

	Parameters
	----------
	pop : dict
		Popalign object
	ks : int or tuple
		Number or range of components to use
	niters : int
		number of replicates to build for each k in `ks`
	training : int or float
		If training is float, the value will be used a percentage to select cells for the training set. Must follow 0<value<1
		If training is int, that number of cells will be used for the training set.
	reg_covar : boolean or float
		If True, the regularization value will be computed from the feature data
		If False, 1e-6 default value is used
		If float, value will be used as reg_covar parameter to build GMMs
	types : dict, str or None
		Dictionary of cell types.
		If None, a default PBMC cell types dictionary is provided
	featuretype: str
		either 'pca' or 'onmf'
	'''
	# Set featuretype to supplied featuretype but first check the variable

	if 'featuretype' in pop.keys():
		if featuretype != pop['featuretype']:
			raise Exception('featuretype supplied is not consistent with existing featuretype. Please reset stored gmms (reset_gmms) and try again.')
	else: 
		pop['featuretype'] = featuretype

	if 'pca' not in pop:
		pca(pop) # build pca space if necessary

	if isinstance(ks, tuple): # if ks is tuple
		ks = np.arange(ks[0], ks[1]) # create array of ks
	if isinstance(ks, int): # if int
		ks = [ks] # # make it a list

	M = cat_data(pop, 'M') # get gene data
	C = get_cat_coeff(pop) # get coefficient data depending on the featuretype
	m = C.shape[0] # get training and validation sets ready

	calc_reg_covar(pop)
	
	if (isinstance(training, int)) & (training<m) & (training > 1): # if training is int and smaller than number of cells
		n = training
	elif (isinstance(training, int)) & (training>=m): # if training is int and bigger than number of cells
		n = int(m*0.8) # since the number of training cells was larger than the number of cells in the sample, take 80%
	elif (isinstance(training, float)) & (0<training) & (training<1):
		n = int(m*training) # number of cells for the training set
	else:
		raise Exception('Value passed to training argument is invalid. Must be an int or a float between 0 and 1.')

	idx = np.random.choice(m, n, replace=False) # get n random cell indices
	not_idx = np.setdiff1d(range(m), idx) # get the validation set indices

	Ctrain = C[idx,:]
	Cvalid = C[not_idx,:]

	if reg_covar == True:
		reg_covar_param = pop['reg_covar'] # retrieve reg value from pop object that was computed from projection data
	elif reg_covar == False:
		reg_covar_param = 0 # default value is 0 (no assumption on the data)
	else:
		reg_covar_param = reg_covar
	with Pool(pop['ncores']) as p:
			q = p.starmap(build_single_GMM, [(k, Ctrain, reg_covar_param) for k in np.repeat(ks, niters)])
	
	# We minimize the BIC score of the validation set
	# to pick the best fitted gmm
	BIC = [gmm.bic(Cvalid) for gmm in q]
	gmm = q[np.argmin(BIC)]
	pop['gmm'] = gmm

	if types != None:
		pop['gmm_types'] = typer_func(gmm=gmm, prediction=gmm.predict(C), M=M, genes=pop['genes'], types=types)
	else:
		pop['gmm_types'] = [str(ii) for ii in range(gmm.n_components)]
	# sd = render_model(pop, 'unique_gmm', figsize)
	sd = render_model(pop, 'global_gmm', figsize)


def calc_reg_covar(pop): 
	'''
	Compute a regularization value for covariance based on scale of coefficients

	Parameters
	----------
	pop : dict
		Popalign object
	'''

	allcoeff = get_cat_coeff(pop)

	if pop['featuretype'] == 'pca': 
		denom = 500
	elif pop['featuretype'] =='onmf':
		denom = 100
	pop['reg_covar'] = max(np.linalg.eig(np.cov(allcoeff.T))[0])/denom # store a regularization value for GMM covariance matrices


def get_gmm_means(pop, sample, rep = None): 
	'''
	Get the value of the mean for a particular sample or replicate

	Parameters
	----------
	pop : dict
		Popalign object
	figsize : tuple, optional
		Size of the figure. Default is (10,10)
	'''
	if 'featuretype' not in pop.keys():
		raise Exception('pop[\'featuretype\'] was not found. Run build_gmms which sets the featuretype for the pop object')
	featuretype = pop['featuretype']

	if featuretype == 'pca':
		W = pop['pca']['components']
	elif featuretype == 'onmf': 
		W = pop['W']
	else:
		raise Exception('featuretype variable must be: \'pca\' or \'onmf\'')

	if sample == 'global': 
		gmm = pop['gmm']
	elif rep == None: 
		gmm = pop['samples'][sample]['gmm']
	else: 
		gmm = pop['samples'][sample]['replicates'][rep]['gmm']

	# gmm_means = np.array(gmm.means_.dot(W.T)) # original
	gmm_means = np.array(gmm.means_)

	return gmm_means


'''
Build gmms using cell types
'''

def build_single_GMM_by_celltype(coeff, cell_types):
	'''
	Generate a gaussian mixture model using existing cell types

	Parameters
	----------
	k : int
		Number of components
	C : array
		Feature data
	reg_covar : float
		Regularization of the covariance matrix
	'''
	# np.random.seed()
	# gmm = smix.GaussianMixture(
	# 	n_components=k,
	# 	covariance_type='full',
	# 	tol=0.001,
	# 	reg_covar=reg_covar,
	# 	max_iter=10000,
	# 	n_init=10,
	# 	init_params='kmeans',
	# 	weights_init=None,
	# 	means_init=None,
	# 	precisions_init=None,
	# 	random_state=None,
	# 	warm_start=False,
	# 	verbose=0,
	# 	verbose_interval=10) # create model
	# return gmm.fit(C) # Fit the data

	main_types = np.unique(cell_types)
	k = len(main_types)
	
	D  = np.shape(coeff)[1]

	# extract weights_init
	counts = [cell_types.count(i) for i in main_types]
	weights_init = np.divide(counts, sum(counts))

	# extract means_init, precisions_init
	means_init = []
	precisions_init = np.zeros((k,D,D))
	for i in range(0,len(main_types)):
		currtype = main_types[i]

		idx = np.where(np.array(cell_types) == currtype)[0]
		currcoeff = coeff[idx,]

		currmean = np.mean(currcoeff,axis=0)
		currprecision = np.linalg.inv(np.cov(currcoeff.T))

		means_init.append(currmean)
		precisions_init[i,:,:] = currprecision

	means_init = np.vstack(means_init)

	gmm = smix.GaussianMixture(
		n_components=k,
		covariance_type='full',
		tol=0.001,
		reg_covar=False,
		max_iter=1,
		n_init=1,
		init_params='kmeans',
		weights_init=weights_init,
		means_init=means_init,
		precisions_init=precisions_init,
		random_state=None,
		warm_start=False,
		verbose=0,
		verbose_interval=10)

	return gmm.fit(coeff)


def build_gmms_by_celltypes(pop, ks=(5,10), only=None, rendering='grouped', figsizegrouped=(20,20), figsizesingle=(5,5), niters=3, training=0.7, nreplicates=0, reg_covar='auto',types='defaultpbmc', featuretype = 'onmf'):
	'''
	Build a Gaussian Mixture Model on feature projected data using cell type labels for each sample
	Parameters
	----------
	pop : dict
		Popalign object
	ks : int or tuple
		Number or range of components to use
	niters : int
		number of replicates to build for each k in `ks`
	training : int or float
		If training is float, the value will be used a percentage to select cells for the training set. Must follow 0<value<1
		If training is int, that number of cells will be used for the training set.
	nreplicates : int
		Number of replicates to generate. These replicates model will be used to provide confidence intervals later in the analysis.
	reg_covar : str or float
		If 'auto', the regularization value will be computed from the feature data
		If float, value will be used as reg_covar parameter to build GMMs
	rendering : str
		Either 'grouped', 'individual' or 'global'
	types : dict, str or None
		Dictionary of cell types.
		If None, a default PBMC cell types dictionary is provided
	figsizegrouped : tuple, optional
		Size of the figure for the renderings together. Default is (20,20)
	figsizesingle : tuple, optional
		Size of the figure for each single sample rendering. Default is (5,5)
	only: list or str, optional
		Sample label or list of sample labels. Will force GMM construction for specified samples only. Defaults to None
	featuretype: str
		either 'pca' or 'onmf'
	'''

	if 'featuretype' in pop.keys():
		if featuretype != pop['featuretype']:
			raise Exception('featuretype supplied is not consistent with existing featuretype. Please reset stored gmms (reset_gmms) and try again.')
	else: 
		pop['featuretype'] = featuretype

	if isinstance(ks, tuple): # if ks is tuple
		ks = np.arange(ks[0], ks[1]) # create array of ks
	if isinstance(ks, int): # if int
		ks = [ks] # # make it a list

	if 'pca' not in pop:
		pca(pop) # build pca space if necessary

	if only != None:
		if isinstance(only,list):
			samples = only
		else:
			samples = [only]
	else:
		samples = pop['order']

	calc_reg_covar(pop)

	for i,x in enumerate(samples): # for each sample x
		print('Building model for %s (%d of %d) using cell type labels' % (x, (i+1), len(samples)))

		coeff = get_coeff(pop,x)# get sample feature data
		celltypes = pop['samples'][x]['cell_type']

		try: 
			gmm = build_single_GMM_by_celltype(coeff, celltypes)
			main_types = np.unique(celltypes).tolist()

		except: 
			print('Building model for %s (%d of %d) using cell type labels didn\'t work' % (x, (i+1), len(samples)))
			print('Building model for %s (%d of %d) using gmm fitting instead and supplied parameters' % (x, (i+1), len(samples)))
			
			m = coeff.shape[0] # number of cells

			if (isinstance(training, int)) & (training<m) & (training > 1): # if training is int and smaller than number of cells
				n = training
			elif (isinstance(training, int)) & (training>=m): # if training is int and bigger than number of cells
				n = int(m*0.8) # since the number of training cells was larger than the number of cells in the sample, take 80%
			elif (isinstance(training, float)) & (0<training) & (training<1):
				n = int(m*training) # number of cells for the training set
			else:
				raise Exception('Value passed to training argument is invalid. Must be an int or a float between 0 and 1.')

			idx = np.random.choice(m, n, replace=False) # get n random cell indices
			not_idx = np.setdiff1d(range(m), idx) # get the validation set indices

			Ctrain = coeff[idx,:] # subset to get the training sdt
			Cvalid = coeff[not_idx,:] # subset to get the validation set

			if reg_covar == 'auto':
				reg_covar_param = pop['reg_covar'] # retrieve reg value from pop object that was computed from projection data
			else:
				reg_covar_param = reg_covar
			with Pool(pop['ncores']) as p: # build all the models in parallel
				q = p.starmap(build_single_GMM, [(k, Ctrain, reg_covar_param) for k in np.repeat(ks, niters)])

			# We minimize the BIC score of the validation set
			# to pick the best fitted gmm
			BIC = [gmm.bic(Cvalid) for gmm in q] # compute the BIC for each model with the validation set
			gmm = q[np.argmin(BIC)] # best gmm is the one that minimizes the BIC
		
			if types != None:
				try:
					M = pop['samples'][x]['M'] # get sample gene data
					main_types = typer_func(gmm=gmm, prediction=gmm.predict(coeff), M=M, genes=pop['genes'], types=types)
				except:
					print('Something went wrong while typing the GMM subopulations. Skipping subpopulation typing.')
					main_types = [str(ii) for ii in range(gmm.n_components)]
			else:
				main_types = [str(ii) for ii in range(gmm.n_components)]

		pop['samples'][x]['gmm'] = gmm # store gmm
		# pop['samples'][x]['gmm_means'] = np.array(gmm.means_.dot(pop['W'].T))
		pop['samples'][x]['gmm_means'] = get_gmm_means(pop,x,None)
		pop['samples'][x]['gmm_types'] = main_types
		pop['nreplicates'] = 0

	# print('Rendering models')
	# render_models(pop, figsizegrouped=figsizegrouped, figsizesingle=figsizesingle, samples=samples, mode=rendering) # render the models

def check_symmetric(mat):
	return (np.allclose(mat, mat.T))

def check_posdef(mat):
	return(np.all(np.linalg.eigvalsh(mat) > 0.))

def check_allprec(precisions):
	value = True
	for prec in precisions:
		value = value and (check_symmetric(prec) and check_posdef(prec))

'''
Entropy functions
'''
def plot_entropies(pop, figsize):
	'''
	Plot entropy values for each single gaussin density

	Parameters
	----------
	pop : dict
		Popalign object
	figsize : tuple, optional
		Size of the figure. Default is (10,10)
	'''
	yub = [pop['samples'][x]['upperbound'] for x in pop['order']]
	lbls = pop['order']
	sidx = np.argsort(yub)
	yub = [yub[i] for i in sidx]
	lbls = [lbls[i] for i in sidx]

	xvals = []
	yvals = []
	xlbls = []
	for i, x in enumerate(lbls): # for each sample
		xlbls.append(i)
		E = pop['samples'][x]['entropy'] # get list of entropy values
		xvals += [i]*len(E) # add x coords
		yvals += E # add y coords

	plt.figure(figsize=figsize)
	plt.scatter(xvals, yvals, label='Gaussian density entropy', s=2) # plot entropies of densities
	plt.plot(xlbls, yub, color='red', label='Model upperbound') # plot upper bound
	plt.xticks(xlbls, lbls, rotation=90)
	plt.ylabel('Entropy')
	plt.title('Entropies of single gaussian densities')
	plt.legend()
	
	dname = 'entropy'
	mkdir(os.path.join(pop['output'], dname))
	plt.savefig(os.path.join(pop['output'], dname, 'models_entropy.pdf'), dpi=200, bbox_inches='tight')
	plt.close()

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

def entropy(pop, figsize):
	'''
	For each sample in `pop`, compute the entropy for the gaussian densities of their Gaussian Mixture Model
	and the mixture of gaussians upper bound

	Parameters
	----------
	pop : dict
		Popalign object
	figsize : tuple, optional
		Size of the figure. Default is (10,10)
	'''
	N = pop['nfeats'] # get number of features
	for x in pop['order']: # for each sample x
		gmm = pop['samples'][x]['gmm'] # get gmm
		w = gmm.weights_ # get weights of components
		entropy = [single_entropy(N, gmm.covariances_[i]) for i in range(gmm.n_components)] # compute entropy for each gaussian density
		pop['samples'][x]['entropy'] = entropy # store entropy
		pop['samples'][x]['upperbound'] = np.sum([w[i]*(-np.log(w[i]) + entropy[i]) for i in range(gmm.n_components)]) # compute sample's entropy upperbound

	plot_entropies(pop, figsize) # generate entropy plots

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
	JD = 0.5*KL(mu1, cov1, mu2, cov2)+0.5*KL(mu2, cov2, mu1, cov1)
	return np.log10(JD)

def checkalignment(pop, refcomp, sample):
	'''
	Returns a boolean indicating whether an alignment exists between reference component and specified sample
	
	Parameters
	----------
	refcomp : int
		component number in reference sample to compare to
	sample: str
		name of string

	Output
	----------
	alignbool : bool
		indicates whether an alignment exists or not
	'''
	if sample not in pop['samples']:
		raise Exception('Sample name not valid. Use show_samples(pop) to display valid sample names.')
	try:
		arr = pop['samples'][sample]['alignments'] # get alignments between reference and test
		irow = np.where(arr[:,1] == refcomp) # get alignment that match reference subpopulation
		itest = int(arr[irow, 0]) # get test subpopulation number
		alignbool = True
	except:
		alignbool = False

	return alignbool    

def plot_deltas(pop, figsize=(10,10), sortby='mu', pthresh = 0.05): # generate plot mu and delta w plots
	'''
	Generate delta mu and delta w plots for the computed alignments

	Parameters
	----------
	pop : dict
		Popalign object
	figsize : tuple, optional
		Size of the figure. Default is (10,10)
	sortby : string
		Either 'mu' (gene expression) or 'w' (abundance)
	pthresh : float 
		p-value threshold at which colors are no longer plotted

	Outputs
	----------
	pop['deltas']: dict
		contains the following objects: 

	pop['deltas'][currtype]['combined'] : dataframe, 
		contains: 'origidx','orderedsamples', 'mean_delta_mu', 'pvals_mu','mean_delta_w','pvals_w'

	% The following should not need to be accessed directly:
	pop['deltas'][currtype]['idx'] = indices of ordered samples
	pop['deltas'][currtype]['orderedsamples'] = ordered samples in currtype
	pop['deltas'][currtype]['singles']={}
	pop['deltas'][currtype]['singles']['delta_mus'] = delta_mu for each model (including replicates)
	pop['deltas'][currtype]['singles']['delta_ws'] = delta_w for each model
	pop['deltas'][currtype]['singles']['xcoords'] = x coordinates for delta_mus 

	'''
	dname = 'deltas'
	mkdir(os.path.join(pop['output'], dname))

	ref = pop['ref'] # get reference sample name
	controlstring = pop['controlstring']
	if controlstring==None:
		raise Exception('Did not supply controlstring during load. Can be set now by executing: pop[\'controlstring\']=X')

	celltypes = pop['samples'][ref]['gmm_types']
	if celltypes == None:
		celltypes = [str(i) for i in range(pop['samples'][ref]['gmm'].n_components)]

	# Make an object that stores the orders for each cell type
	deltaobj = dict()

	for i, currtype in enumerate(celltypes): # for each reference subpopulation
		samplelbls = []
		xcoords = []
		delta_mus = []
		delta_ws = []
		mean_mus = []
		mean_ws = []
		stds_mus = []
		stds_ws = []
		# mu_ref = pop['samples'][ref]['gmm_means'][i] # get the mean i value
		mu_ref = get_gmm_means(pop, ref, None)
		w_ref = pop['samples'][ref]['gmm'].weights_[i] # get the weight i value

		k = 0
		for x in pop['order']: # for each sample x
			added = False
			tmp_delta_mus = []
			tmp_delta_ws = []
			if pop['nreplicates'] > 1: # if gmm replicates exist
				for j in range(pop['nreplicates']):
					arr = pop['samples'][x]['replicates'][j]['alignments']
					try:
						irow = np.where(arr[:,1] == i) # try to get the row where the ref comp number matches i
						itest = int(arr[irow, 0]) # get test comp number from row
						# mu_test = pop['samples'][x]['replicates'][j]['gmm_means'][itest] # get the test comp mean value
						mu_test = get_gmm_means(pop, x, j)[itest]
						w_test = pop['samples'][x]['replicates'][j]['gmm'].weights_[itest] # get the test comp weight value
						samplelbls.append(x)
						tmp_delta_mus.append(np.linalg.norm([np.array(mu_test).flatten() - np.array(mu_ref[i]).flatten()], ord='fro')) # store delta mu
						tmp_delta_ws.append((w_test - w_ref)*100) # store delta w
						xcoords.append(k)
						added = True
					except:
						pass

			if x != ref: # if x is not the reference
				arr = pop['samples'][x]['alignments'] # get the alignments between x and the reference
				try:
					irow = np.where(arr[:,1] == i) # try to get the row where the ref comp number matches i
					itest = int(arr[irow, 0]) # get test comp number from row
					# mu_test = pop['samples'][x]['gmm_means'][itest] # get the test comp mean value
					mu_test = get_gmm_means(pop, x, None)[itest]
					w_test = pop['samples'][x]['gmm'].weights_[itest] # get the test comp weight value
					samplelbls.append(x) # store test sample label x
					tmp_delta_mus.append(np.linalg.norm([np.array(mu_test).flatten() - np.array(mu_ref[i]).flatten()], ord='fro')) # store delta mu
					tmp_delta_ws.append((w_test - w_ref)*100) # store delta w
					xcoords.append(k)
					added = True
				except:
					pass

			if added == True:
				k += 1
				delta_mus += tmp_delta_mus
				delta_ws += tmp_delta_ws
				mean_mus.append(np.mean(tmp_delta_mus))
				mean_ws.append(np.mean(tmp_delta_ws))
				stds_mus.append(np.std(tmp_delta_mus))
				stds_ws.append(np.std(tmp_delta_ws))

		seen = set()
		seen_add = seen.add
		xlbls = [x for x in samplelbls if not (x in seen or seen_add(x))]
		x = [x for x in xcoords if not (x in seen or seen_add(x))]

		control_delta_mus = [mean_mus[i] for i in x if controlstring in xlbls[i]]
		control_delta_ws = [mean_ws[i] for i in x if controlstring in xlbls[i]]

		# Calculate the p-values for the means and abundances
		if len(control_delta_mus)>1:
			pvals_mus, control_mus_CI_min,control_mus_CI_max = calc_p_value(control_delta_mus, mean_mus, 1)
			pvals_ws,control_ws_CI_min,control_ws_CI_max = calc_p_value(control_delta_ws, mean_ws, 2)
		else:
			pvals_mus, control_mus_CI_min,control_mus_CI_max = calc_p_value(control_delta_mus, mean_mus, 1)
			pvals_mus = np.ones(len(pvals_mus))
			pvals_ws,control_ws_CI_min,control_ws_CI_max = calc_p_value(control_delta_ws, mean_ws, 2)
			pvals_ws = np.ones(len(pvals_ws))

		# Max/Min of bootstrapped measurements
		control_delta_mus_min = min([delta_mus[i] for i in range(len(delta_mus)) if controlstring in xlbls[xcoords[i]]])
		control_delta_mus_max = max([delta_mus[i] for i in range(len(delta_mus)) if controlstring in xlbls[xcoords[i]]])

		control_delta_ws_min = min([delta_ws[i] for i in range(len(delta_ws)) if controlstring in xlbls[xcoords[i]]])
		control_delta_ws_max = max([delta_ws[i] for i in range(len(delta_ws)) if controlstring in xlbls[xcoords[i]]])

		# reorder data 
		if sortby=="mu": 
			idx = np.argsort(mean_mus)
		elif sortby=="w": 
			idx = np.argsort(mean_ws)
		else: 
			raise Exception("Sortby must be either mu (gene expression) or w (abundance)")
		xlbls = [xlbls[i] for i in idx]
		mean_mus = [mean_mus[i] for i in idx]
		mean_ws = [mean_ws[i] for i in idx]
		stds_mus = [stds_mus[i] for i in idx]
		stds_ws = [stds_ws[i] for i in idx]
		pvals_mus = [pvals_mus[i] for i in idx]
		pvals_ws= [pvals_ws[i] for i in idx]

		xcoords = [np.where(idx==value)[0][0] for value in xcoords]

		# Only plot colors for adjusted p-values < pthresh
		plot_pval_ws=[x if x < pthresh else 1 for x in pvals_ws]
		plot_pval_mus=[x if x < pthresh else 1 for x in pvals_mus]

		rbcmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","red"])
		plt.figure(figsize=figsize)

		# Plot the delta ws - upper
		ax1 = plt.subplot(2,1,1)
		plt.title('Reference sample %s\nComponent %d: %s' %(ref, i, currtype))
		plt.scatter(xcoords, delta_ws, s=2, c='k')
		plt.scatter(x, mean_ws, s=36, c =-np.log10(plot_pval_ws) ,cmap=rbcmap, label = 'mean Δ\u03C9 (%)')
		plt.errorbar(x, mean_ws, stds_ws, color='k', elinewidth=.5, capsize=1, fmt=' ')
		# Plot the control ranges:
		plt.hlines([control_delta_ws_min,control_delta_ws_max], -1, len(mean_ws), colors='k', linestyles='dotted', label = ' control min/max')   
		plt.xticks([])
		plt.ylabel('Δ\u03C9 (%)')
		# Only plot CI and p-values if we can calculate them
		if len(control_delta_ws)>1 : 
			plt.fill_between(range(-1,len(mean_ws)+1),control_ws_CI_min,control_ws_CI_max,alpha=0.2, color='black', label = 'control CI')
			cbar=plt.colorbar()
			cbar.set_label('-log10(p-val)', rotation=90)
		else : 
			plt.clim(0,0)
		plt.legend()

		# Plot the delta mus - lower
		plt.subplot(2,1,2)
		plt.scatter(xcoords, delta_mus, s=2, c='k')
		plt.scatter(x, mean_mus, s=36,  c = -np.log10(plot_pval_mus), cmap=rbcmap, label = 'mean Δ\u03BC')
		plt.errorbar(x, mean_mus, stds_mus, color='k', elinewidth=.3, capsize=1, fmt=' ')
		plt.hlines([control_delta_mus_min,control_delta_mus_max], -1, len(mean_mus), colors='k', linestyles='dotted', label = 'control min/max')
		plt.xticks(x, xlbls, rotation=90)
		plt.ylabel('Δ\u03BC')
		# Only plot CI and p-values if we can calculate them
		if len(control_delta_mus)>1 : 
			plt.fill_between(range(-1,len(mean_mus)+1),control_mus_CI_min,control_mus_CI_max,alpha=0.2, color='black', label = 'control CI')
			cbar=plt.colorbar()
			cbar.set_label('-log10(p-val)', rotation=90)
		else : 
			plt.clim(0,0)
		plt.legend()

		plt.tight_layout()
		currtype = currtype.replace('/','')
		plt.rc('font', size= 12) 
		plt.savefig(os.path.join(pop['output'], dname, 'deltas_comp%d_%s_%ssort.pdf' % (i,currtype, sortby)), format='pdf', bbox_inches='tight')
		plt.close()

		# Combine mean data together into a single dataframe
		t = pd.DataFrame(np.array([idx,xlbls, mean_mus, pvals_mus, mean_ws, pvals_ws]),
			index=['origidx','orderedsamples', 'mean_delta_mu', 'pvals_mu','mean_delta_w','pvals_w'])      
		t = pd.DataFrame.transpose(t);
		deltaobj[currtype]={}
		deltaobj[currtype]['idx'] = idx
		deltaobj[currtype]['orderedsamples'] = xlbls
		deltaobj[currtype]['singles']={}
		deltaobj[currtype]['singles']['delta_mus'] = delta_mus
		deltaobj[currtype]['singles']['delta_ws'] = delta_ws
		deltaobj[currtype]['singles']['xcoords'] = xcoords
		deltaobj[currtype]['combined'] = t # table of data

	pop['deltas'] = deltaobj

def aligner(refgmm, testgmm, method):
	'''
	Align the components of two models

	Parameters
	----------
	refgmm : sklearn.mixture.GaussianMixture
		Reference model
	testgmm	: sklearn.mixture.GaussianMixture
		Test model
	method : str
		Alignment method. Must be one of: test2ref, ref2test, conservative

	Output
	----------
	arr: array, int
		pairwise JD values between two GMMs
	res: array, int
		Array giving best alignments. Leftmost square matrix:  'boolean' shows associated pairs 
		Rightmost column: JD values for those associated pairs 
	'''
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

	if method not in ['test2ref', 'ref2test', 'conservative']:
		raise Exception('method must be one of: test2ref, ref2test, conservative')
	if method == 'test2ref':
		minsidx = np.argmin(arr, axis=1) # get idx of closest ref mixture for each test mixture
		mins = np.min(arr, axis=1) # get min divergence values
		res = np.zeros((ltest, 3))
		for i in range(ltest):
			res[i,:] = np.array([i, minsidx[i], mins[i]])

	elif method == 'ref2test':
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
	return res, arr

def align(pop, ref=None, method='conservative', figsizedeltas=(10,10), figsizeentropy=(10,10)):
	'''
	Align the components of each sample's model to the components of a reference model

	Parameters
	----------
	pop : dict
		Popalign object
	ref : str
		Name of reference sample
	method : str
		Method to perform the alignment
		If conservative, the reference component and the test component have to be each other's best match to align
		If test2ref, the closest reference component is found for each test component
		If ref2test, the closest test component is found for each test component
	figsizedeltas : tuple, optional
		Size of the figure for the delta plot. Default is (10,5)
	figsizeentropy : tuple, optional
		Size of the figure for the entropy plot. Default is (10,5)
	'''
	if ref == None:
		raise Exception('Please provide sample id of reference')
	elif ref not in pop['samples']:
		raise Exception('Provided reference not in sample list.\nYou can print the list of available samples with show_samples()')

	pop['ref'] = ref # store reference sample name

	refgmm = pop['samples'][ref]['gmm'] # get reference gmm
	for x in pop['order']: # for each sample x
		if pop['nreplicates'] >= 1: # if replicates exist
			for j in range(pop['nreplicates']): # for each replicate j
				testgmm = pop['samples'][x]['replicates'][j]['gmm'] # grab replicate gmm
				alignments, arr = aligner(refgmm, testgmm, method) # align that replicate to reference model
				pop['samples'][x]['replicates'][j]['alignments'] = alignments
				pop['samples'][x]['replicates'][j]['fullalignments'] = arr

		# if x != ref: # if sample is not ref
		# for all samples, re-align the 'main' gmm at the upper level
		testgmm = pop['samples'][x]['gmm'] # get test gmm
		alignments, arr = aligner(refgmm, testgmm, method) # align gmm to reference
		pop['samples'][x]['alignments'] = alignments
		pop['samples'][x]['fullalignments'] = arr
		try:
			pop['samples'][x]['test2ref'] = np.zeros(testgmm.n_components, dtype=int)
			pop['samples'][x]['ref2test'] = np.zeros(testgmm.n_components, dtype=int)
		except:
			pass

	# plot_deltas(pop, figsizedeltas) # generate plot mu and delta w plots
	# entropy(pop, figsizeentropy)


# Assign sample cells to reference gmm components and compute deltas - This section needs work
'''
Parameters
----------
'''

# def plot_deltas_aa(pop, ref, deltamus, deltaws, figsize, pointsize): 
#     labels = pop['order'] # get sample labels
#     complbls = pop['samples'][ref]['gmm_types'] # get reference component labels
    
#     dname = 'deltas_assign_align'
#     mkdir(os.path.join(pop['output'], dname))
    
#     for i,lbl in enumerate(complbls): # for each component i
#         ydmus = deltamus[i,:] # get the delta mu values for component i
#         ydws = deltaws[i,:] # get the delta w values for component i
#         idx = np.where(ydmus==0)[0] # get indices of deltas mus with a value of 0
#         ydmus = np.delete(ydmus, idx) # remove values by index
#         ydws = np.delete(ydws, idx) # remove values by index
#         xlabels = [labels[ii] for ii in range(pop['nsamples']) if ii not in idx] # remove labels by index
#         x = np.arange(len(ydmus)) # create x coordinates vector
#         idx = np.argsort(ydmus) # get indices of sorted delta mu values
#         ydmus = ydmus[idx] # reorder delta mu values
#         ydws = ydws[idx] # reorder delta w values
#         xlabels = [labels[ii] for ii in idx] # reorder label values
        
#         plt.figure(figsize=figsize)
#         ax1 = plt.subplot(2,1,1)
#         plt.title('Reference sample %s\nComponent %d: %s' %(ref, i, lbl))
#         plt.scatter(x, ydws, s=pointsize, c='k')
#         plt.xticks([])
#         plt.ylabel('Δ\u03C9 (%)')

#         plt.subplot(2,1,2)
#         plt.scatter(x, ydmus, s=pointsize, c='k')
#         plt.xticks(x, xlabels, rotation=90)
#         plt.ylabel('Δ\u03BC')

#         plt.tight_layout()
#         lbl = lbl.replace('/','')
#         plt.savefig(os.path.join(pop['output'], dname, 'deltas_comp%d_%s.pdf' % (i,lbl)), dpi=200, bbox_inches='tight')
#         plt.close()

# def assign_align(pop, ref=None, figsize=(15,15), pointsize=10):
#     if ref == None:
#         raise Exception('Please provide sample id of reference')
#     elif ref not in pop['samples']:
#         raise Exception('Provided reference not in sample list.\nYou can print the list of available samples with show_samples()')
#     pop['ref'] = ref # assign reference
#     gmm = pop['samples'][ref]['gmm'] # retrieve reference gmm
#     deltamus = np.zeros((gmm.n_components,pop['nsamples']), dtype=float) # array to store delta mu values
#     deltaws = np.zeros((gmm.n_components,pop['nsamples']), dtype=float) # array to store delta w values
#     # W = pop['W'] # retrieve W matrix to project feature means later
#     W = get_features(pop) #retrieve W matrix to project feature means later

#     for j,x in enumerate(pop['order']): # for each sample
#         if x != ref: # if sample x is different from reference sample
#             # C = pop['samples'][x]['C'] # retrieve feature data for sample x
#             C = get_coeff(pop,sample)
#             prediction = gmm.predict(C) # get cells assignments 
#             pop['samples'][x]['prediction'] = prediction # save cell assignments
#             ntestcells = C.shape[0] # get total number of cells for sample x
#             for i in range(gmm.n_components): # for each reference component i
#                 try:
#                     idx = np.where(prediction==i)[0] # get test cells indices that match reference component #i
#                     sub = C[idx,:] # subset test feature data with above indices
#                     mean = sub.mean(axis=0) # get cloud mean
#                     # mu_test = np.array(mean.dot(W.T)).flatten() # compute test component mu in filtered gene space
#                     # mu_ref = pop['samples'][ref]['gmm_means'][i] # get ref component mu in filtered gene space
#                     mu_ref = get_gmm_means(pop, ref, None)[i]
#                     mu_test = mean[i]

#                     w_test = len(idx)/ntestcells # compute test cloud w
#                     w_ref = gmm.weights_[i] # get reference component i's w
#                     deltamu = np.linalg.norm([np.array(mu_test).flatten() - np.array(mu_ref).flatten()], ord='fro') # compute delta mu between ref component and test cloud
#                     deltaw = (w_test - w_ref)*100 # compute delta w
#                     deltamus[i,j] = deltamu # store delta mu value
#                     deltaws[i,j] = deltaw # store delta w value
#                 except:
#                     pass
#     plot_deltas_aa(pop, ref, deltamus, deltaws, figsize, pointsize) # plot deltas

'''
Rank functions
'''
def LL(x, mu, sigma, k):
	tmp = x-mu
	ll = -.5*(np.log(np.linalg.det(sigma)) + tmp.T.dot(np.linalg.inv(sigma).dot(tmp)) + k*np.log(2*np.pi))
	return ll

def score_subpopulations(pop, ref=None, figsize=(10,5)):
	order = np.array(pop['order']) # get order of samples
	if ref not in order: # check that ref is in the order list
		raise Exception('ref value not a valid sample name')
	order = order[order != ref] # remove ref label from order list
	gmm = pop['samples'][ref]['gmm'] # retrive reference gmm
	k = pop['nfeats'] # retrieve dimensionality

	for i,subpop in enumerate(pop['samples'][ref]['gmm_types']): # for each subpopulation in the ref gmm
		mu = gmm.means_[i] # get matching mean vector
		sigma = gmm.covariances_[0] # get matching covariance matrix
		data = [] # empty list to store the samples cells log-likelihoods
		means = [] # empty list to store the samples log-likelihood means 
		for x in order: # for each test sample
			C = get_coeff(pop, x) # retrive the feature space data (dimensionality k)
			LLS = np.array([LL(v,mu,sigma,k) for v in C]) # compute the LL for each cell for a given gaussian density
			data.append(LLS) # store the LL values
			means.append(LLS.mean()) # store the matching mean

		idx = np.argsort(means) # sort means
		lblorder = order[idx] # sample labels in sort means order

		lbls = np.concatenate([[lbl]*len(data[i]) for i,lbl in enumerate(order)])
		scores = np.concatenate(data)
		df = pd.DataFrame({'scores': scores, 'labels': lbls})

		# create stripplot using the computed order based on score means
		plt.figure(figsize=figsize)
		ax = sns.stripplot(x="labels", y="scores", data=df, order=lblorder, palette='tab20', size=.5)
		
		#x = plt.xlim()
		#plt.fill_between(x, min_, max_, alpha=0.1, color='black')
		# adjusting plot labels
		x = range(len(lblorder))
		plt.xticks(x, lblorder, rotation=90)
		plt.xlabel('Samples')
		plt.ylabel('Log-likelihood scores')
		plt.title('Likelihood scores against reference\nSubpopulation #%d %s' % (i,subpop))
		plt.tight_layout()
		dname = 'ranking_test'
		mkdir(os.path.join(pop['output'], dname))
		plt.savefig(os.path.join(pop['output'], dname, 'rankings_%d_%s.pdf' % (i,subpop)), dpi=200)
		plt.close()
'''
RANK
'''
def rank(pop, ref=None, k=100, niter=200, method='LLR', mincells=50, figsize=(10,5)):
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
    method : str
        Scoring method to use. 'LLR' for log-likelihood ratio, 'LL' for log-likelihood.
    mincells : int
        If a sample has less than `mincells` cells, is discarded
    figsize : tuple, optional
        Size of the figure. Default is (10,5)
        
    Outputs
    ----------
    pop['rankings']: dataframe
        Add dataframe of samples sorted by meanLLR including p-values

    '''
    # For a given sample S, k random cells are scored against the reference
    # model. This process is repeated niter times and the results are 
    # shown as boxplots
    scores = []
    lbls = []
    gmmctrl = pop['samples'][ref]['gmm']
    controlstring = pop['controlstring']

    for x in pop['order']:
        C = get_coeff(pop,x)
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
            # keep track of scores, labels and sample(drug) classes
            if x == ref:
                try:
                    gmmtest = pop['samples'][x]['replicates'][0]['gmm']
                except:
                    gmmtest = pop['samples'][x]['gmm']
            else:
                gmmtest = pop['samples'][x]['gmm']
            for _ in range(niter):
                idx = np.random.choice(m, nk, replace=False)
                sub = C[idx,:]
                if method == 'LLR':
                    scores.append(gmmctrl.score(sub) - gmmtest.score(sub)) # for Log Likelihood Ratio LLR
                elif method == 'LL':
                    scores.append(gmmctrl.score(sub))
                else:
                    raise Exception('method must be one of: LLR, LL')
                lbls.append(x)
        else:
            print('Not enough cells for samples: %s' % x)

    # create dataframe from scores, labels and sample (i.e. drug) classes
    # find sample order based on score means
    df = pd.DataFrame({'scores': scores, 'labels': lbls})
    if method == 'LL':
        df.scores = df.scores - df.scores.max()
    df2 = pd.DataFrame({col:vals["scores"] for col, vals in df.groupby("labels")})
    means = df2.mean().sort_values()
    lblorder = means.index.values.tolist()

    x = range(len(lblorder))

    # Calculate the 95 % confidence interval for 
    # the control samples. Then plot grey bar to
    # designate the 95% CI
    controls = [i for i in pop['samples'].keys() if controlstring in i]
    control_means = [means.get(x) for x in controls]
    mean_ = sum(control_means)/len(control_means)
    std_ = np.std(control_means)
    CI_ = 1.96*std_/np.sqrt(len(control_means))
    min_ = mean_ -(abs(CI_)) 
    max_ = mean_ +(abs(CI_)) 
    
    # Calculate the p-value for each sample
    z = (means - mean_)/(std_*np.sqrt(len(control_means)))
    pvals = stats.norm.sf(abs(z))

    # FDR correction
    ranked_pvals = rankdata(pvals)
    pvals_new = pvals * len(pvals) / ranked_pvals
    pvals_new[pvals_new > 1] = 1

    # Make dataframe with means and p-values
    means_df = pd.Series.to_frame(means)
    pvals_df = pd.DataFrame(pvals_new)
    pvals_df.index = means_df.index
    final_df = pd.concat([means_df,pvals_df],axis=1)
    final_df.columns = ['meanLLR','pval']

    if method=='LL':
        ylabel = 'Log-likelihood scores'
        title = 'Log-likelihood scores against reference model (%s)' % ref
    else:
        ylabel = 'Log-likelihood ratio scores'
        title = 'Log-likelihood ratio scores against reference model (%s)' % ref

    # create boxplot using the computed order based on score means
    plt.figure(figsize=figsize)

    boxprops = dict(linestyle='-', linewidth=0.5)
    flierprops = dict(marker='D', markersize=2, linestyle='none')
    ax = sns.boxplot(x="labels", y="scores", data=df, order=lblorder, palette='tab20',
                     boxprops=boxprops, whiskerprops=boxprops, medianprops=boxprops,capprops=boxprops, flierprops = flierprops)
    x = plt.xlim()
    plt.fill_between(x, min_, max_, alpha=0.2, color='black')
    # adjusting plot labels
    x = range(len(lblorder))
    plt.xticks(x, lblorder, rotation=90)
    plt.xlabel('Samples')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.rc('font', size= 12) 
    dname = 'ranking'
    mkdir(os.path.join(pop['output'], dname))
    plt.savefig(os.path.join(pop['output'], dname, '%s_rankings_boxplot.png' % method), dpi=200)
    plt.savefig(os.path.join(pop['output'], dname, '%s_rankings_boxplot.pdf' % method))
    plt.close()

    # create stripplot using the computed order based on score means
    plt.figure(figsize=figsize)
    ax = sns.stripplot(x="labels", y="scores", data=df, order=lblorder, palette='tab20', size=2)
    x = plt.xlim()
    plt.fill_between(x, min_, max_, alpha=0.2, color='black')
    # adjusting plot labels
    x = range(len(lblorder))
    plt.xticks(x, lblorder, rotation=90)
    plt.xlabel('Samples')
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.title('Sample scores against %s sample\n(For each sample: %d random cells %d times)' % (ref, k, niter))
    plt.tight_layout()
    dname = 'ranking'
    mkdir(os.path.join(pop['output'], dname))
    plt.savefig(os.path.join(pop['output'], dname, '%s_rankings_stripplot.png' % method), dpi=200)
    plt.savefig(os.path.join(pop['output'], dname, '%s_rankings_stripplot.pdf' % method))
    plt.close()

    pop['rankings'] = final_df

'''
Query functions
'''

def cluster_rows(X, metric='correlation', method='complete'):
	'''
	Cluster the rows of an array X

	Parameters
	----------
	X : ndarray
		The rows of X will be clustered
	metric : str
		Metric to build pairwise distance matrix
	method : str
		Method to use for hierarchical clustering
	'''
	c = pairwise_distances(X=X,metric=metric,n_jobs=-1) # create pairwise distance matrix
	np.fill_diagonal(c,0.) # make sure diagonal is not rounded
	c = scd.squareform(c,force='tovector',checks=False) # force distance matrix to vector
	c = np.nan_to_num(c) 
	z = fc.linkage(c, method=method) # create linkage
	z = z.clip(min=0)
	return shc.leaves_list(z)

def plot_query(pop, pcells=.2, nreps=10, figsize=(10,20), sharey=True):
	'''
	Plot proportions of the samples in each of the GMM components

	Parameters
	----------
	pop: dict
		Popalign object
	pcells: float
		Percentage of random cells to use for each query repetition
	nreps: int
		Number of times to repeat the query process with different cells for each sample
	figsize : tuple, optional
		Size of the figure. Default is (5,20)
	'''

	#ncells = 1000

	gmm = pop['gmm'] # get global gmm
	N = len(pop['order']) # get number of samples
	arrmus = np.zeros((N, gmm.n_components)) # empty array to store the proportion means of each sample for all components
	arrstds = np.zeros((N, gmm.n_components)) # matching array to store the matching standard deviations

	for i,x in enumerate(pop['order']): # for each sample x
		C = get_coeff(pop,x) # get feature data
		ncells = int(C.shape[0]*pcells) # compute number of random cells to select from percentage value
		concat = [] # list to store the proportion arrays for each rep
		for rn in range(nreps): # for each repetition rn
			tmp = np.zeros(gmm.n_components) # create empty proportion array
			idx = np.random.choice(C.shape[0], ncells, replace=False) # get random indices of ncells cells
			X = C[idx,:] # subset C witht the random indices
			prediction = gmm.predict(X) # get the component assignments for cells in subset X
			unique, counts = np.unique(prediction, return_counts=True) # count how many cells fall in each component
			counts = np.array([x/ncells for x in counts]) # transform counts to percentage values
			d = dict(zip(unique, counts)) # zip the percentage values with their respective componen
			for j in range(gmm.n_components): # for each component j
				try:
					tmp[j] = d[j] # try to assign the matching percentage value if it exists to the tmp array
				except:
					pass
			concat.append(tmp) # add tmp proportion array to concat list
		concat = np.vstack(concat) # stack all proportion arrays for sample x 
		arrmus[i,:] = concat.mean(axis=0) # get proportion means (one mean per component) for sample x
		arrstds[i,:] = concat.std(axis=0) # get standard deviations (one std per component)
		
	arrmus = arrmus.T # components are rows, samples are columns
	arrstds = arrstds.T # components are rows, samples are columns
	xvals = np.arange(N) # generate x coordinationes (number of samples)
	components = [pop['gmm_types'][i] for i in np.arange(gmm.n_components)] # retrieve components labels (after cell typing)

	# cluster samples and components
	idx_cols = cluster_rows(arrmus.T) # get indices of clustered samples (columns)
	idx_rows = cluster_rows(arrmus) # get indices of clustered commponents (rows)
	arrmus = arrmus[:,idx_cols] # reorder columns of means array
	arrmus = arrmus[idx_rows,:] # reorder rows of means array
	arrstds = arrstds[:,idx_cols] # reorder columns of std array
	arrstds = arrstds[idx_rows,:] # reorder rows of std array
	clusteredlbls = [pop['order'][i] for i in idx_cols] # reordered sample labels to match clustered samples

	fig, axes = plt.subplots(nrows=arrmus.shape[0], ncols=1, sharex=True, sharey=sharey, figsize=figsize) # create subplots

	for i, (mus,stds) in enumerate(zip(arrmus,arrstds)): # for each component i and the %age values in yvals
		ax = axes[i] # get sub axes
		ax.plot(xvals, mus, color='k') # plot proportions
		ax.fill_between(xvals, mus-stds, mus+stds, alpha=.2, color='k') # plot ribbon based on standard deviation from mu
		axes[i].set(ylabel='Ref. comp %d\n%s' % (idx_rows[i],components[idx_rows[i]])) # set y label (component cell type)

		if i == gmm.n_components-1: # if last component
			plt.xticks(xvals, clusteredlbls, rotation=90) # set sample names as x tick labels
			axes[i].set(xlabel=' Samples')
			
	plt.tight_layout()

	dname = 'query'
	mkdir(os.path.join(pop['output'], dname))
	path_pdf = os.path.join(pop['output'], dname, 'query_plot.pdf')
	path_png = os.path.join(pop['output'], dname, 'query_plot.png')
	plt.savefig(path_pdf, bbox_inches='tight')
	plt.savefig(path_png, dpi=200, bbox_inches='tight')
	plt.close()

def plot_query_heatmap(pop, figsize=(10,10)):
	'''
	Plot heatmap samples x components matrix (how many % of cells of sample i in component j)
	
	Parameters
	----------

	pop: dict
		Popalign object
	figsize : tuple, optional
		Size of the figure. Default is (10,10)
	'''
			
	# for each sample
	# get sample cells and get respective gmm component predictions
	# get percentage of cells per component
	gmm = pop['gmm']
	N = len(pop['order'])
	arr = np.zeros((N, gmm.n_components)) 
	for i,x in enumerate(pop['order']):
		C = get_coeff(pop, x)
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
		figsize=figsize
	)

	# rotate xlabels
	plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

	# adjusting labels
	g.ax_heatmap.set_xlabel('GMM components')
	plt.suptitle('Samples proportions in GMM components (%)', size=20)
	plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

	dname = 'query'
	mkdir(os.path.join(pop['output'], dname))
	path_ = os.path.join(pop['output'], dname, 'query_heatmap.pdf')
	plt.savefig(path_, bbox_inches='tight')
	plt.close()



'''
Visualization functions
'''
def plot_heatmap(pop, refcomp, genelist, clustersamples=True, clustercells=True, savename=None, figsize=(15,15), cmap='Purples', samplelimits=False, scalegenes=False, only=None, equalncells=False):
	'''
	Plot specific genes for cells of a reference subpopulation S and subpopulations that aligned to S
	
	Parameters
	----------
	pop : dict
		Popalign dict
	refcomp : int
		Reference subpopulation number
	genelist : list
		List of genes to plot
	clustersamples : bool
		Cluster the samples
	clustercells : bool
		Cluster the cells within each sample subpopulation
	savename : str, optional
		The user can specify a name for the file to be written. When savename is None, a filename is computed with the reference component number. Default is None
	figsize : tuple, optional
		Size of the figure. Default is (15,15)
	cmap : str, optional
		Name of the Matplotlib colormap to use. Default is Purples
	samplelimits : bool, optional
		Whether to draw vertical lines on the heatmap to visually separate cells from different samples
	scalegenes : bool, optional
		Whether to scale the genes by substracting the min and dividing by the max for each gene
	'''
	genelist = [g for g in genelist if g in pop['genes']] # only keep valid genes
	gidx = [np.where(pop['genes']==g)[0][0] for g in genelist] # get indices for those genes

	ref = pop['ref'] # get reference sample label
	reftype = pop['samples'][ref]['gmm_types'][refcomp]
	C = get_coeff(pop,ref)# get reference data in feature space
	M = pop['samples'][ref]['M'][gidx,:] # get reference data in gene space, subsample genes
	prediction = pop['samples'][ref]['gmm'].predict(C) # get cell predictions
	idx = np.where(prediction == refcomp)[0] # get indices of cells in component #refcomp
	M = M[:,idx] # get cells from gene space data

	cmetric='correlation' # cluster metric
	cmethod='single' # cluster method
	if clustercells == True:
		cidx = cluster_rows(M.toarray().T, metric=cmetric, method=cmethod) # cluster cells of subpopulation
		M = M[:,cidx] # reorder matrix

	MS = [M] # create list of matrices, with the reference matrix as the first element
	MSlabels = ['%s (%d)' % (ref,refcomp)] # create list of sample labels, with the reference label as the first element
	ncols = [M.shape[1]] # create list of sample cell numbers, with the number of reference cells as the first element
	means = [M.mean(axis=1)]

	if only == None:
		for x in pop['order']: # for each sample in pop
			if x != pop['ref']: # if that sample is not the reference sample
				try: # check if an aligned subpopulation exists for that sample
					arr = pop['samples'][x]['alignments'] # retrive test sample alignments
					irow = np.where(arr[:,1] == refcomp)[0] # get row number in alignments where ref subpop is the desired ref subpop

					if len(irow)==1:
						itest = int(arr[irow, 0]) # get test subpopulation number if exists
						C = get_coeff(pop,x) # get test sample feature space data
						prediction = pop['samples'][x]['gmm'].predict(C) # get the subpopulations assignments
						idx = np.where(prediction == itest)[0] # get indices of cells that match aligned test subpopulation
						M = pop['samples'][x]['M'][gidx,:] # get test sample gene space data, subsample
						M = M[:,idx] # select test subpopulation cells
						if clustercells == True:
							cidx = cluster_rows(M.toarray().T, metric=cmetric, method=cmethod) # cluster cells of subpopulation
							M = M[:,cidx] # reorder matrix
						ncols.append(M.shape[1])
						MS.append(M) # append to list
						means.append(M.mean(axis=1))
						MSlabels.append('%s (%d)' % (x,itest)) # append matching sample label to list
					else:
						for itest in irow:
							C = get_coeff(pop,x) # get test sample feature space data
							prediction = pop['samples'][x]['gmm'].predict(C) # get the subpopulations assignments
							idx = np.where(prediction == itest)[0] # get indices of cells that match aligned test subpopulation
							M = pop['samples'][x]['M'][gidx,:] # get test sample gene space data, subsample
							M = M[:,idx] # select test subpopulation cells
							if clustercells == True:
								cidx = cluster_rows(M.toarray().T, metric=cmetric, method=cmethod) # cluster cells of subpopulation
								M = M[:,cidx] # reorder matrix
							ncols.append(M.shape[1])
							MS.append(M) # append to list
							means.append(M.mean(axis=1))
							MSlabels.append('%s (%d)' % (x,itest)) # append matching sample label to list
				except:
					pass
	else:
		try:
			x = only
			arr = pop['samples'][x]['alignments'] # retrieve test sample alignments
			irow = np.where(arr[:,1] == refcomp)[0] # get row number in alignments where ref subpop is the desired ref subpop
			if len(irow)==1:
				itest = int(arr[irow, 0]) # get test subpopulation number if exists
				C = get_coeff(pop,x) # get test sample feature space data
				prediction = pop['samples'][x]['gmm'].predict(C) # get the subpopulations assignments
				idx = np.where(prediction == itest)[0] # get indices of cells that match aligned test subpopulation

				M = pop['samples'][x]['M'][gidx,:] # get test sample gene space data, subsample
				M = M[:,idx] # select test subpopulation cells
				if clustercells == True:
					cidx = cluster_rows(M.toarray().T, metric=cmetric, method=cmethod) # cluster cells of subpopulation
					M = M[:,cidx] # reorder matrix
				ncols.append(M.shape[1])
				MS.append(M) # append to list
				means.append(M.mean(axis=1))
				MSlabels.append('%s (%d)' % (x,itest)) # append matching sample label to list
			else:
				for itest in irow:
					C = get_coeff(pop,x) # get test sample feature space data
					prediction = pop['samples'][x]['gmm'].predict(C) # get the subpopulations assignments
					idx = np.where(prediction == itest)[0] # get indices of cells that match aligned test subpopulation

					M = pop['samples'][x]['M'][gidx,:] # get test sample gene space data, subsample
					M = M[:,idx] # select test subpopulation cells
					if clustercells == True:
						cidx = cluster_rows(M.toarray().T, metric=cmetric, method=cmethod) # cluster cells of subpopulation
						M = M[:,cidx] # reorder matrix
					ncols.append(M.shape[1])
					MS.append(M) # append to list
					means.append(M.mean(axis=1))
					MSlabels.append('%s (%d)' % (x,itest)) # append matching sample label to list	
		except:
			pass

	if equalncells == True:
		n = np.min(ncols)
		for iii,m_ in enumerate(MS):
			idx = np.random.choice(m_.shape[1], n, replace=False) 
			MS[iii] = m_[:,idx]
			ncols[iii] = n

	if clustersamples == True: # if cluster == True
		means = np.hstack(means) # stack means together horizontally
		l = cluster_rows(means.T) # cluster mean vectors
		MSlabels = [MSlabels[ii] for ii in l] # reorder labels
		MS = [MS[ii] for ii in l] # reorder matrices
		ncols = [ncols[ii] for ii in l] # reorder number of cells

	M = ss.hstack(MS) # create full matrix
	M = M.toarray() # to dense
	if scalegenes == True:
		# tmp = (M.T-M.min(axis=1)).T # subtract min
		tmp = M # Do not subtract the min
		M = (tmp.T/tmp.max(axis=1)).T # divide by max

	cols = np.concatenate([[0]*x if i%2==0 else [1]*x for i,x in enumerate(ncols)]) # create binary vector to color columns, length equals to number of cells. Should be: [0,...,0,1,...,1,0,...,0,1...,1,etc]
	cols = cols.reshape(1,len(cols)) # reshape vector to plot it as a heatmap
	xtickscoords = [x/2 for x in ncols] # calculate label tick offset to center label
	cumsum = np.cumsum(ncols) # compute cumulative sum of bins 
	for i,(x,y) in enumerate(zip(ncols,xtickscoords)):
		if i!=0:
			xtickscoords[i] += cumsum[i-1] # update x tick coordinates with cumulative sum

	fig = plt.figure(1,figsize=figsize) # create figure with given figure size
	nr = 20 # number of rows in plot grid
	nc = 20 # number of cols in plot grid
	gridspec.GridSpec(nr,nc) # create plot grid

	# heatmap
	plt.subplot2grid((nr,nc), (0,0), colspan=nc, rowspan=nr-1) # create subplot for heatmap, leave space for column colors
	plt.imshow(M, aspect='auto', interpolation='none', cmap=cmap) # plot heatmap
	plt.yticks(np.arange(len(genelist)),genelist) # display gene names
	plt.xticks([]) # remove x ticks
	plt.title('Reference sample: %s\nSubpopulation #%d: %s' % (ref, refcomp, pop['samples'][ref]['gmm_types'][refcomp]))
	
	if samplelimits == True: # if parameter is True
		for xx in cumsum[:-1]: # for each limit
			plt.axvline(xx, color='k') # plot vertical line

	# col colors
	plt.subplot2grid((nr,nc), (nr-1, 0), colspan=nc, rowspan=1) # create subplot for column colors
	plt.imshow(cols, aspect='auto', cmap='binary') # plot column colors
	plt.yticks([]) # remove y ticks
	plt.xticks(xtickscoords, MSlabels, rotation=90) # display sample names
	
	if only != None:
		# dname = 'diffexp/%d_%s_%s/' % (refcomp, reftype, only) # define directory name
		dname = 'diffexp/refpop%d_%s_%s/' % (refcomp, reftype, only)
	else:
		dname = 'heatmaps' # define directory name
		mkdir(os.path.join(pop['output'], dname)) # create directory if needed
	if savename != None:
		filename = savename
	else:
		filename = 'comp%d_heatmap' % refcomp
	filename = filename.replace('/','')
	plt.savefig(os.path.join(pop['output'], dname, '%s.pdf' % filename),bbox_inches='tight')
	plt.close()

def plot_genes_gmm_cells(pop, sample='', genelist=[], savename='', metric='correlation', method='single', clustergenes=True, clustercells=True, cmap='magma', figsize=(10,15)):
	'''
	Plot a heatmap of genes ~ GMM subpopulations cells for a given model

	Parameters
	----------
	pop : dict
		Popalign object
	sample : str
		Sample name to select model from pop dictionary
	genelist : list
		List of gene names. If empty, the filtered genes will be used.
	savename : str
		File name to use
	metric : str
		Metric to use to cluster
	method : str
		Method to use to cluster
	clustergenes : boolean
		Whether or not to cluster the genes (rows)
	cmap : str
		Name of the colormap to use
	figsize : tuple
		Figure size
	'''
	if genelist == []: 
		datatype = 'M_norm' # if no gene list, use filtered genes data
	else:
		datatype = 'M' # if gene list, use non filtered data to extract genes

	if sample == 'global': # if the user wants to access the global model
		gmm = pop['gmm'] # get global gmm
		M = cat_data(pop,datatype) # get data in gene space from all samples
		# C = cat_data(pop,'C') # get data in feat space from all samples
		C = get_cat_coeff(pop)
		columns = pop['gmm_types'] # get the GMM subpopulation labels
	elif sample in pop['order']: # if the user wants to access the gmm of a given sample
		gmm = pop['samples'][sample]['gmm'] # get sample's gmm
		M = pop['samples'][sample][datatype] # get sample data in gene space
		C = get_coeff(pop, sample)# get sample data in feat space
		columns = pop['samples'][sample]['gmm_types'] # get the GMM subpopulation labels
	else: 
		raise Exception('sample should be `global` or a valid sample name.')
	columns = ['%d: %s' % (i,lbl) for i,lbl in enumerate(columns)]
	genes = pop['genes'] # get gene names

	if genelist != []: # if gene list is specified
		genelist = [g for g in genelist if g in genes] # only keep valid gene names
		gidx = [np.where(genes==g)[0][0] for g in genelist] # get gene indices
		M = M[gidx,:] # subset genes from data

	prediction = gmm.predict(C) # get subpopulation assignments for all the cells
	cols = [] # empty list to store averages
	ncols = [] # empty list to store number of cells per subpopulation
	MS = [] # empty list to store matrices
	for i in range(gmm.n_components): # for each subpopulation i from GMM
		idx = np.where(prediction==i)[0] # get indices of cells that match subpopulation i 
		sub = M[:,idx] # subset cells from data
		ncols.append(sub.shape[1])
		if clustercells == True:
			cidx = cluster_rows(sub.toarray().T, method=method, metric=metric)
			sub = sub[:,cidx]
		MS.append(sub)

	MS = ss.hstack(MS).toarray() # concatenate matrices
	cols = np.concatenate([[0]*x if i%2==0 else [1]*x for i,x in enumerate(ncols)]) # create binary vector to color columns, length equals to number of cells. Should be: [0,...,0,1,...,1,0,...,0,1...,1,etc]
	cols = cols.reshape(1,len(cols)) # reshape vector to plot it as a heatmap
	xtickscoords = [x/2 for x in ncols] # calculate label tick offset to center label
	cumsum = np.cumsum(ncols) # compute cumulative sum of bins 
	for i,(x,y) in enumerate(zip(ncols,xtickscoords)):
		if i!=0:
			xtickscoords[i] += cumsum[i-1] # update x tick coordinates with cumulative sum

	if genelist == []:
		genelist = pop['filtered_genes']
	if clustergenes == True:
		cidx = cluster_rows(MS, method=method, metric=metric)
		genelist = [genelist[i] for i in cidx]
		MS = MS[cidx,:]

	fig = plt.figure(1,figsize=figsize) # create figure with given figure size
	nr = 20 # number of rows in plot grid
	nc = 20 # number of cols in plot grid
	gridspec.GridSpec(nr,nc) # create plot grid

	cmap='magma'
	# heatmap
	plt.subplot2grid((nr,nc), (0,0), colspan=nc, rowspan=nr-1) # create subplot for heatmap, leave space for column colors
	plt.imshow(MS, aspect='auto', interpolation='none', cmap=cmap) # plot heatmap
	plt.yticks(np.arange(len(genelist)),genelist) # display gene names
	plt.xticks([]) # remove x ticks
	plt.title('%s GMM' % sample)

	# col colors
	plt.subplot2grid((nr,nc), (nr-1, 0), colspan=nc, rowspan=1) # create subplot for column colors
	plt.imshow(cols, aspect='auto', cmap='binary') # plot column colors
	plt.yticks([]) # remove y ticks
	plt.xticks(xtickscoords, columns, rotation=90) # display sample names

	dname = 'heatmaps' # define directory name
	mkdir(os.path.join(pop['output'], dname)) # create directory if needed
	if savename == '':
		savename = 'gmm_%s' % sample # default name if none is provided
	savename = savename.replace('/','')
	plt.savefig(os.path.join(pop['output'], dname, '%s_cells.pdf' % savename), dpi=200, bbox_inches='tight')
	plt.close()

def scatter(pop, method='tsne', sample=None, compnumber=None, marker=None, size=.3, extension='pdf', cmap='Blues',samplecolor='red'):	
	'''
	Run an embedding algorithm and plot the data in a scatter plot

	pop: dict
		Popalign object
	method : str
		Embedding method. One of umap, tsne. Defaults to umap
	marker : str
		Either `samples` or a valid gene symbol. Defaults to None
	size : float or int
		Point size. Defaults to .1
	extension: string
		File extension like 'png' or 'pdf' to designate how to save the plot
	'''

	if method not in pop: # if method not run before
		# X = cat_data(pop, 'C') # retrieve feature space data
		X = get_cat_coeff(pop)
		if method == 'umap': # if method is umap
			X = umap.UMAP().fit_transform(X) # run umap
		elif method == 'tsne': # if method is tsne
			X = TSNE(n_components=2).fit_transform(X) # run tsne
		elif method == 'pca': # if method is pca
			pca(pop) # build pca space if necessary
		else: # if method not valid
			raise Exception('Method value not supported. Must be one of tsne, umap.') # raise exception
		pop[method] = X # store embedded coordinates
	else: # if method has been run before
		if method == 'pca':
			X = pop[method]['proj']
		else:
			X = pop[method] # retrieve embedded coordinates
		
	if sample != None: # if a sample is provided
		M = pop['samples'][sample]['M'] # get matrix
		idx = pop['samples'][sample]['indices'] # get cells indices of that sample
		start = idx[0] # get start index
		end = idx[1] # get end index
		xsample = X[start:end,0] # subset embedded coordinates
		ysample = X[start:end,1] # subset embedded coordinates
		
		if compnumber != None:
			gmm = pop['samples'][sample]['gmm']
			C = get_coeff(pop, sample)
			prediction = gmm.predict(C)
			idx = np.where(prediction==compnumber)[0]
			M = M[:,idx] # only keep cells from component
			xsample = xsample[idx]
			ysample = ysample[idx]
	else:
		M = cat_data(pop,'M')
		
	if marker:
		try:
			ig = np.where(pop['genes']==marker)[0][0] # get gene index if valid gene name
			c = M[ig,:].toarray().flatten() # get expression values of gene
		except:
			raise Exception('Gene name not valid') # raise exception if gene name not valid
	else:
		c = samplecolor
		cmap = None
	
	if not sample:
		plt.scatter(X[:,0], X[:,1], c=c, cmap=cmap, s=size)
	else:
		plt.scatter(X[:,0], X[:,1], c='lightgrey', cmap=cmap, s=size)
		plt.scatter(xsample, ysample, c=c, cmap=cmap, s=size*3)
	
	plt.xticks([]) # remove x ticks
	plt.yticks([]) # remove y ticks
	plt.xlabel('%s 1' % method) # x label
	
	plt.ylabel('%s 2' % method) # y label
	
	# update filename and plot title
	title = '%s plot' % method
	filename = '%s' % method
	if marker:
		title += ', marker: %s' % marker
		filename += '_%s' % marker
		plt.colorbar() # display colorbar
	if sample != None:
		title += '\n%s' % sample
		filename += '_%s' % sample
	if compnumber != None:
		title += ' Component #%d' % compnumber
		filename += '_comp%d' % compnumber
	plt.title(title)
	
	dname = 'embedding/markers/' # directory name
	mkdir(os.path.join(pop['output'], dname)) # create directory if needed
	plt.savefig(os.path.join(pop['output'], dname, '%s.%s' % (filename, extension)), dpi=200, bbox_inches='tight') # save figure
	plt.close()

def samples_grid(pop, method='tsne', figsize=(20,20), size_background=.1, size_samples=.3, samplecolor='purple'):
	'''
	Generate a grid plot of sample plots in an embedding space

	Parameters
	----------
	pop : dict
		Popalign object
	method : str
		Embedding method. One of tsne, umap
	figsize : tuple
		Figure size
	size_background : float, int
		Point size for the embedding scatter in the background
	size_samples : float, int
		Point size for the highlighted samples
	'''

	if method not in pop: # if method not run before
		X = get_cat_coeff(pop) # retrieve feature space data
		if method == 'umap': # if method is umap
			X = umap.UMAP().fit_transform(X) # run umap
		elif method == 'tsne': # if method is tsne
			X = TSNE(n_components=2).fit_transform(X) # run tsne
		elif method == 'pca': # if method is pca
			pca(pop) # build pca space if necessary
		else: # if method not valid
			raise Exception('Method value not supported. Must be one of tsne, umap.') # raise exception
		pop[method] = X # store embedded coordinates
	else: # if method has been run before
		if method == 'pca':
			X = pop[method]['proj']
		else:
			X = pop[method] # retrieve embedded coordinates

	x = X[:,0] # get x coordinates
	y = X[:,1] # get y coordinates

	nr, nc = nr_nc(len(pop['order'])) # based on number of samples, get number of rows and columns for grid plot
	fig, axes = plt.subplots(nr,nc,figsize=figsize) # create figure and sub axes
	axes = axes.flatten()
	start = 0 # start index to retrive cells for a given sample
	end = 0 # end index to retrive cells for a given sample
	for i, name in enumerate(pop['order']): # for each sample
		ax = axes[i] # assign sub axis
		end = start+pop['samples'][name]['M'].shape[1] # adjust end index with number of cells
		xsub = x[start:end] # splice x coordinates
		ysub = y[start:end] # splice y coordinates
		start = end # update start index
		ax.scatter(x, y, c='lightgrey', s=size_background) # plot all cells as background
		ax.scatter(xsub, ysub, c=samplecolor, s=size_samples) # plot sample cells on top
		ax.set(xticks=[]) # remove x ticks
		ax.set(yticks=[]) # remove y ticks
		ax.set(title=name) # sample name as title

		if i % nc == 0:
			ax.set(ylabel='%s2' % method) # set y label
		if i >= len(pop['order'])-nc:
			ax.set(xlabel='%s1' % method) # set x label

	rr = len(axes)-len(pop['order']) # count how many empty plots in grid
	for i in range(1,rr+1): # 
		ax = axes[-i] # backtrack extra sub axes
		ax.axis('off') # clear empty axis from plot

	plt.suptitle('Samples in t-SNE space') # set main title
	dname = 'embedding/samples/' # folder name
	mkdir(os.path.join(pop['output'], dname)) # create folder if doesn't exist
	plt.savefig(os.path.join(pop['output'], dname, 'embedding_grid_%s.png' % method), dpi=200) # save plot
	plt.close() # close plot

def subpopulations_grid(pop, method='tsne', figsize=(20,20), size_background=.1, size_subpops=1):
	'''
	Generate grid plots of sample subpopulations in an embedding space

	Parameters
	----------
	pop : dict
		Popalign object
	method : str
		Embedding method. One of tsne, umap
	figsize : tuple
		Size of the figures to be generated
	size_background : float, int
		Point size for the embedding scatter in the background
	size_subpops : float, int
		Point size for the highlighted subpopulations
	'''

def subpopulations_grid_global(pop, method='tsne', figsize=(20,20), size_background=.1, size_subpops=1):
	if method not in pop: # if method not run before
		# X = cat_data(pop, 'C') # retrieve feature space data
		X = get_cat_coeff(pop) # retrieve feature space data
		if method == 'umap': # if method is umap
			X = umap.UMAP().fit_transform(X) # run umap
		elif method == 'tsne': # if method is tsne
			X = TSNE(n_components=2).fit_transform(X) # run tsne
		elif method == 'pca': # if method is pca
			pca(pop) # build pca space if necessary
		else: # if method not valid
			raise Exception('Method value not supported. Must be one of tsne, umap.') # raise exception
		pop[method] = X # store embedded coordinates
	else: # if method has been run before
		if method == 'pca':
			X = pop[method]['proj']
		else:
			X = pop[method] # retrieve embedded coordinates

	x = X[:,0] # get x coordinates
	y = X[:,1] # get y coordinates

	start = 0 # start index to subset sample cells
	end = 0 # end index to subset sample cells

#	C = cat_data(pop, 'C')
	C = get_cat_coeff(pop)
	gmm = pop['gmm']
	poplabels = pop['gmm_types']
	prediction = gmm.predict(C) # get subpopulation assignments for the cells

	nr, nc = nr_nc(gmm.n_components) # get number of rows and columns for the grid plot
	fig, axes = plt.subplots(nr,nc,figsize=figsize) # create figure and subaxes
	axes = axes.flatten()

	for i in range(gmm.n_components): # for each subpopulation of sample
		ax = axes[i] # assign sub axis
		idx = np.where(prediction==i)[0] # get cell indices for that subpopulations
		xtmp = x[idx] # subset sample's cells
		ytmp = y[idx] # subset sample's cells
		ax.scatter(x, y, c='lightgrey', s=size_background) # plot all cells as background
		ax.scatter(xtmp, ytmp, c='purple', s=size_subpops) # plot subpopulation cells on top
		ax.set(xticks=[]) # remove x ticks
		ax.set(yticks=[]) # remove y ticks
		ax.set(title='Subpopulation #%d\n%s' % (i, poplabels[i])) # set title
		if i % nc == 0:
			ax.set(ylabel='%s2' % method) # set y label
		if i >= len(pop['order'])-nc:
			ax.set(xlabel='%s1' % method) # set x label

	rr = len(axes)-gmm.n_components # count how many empty plots in grid
	for i in range(1,rr+1):
		ax = axes[-i] # backtrack extra sub axes
		ax.axis('off') # clear empty axis from plot

	dname = 'embedding/subpopulations/' # folder name
	mkdir(os.path.join(pop['output'], dname)) # create folder if does not exist
	plt.savefig(os.path.join(pop['output'], dname, 'global_subpopulations_%s.png' % method), dpi=200) # save plot
	plt.close() # close plot

def reset_gmms(pop):
	'''
	This function deletes the following keys from the pop dictionary:

	pop['gmm']
	pop['gmmtypes']
	pop['featuretype']
	pop['samples'][sample]['gmm']

	Parameters
	----------
	pop : dict
		Popalign object
	sample : str
		sample name
	dname : str
		directory name

	'''
	try: 
		del pop['gmm']
		print('deleted: pop[\'gmm\']')
	except: 
		print('not present: pop[\'gmm\']')

	try: 
		del pop['gmmtypes']
		print('deleted: pop[\'gmmtypes\']')
	except: 
		print('not present: pop[\'gmmtypes\']')

	try: 
		del pop['featuretype']
		print('deleted: pop[\'featuretype\']')
	except: 
		print('not present: pop[\'featuretype\']')

	for i in range(0, len(pop['samples'].keys())):
		sample = pop['order'][i]

		try: 
			del pop['samples'][sample]['gmm']
			print('deleted gmm for sample: ' + sample)			
		except: 
			print('no gmm for sample: ' + sample)


def plot_L1_heatmap(pop, sample, dname,cmap='RdBu'):

	'''
	Plots a heatmap of L1norm values for significant differentially expressed genes 
	across all cell types for a specific sample. Genes are organized in order of 
	1) shared genes
	2) celltype specific 

	Parameters
	----------
	pop : dict
		Popalign object
	sample : str
		sample name
	dname : str
		directory name

	'''
	if 'diffexp' not in pop: 
		raise Exception ('Differential expression analysis has not been run yet')

	deobj = pop['diffexp']
	celltypes = list(set(pop['diffexp']['de_df'].celltype))
	celltypes.insert(0, celltypes.pop(celltypes.index('shared'))) # put shared first
	genes = deobj[celltypes[1]]['all_genes'] # don't use index 0['shared'] which does not have a separate entry in the object
	
	allsamples =deobj[celltypes[1]]['all_samples']
	sidx = allsamples.index(sample)

	# xref = pop['ref'] # get reference sample label
	# currtype = celltypes[refcomp]
	# ncomps = pop['samples'][xref]['gmm'].n_components-1

	# dual_impact_samples = ['Alprostadil', 'Loteprednol etabonate', 'Budesonide', 'Betamethasone Valerate', 'Triamcinolone Acetonide', 'Meprednisone']
	# dname = 'diffexp/'
	# celltypes = list({'Monocytes','T cells'})

	# genes = deobj[celltypes[0]]['all_genes']
	samplelist=deobj['de_df']['sample']
	smdf = deobj['de_df'][samplelist==sample]
	combogenes = []
	for i in range(len(celltypes)) :
		currtype = celltypes[i]
		# get upregulated genes
		currgenes = smdf['genes'][smdf['celltype']==currtype]
		currgenes = ','.join(currgenes)
		currgenes = currgenes.split(',')
		combogenes = combogenes+currgenes

	combogenes = list(dict.fromkeys(combogenes)) # Remove duplicates

	ri = [] # row (gene) indexes 
	for i in range(len(combogenes)) :
		curridx = np.where(genes==combogenes[i])[0]
		if len(curridx)>1:
			ri.append(np.asscalar(curridx[0])) # only keep the first gene

	ri = np.asarray(ri)
	ri = ri.astype(int) # convert to integer

	# Concatenate all cell types together to get single matrix of L1 norms 
	comboM = []
	for i in range(1,len(celltypes)) : # don't start with 0 because that's 'shared'
		currtype = celltypes[i]
		# get upregulated genes
		currM = deobj[currtype]['all_l1norm'][sidx]
		currthresh = deobj[currtype]['cutoff']
		currM[abs(currM)<currthresh] = 0
		if i==1:
			comboM = currM;
		else:
			comboM = np.vstack((comboM,currM))
	comboM = comboM.T # transpose to get the matrix oriented correctly

	mkdir(os.path.join(pop['output'], dname)) # create directory if needed

	fig=plt.figure(figsize=(5, 6), dpi= 80)
	plt.imshow(comboM [ri,:], aspect='auto', interpolation='none', cmap=cmap,vmin=-1,vmax=1) # plot heatmap
	plt.yticks(np.arange(len(ri)), genes[ri],fontsize=6) # display gene names
	plt.xticks(np.arange(len(celltypes)-1),celltypes[1:],rotation=90,fontsize=8) # remove x ticks
	cbar=plt.colorbar()
	cbar.set_label('L1-error', rotation=90,fontsize=14)
	plt.title(sample)
	plt.ylabel('ngenes')
	plt.tight_layout()
	plt.savefig(os.path.join(pop['output'], dname, '%s_degenes_L1_heatmap.pdf' % sample))
	plt.close()

def plot_violins(pop, refcomp, samples, plotgenes, prefix, **kwargs):
	'''
	Plot violin plots of gene distributions for all samples that align
	to a specified component from the reference sample

	Parameters
	----------
	pop : dict
	    Popalign object
	refcomp : int
	    Subpopulation number of the reference sample's GMM
	samples : str
	    list of samples to compare, sets the order for plotting
	genes : str
	    list of genes to pull out
	prefix : str
		filename prefix for all plots
	**kwargs : 
	    arguments that are be passed on to seaborn.violinplot        
	'''
	# start of file
	xref = pop['ref'] # get reference sample label
	currtype = pop['samples'][xref]['gmm_types'][refcomp]
	if not(set(samples).issubset(pop['order'])) : 
		raise Exception('Sample names not valid. Use show_samples(pop) to display valid sample names.')
    
	genes = pop['genes']
	dname = 'violins/'
	mkdir(os.path.join(pop['output'], dname)) # create directory if needed

	for i in range(len(plotgenes)):
		currgene = plotgenes[i]
		gidx = np.where(genes==currgene)[0]

		arrlist = []
		lblslist = []
		for j in range(len(samples)):
			xtest = samples[j] # test sample label        
			try:
				arr = pop['samples'][xtest]['alignments'] # get alignments between reference and test
				irow = np.where(arr[:,1] == refcomp) # get alignment that match reference subpopulation
				itest = int(arr[irow, 0]) # get test subpopulation number
			except:
				raise Exception('Could not retrieve a matching alignment between sample %s and reference component %d' % (xtest, refcomp))
			Mtest = pop['samples'][xtest]['M']
			predictiontest = pop['samples'][xtest]['gmm'].predict(get_coeff(pop,xtest)) # get test cell assignments
			idxtest = np.where(predictiontest==itest)[0] # get matching indices
			subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest
			subtest = subtest.toarray() # from sparse matrix to numpy array for slicing efficiency

			currarray = subtest[gidx,:]
			currarray = currarray.tolist()
			labels = [xtest] * np.shape(currarray)[1]
			arrlist.append(currarray)
			lblslist.append(labels)

		v1 = np.concatenate(arrlist,axis=1)
		v1 = v1[0]
		v2 = np.concatenate(lblslist)
		fakey = [1]*(len(v1))

		arrdf = pd.DataFrame(data = list([v1[:],v2]))
		arrdf = pd.DataFrame.transpose(arrdf)
		arrdf.rename(columns = {0:'values',1:'sample'},inplace=True)
		arrdf['values']=arrdf['values'].astype('float64')
		arrdf['sample']=arrdf['sample'].astype('category')
		arrdf['y'] = fakey
		arrdf['y'] = arrdf['y'].astype('float64')

		# Determine number of columns 
		if len(samples)>3:
			ncols = 2
		else:
			ncols = 1
        
		plt.figure(figsize=(3,3));
		plt.rc('font',size=12)
		ax=sns.violinplot(y='values' ,x='y',data=arrdf,hue='sample',split=False, orient='v',**kwargs)
		ax.set_ylabel('Normalized log(counts)',fontsize=12)
		ax.set_xlabel('')
		ax.set_xticks([])
		plt.title(currgene, fontsize=24)
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=ncols, fontsize=10)

		filename = '%s_%s_%s' % (prefix,currtype, currgene)
		plt.savefig(os.path.join(pop['output'], dname, '%s.pdf' % filename),bbox_inches='tight')
		plt.close()

def plot_violin_entirepop(pop, samples, plotgenes, prefix, **kwargs):
	'''
	Plot violin plots of gene distributions for the entire population across samples

	Parameters
	----------
	pop : dict
	    Popalign object
	samples : str
	    list of samples to compare, sets the order for plotting
	genes : str
	    list of genes to pull out
	prefix : str
	    filename prefix for all plots
	**kwargs : 
	    arguments that are be passed on to seaborn.violinplot        
	'''
	# start of file
	xref = pop['ref'] # get reference sample label
	if not(set(samples).issubset(pop['order'])) : 
		raise Exception('Sample names not valid. Use show_samples(pop) to display valid sample names.')

	genes = pop['genes']
	dname = 'violins/'
	mkdir(os.path.join(pop['output'], dname)) # create directory if needed

	for i in range(len(plotgenes)):
		currgene = plotgenes[i]
		gidx = np.where(genes==currgene)[0]

		arrlist = []
		lblslist = []
		for j in range(len(samples)):
			xtest = samples[j] # test sample label        
			Mtest = pop['samples'][xtest]['M']
			subtest = Mtest # subset cells that match subpopulation itest
			subtest = subtest.toarray() # from sparse matrix to numpy array for slicing efficiency

			currarray = subtest[gidx,:]
			currarray = currarray.tolist()
			labels = [xtest] * np.shape(currarray)[1]
			arrlist.append(currarray)
			lblslist.append(labels)

		v1 = np.concatenate(arrlist,axis=1)
		v1 = v1[0]
		v2 = np.concatenate(lblslist)
		fakey = [1]*(len(v1))

		arrdf = pd.DataFrame(data = list([v1[:],v2]))
		arrdf = pd.DataFrame.transpose(arrdf)
		arrdf.rename(columns = {0:'values',1:'sample'},inplace=True)
		arrdf['values']=arrdf['values'].astype('float64')
		arrdf['sample']=arrdf['sample'].astype('category')
		#         arrdf['sample'].cat.categories = samples  # enforces original ordering in plots
		arrdf['y'] = fakey
		arrdf['y'] = arrdf['y'].astype('float64')

		# Determine number of columns 
		if len(samples)>3:
			ncols = 2
		else:
			ncols = 1

		plt.figure(figsize=(3,3));
		plt.rc('font',size=12)
		ax=sns.violinplot(y='values' ,x='y',data=arrdf,hue='sample',split=False, orient='v',**kwargs)
		ax.set_ylabel('Normalized log(counts)',fontsize=12)
		ax.set_xlabel('')
		ax.set_xticks([])
		plt.title(currgene, fontsize=24)
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=ncols, fontsize=10)

		filename = '%s_entirepop_%s' % (prefix, currgene)
		plt.savefig(os.path.join(pop['output'], dname, '%s.pdf' % filename),bbox_inches='tight')
		plt.close()

def plot_ribbon_ngenes(pop, samples = None, prefix='all_samples',toplot = 'ngenes', sortby = 'ngenes',colors = None, **kwargs):
	'''
	Plots a ribbon plot showing the number of genes that have changed across all cell types 

	Parameters
	----------
	pop : dict
	    Popalign object
	samples : str
	    list of samples to compare
	    if samples is empty, we use all samples from the pop object
	prefix : str
	    prefix of saved plot name
	toplot : str
	    options are 'ngenes'(the number of genes) or 'perc' (percentage of genes). Default: ngenes
	sortby : str
	    options are 'ngenes'(total number of genes affected) or 'orig'(keep original order). Default: ngenes
	colors: list
	    list of hex colors, must be more than the number of cell types
	**kwargs : 
	    arguments that are passed onto df.plot.area()
	'''
	if samples == None:
		samples = list(pop['samples'].keys())
	if colors == None:
		colors = sns.color_palette('muted')
	if prefix == None: 
		prefix = ''
	    
	dname = 'diffexp/'
	celltypes = list(set(pop['diffexp']['de_df'].celltype))
	celltypes.insert(0, celltypes.pop(celltypes.index('shared'))) # put shared first

	# check to make sure that we have enough colors to plot all cell types
	if len(colors) < len(celltypes): 
		raise Exception('Must supply as many colors as celltypes')

	# check to make sure that the differential expression boject has already been created in pop
	if 'diffexp' not in pop.keys():
		raise Exception('Must run all_samples_diffexp before calling this function')

	deobj = pop['diffexp']

	allns = np.zeros(len(celltypes))
	for i in range(len(samples)):
		xtest = samples[i]

		didx = list(pop['samples'].keys()).index(xtest)
		smdf = deobj['de_df'][deobj['de_df']['sample']==xtest]

		# Start sample vector
		samplens = []
		for i in range(len(celltypes)) :
			currtype = celltypes[i]
			if currtype == 'shared': 
				currn = np.sum(smdf[smdf['celltype']==currtype]['ngenes'])
			else:
				currn = np.sum(smdf[smdf['celltype']==currtype]['ngenes']) - np.sum(smdf[smdf['celltype']=='shared']['ngenes'])
			samplens.append(np.asscalar(currn))
		allns = np.column_stack((allns,samplens))

	print(allns)
	print(celltypes)

	# Transform array into percentages
	allns = np.transpose(allns)
	allns = np.delete(allns, obj=0, axis=0)
	totgenes = np.sum(allns,axis=1)
	allperc = allns / totgenes[:,None]

	# Sort rows by percentage shared
	if sortby == 'orig':
		colidx = range(0,len(samples))
	elif sortby == 'ngenes':
		colidx = np.argsort(totgenes)
	sortedsamples = [samples[i] for i in colidx]

	if toplot == 'perc':
		df = pd.DataFrame(data = allperc[colidx,:], columns = celltypes, index = sortedsamples)
	elif toplot == 'ngenes':
		df = pd.DataFrame(data = allns[colidx,:], columns = celltypes, index = sortedsamples )

	fig = plt.figure(figsize=(6,3), dpi=200)
	df.plot.area(colors = colors, **kwargs)
	plt.xticks(np.arange(len(samples)),sortedsamples,rotation=90,fontsize=14) # remove x ticks
	plt.legend(fontsize=14,loc='upper left')
	plt.ylabel('ngenes')
	plt.savefig(os.path.join(pop['output'], dname, '%s_ngenes_ribbon.pdf' % prefix),bbox_inches='tight')
	plt.close()

'''
Differential expression functions
'''
def l1norm(ig, arr1, arr2, nbins):
	'''
	Compute the l1-norm between two histograms
	
	Parameters
	----------
	ig : int
		Index of gene
	sub1 : sparse matrix
		Matrix of first subpopulation
	sub2 : sparse matrix
		Matrix of second subpopulation
	nbins : int
		Number of histogram bins to use
	'''
	max1, max2 = np.max(arr1), np.max(arr2) # get max values from the two subpopulations
	max_ = max(max1,max2) # get max value to define histogram range
	if max_ == 0:
		return 0
	else:
		b1, be1 = np.histogram(arr1, bins=nbins, range=(0,max_)) # compute histogram bars
		b2, be2 = np.histogram(arr2, bins=nbins, range=(0,max_)) # compute histogram bars
		b1 = b1/len(arr1) # scale bin values
		b2 = b2/len(arr2) # scale bin values
		if arr1.mean()>=arr2.mean(): # sign l1-norm value based on mean difference
			return -np.linalg.norm(b1-b2, ord=1)
		else:
			return np.linalg.norm(b1-b2, ord=1)

def diffexp(pop, refcomp=0, testcomp=0, sample='', nbins=20, cutoff=.5, renderhists=True, usefiltered='filtered', equalncells=True, figsize=(20,20)):
	'''
	Find  differentially expressed genes between a reference subpopulation
	and the subpopulation of a sample that aligned to it

	Parameters
	----------
	pop : dict
		Popalign object
	refcomp : int
		Subpopulation number of the reference sample's GMM
	testcomp : int
		Subpopulation number of the test sample's GMM
	sample : str
		Name of the sample to compare
	nbins : int, optional
		Number of histogram bins to use
	nleft : int
		Number of underexpressed genes to retrieve
	nright : int
		Number of overexpressed genes to retrieve
	renderhists : bool
		Render histograms or not for the top differentially expressed genes
	usefiltered : str
		Either 'filtered', 'unfiltered', or 'refilter'. Default: 'filtered'	
	'''
	xref = pop['ref'] # get reference sample label
	reftype = pop['samples'][xref]['gmm_types'][refcomp]
	ncomps = pop['samples'][xref]['gmm'].n_components-1

	if sample not in pop['order']:
		raise Exception('Sample name not valid. Use show_samples(pop) to display valid sample names.')
	if refcomp > ncomps:
		raise Exception('Component number too high. Must be between 0 and %d' % ncomps)

	xtest = sample # test sample label
	arr = pop['samples'][xtest]['alignments'] # get alignments between reference and test
	if [testcomp, refcomp] not in arr[:,:-1].tolist():
		raise Exception('Could not retrieve a matching alignment for sample %s, between test component %d and reference component %d' % (sample, testcomp, refcomp))

	'''
	try:
		arr = pop['samples'][xtest]['alignments'] # get alignments between reference and test
		print(arr)
		irow = np.where(arr[:,1] == refcomp)[0] # get alignment that match reference subpopulation
		itest = int(arr[irow, 0]) # get test subpopulation number
		#print(itest)
	except:
		raise Exception('Could not retrieve a matching alignment between sample %s and reference component %d' % (sample, refcomp))
	'''

	predictionref = pop['samples'][xref]['gmm'].predict(get_coeff(pop,xref)) # get ref cell assignments
	predictiontest = pop['samples'][xtest]['gmm'].predict(get_coeff(pop,xtest)) # get test cell assignments

	idxref = np.where(predictionref==refcomp)[0] # get matching indices
	idxtest = np.where(predictiontest==testcomp)[0] # get matching indices

	if usefiltered == 'filtered':
		Mref = pop['samples'][xref]['M_norm'] # get filtered reference sample matrix
		Mtest = pop['samples'][xtest]['M_norm'] # get filtered test sample matrix
		genes = np.array(pop['filtered_genes']) # get filtered gene labels
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest

	elif usefiltered == 'unfiltered': # Use just nonzero genes
		nzidx = pop['nzidx']
		Mref = pop['samples'][xref]['M'][nzidx,:] # get reference sample matrix
		Mtest = pop['samples'][xtest]['M'][nzidx,:]  # get test sample matrix
		genes = np.array(pop['genes']) # get gene labels
		genes = genes[nzidx] 
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest

	elif usefiltered =='refilter':
		# Only keep the genes that are present in >10% of cells
		nzidx = pop['nzidx']
		Mref = pop['samples'][xref]['M'][nzidx,:] # get reference sample matrix
		Mtest = pop['samples'][xtest]['M'][nzidx,:]  # get test sample matrix
		# Calculate the best set of genes for the current subset
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest
		Mtot = ss.hstack((subref,subtest))
		numexpr=np.sum(Mtot>0,axis=1)
		percexpr = numexpr/np.size(Mtot,1)
		gidx = np.where(percexpr>0.10)[0]
		subref = subref[gidx,:]
		subtest = subtest[gidx,:]
		genes = np.array(pop['genes']) # get original gene labels
		genes = genes[nzidx[gidx]] 
	else: 
		raise Exception('The option usefiltered must be one of three strings: \'filtered\', \'unfiltered\', \'refilter\'')

	subref = subref.toarray() # from sparse matrix to numpy array for slicing efficiency
	subtest = subtest.toarray() # from sparse matrix to numpy array for slicing efficiency
	
	with Pool(pop['ncores']) as p:
		q = np.array(p.starmap(l1norm, [(ig, subref[ig,:], subtest[ig,:], nbins) for ig in range(subref.shape[0])])) # for each gene idx ig, call the l1norm function

	# reorder variables based on l1norm values order
	idx = np.argsort(q)
	q = q[idx]
	genes = genes[idx]
	subref = subref[idx,:]
	subtest = subtest[idx,:]

	# render l1norm values
	samplename = sample.replace('/','') # remove slash char to not mess up the folder path
	dname = 'diffexp/%d_%s_%s_%d/' % (refcomp, reftype, samplename,testcomp) # define directory name
	# dname = 'diffexp/%d_%s/' % (refcomp, samplename) # define directory name
	mkdir(os.path.join(pop['output'], dname)) # create directory if needed
	x = np.arange(len(q))
	y = q
	plt.scatter(x, y, s=.1, alpha=1)
	plt.axhline(y=cutoff, color='red', linewidth=.5, label='Cutoff')
	plt.axhline(y=-cutoff, color='red', linewidth=.5)
	plt.xticks([])	
	plt.ylabel('l1-norm')
	plt.xlabel('Genes')
	plt.legend()
	filename = 'l1norm_values'
	filename = os.path.join(pop['output'], dname, '%s.pdf' % filename)
	plt.savefig(filename, dpi=200, bbox_inches='tight')
	plt.close()

	downregulated_idx = np.where(np.array(q)<-cutoff)[0] # get indices of genes with low l1-norm values
	upregulated_idx = np.where(np.array(q)>cutoff)[0] # get indices of genes with high l1-norm values
	downregulated = [genes[i] for i in downregulated_idx] # get gene labels
	upregulated = [genes[i] for i in upregulated_idx] # get gene labels
	if len(downregulated+upregulated) == 0:
		raise Exception('Cutoff value did not retrieve any gene. Please modify cutoff based on %s' % filename)

	# gsea
	currpath = os.path.abspath(os.path.dirname(__file__)) # get current path of this file to find the genesets
	geneset = 'c5bp' # name of the geneset file
	d = load_dict(os.path.join(currpath, "gsea/%s.npy" % geneset)) # load geneset dictionar
	ngenesets = 20

	dr_genesets = enrichment_analysis(pop, d, downregulated, len(pop['genes']), ngenesets) # find genesets pvalues for the list of downregulated genes
	ur_genesets = enrichment_analysis(pop, d, upregulated, len(pop['genes']), ngenesets) # find genesets pvalues for the list of upregulated genes

	lidx = np.concatenate([downregulated_idx,upregulated_idx])
	labels = ['downregulated']*len(downregulated_idx)+['upregulated']*len(upregulated_idx)

	with open(os.path.join(pop['output'], dname, 'downregulated_genes.txt'),'w') as fout:
		fout.write('Downregulated genes for sample %s relative to the reference sample\n\n' % sample)
		fout.write('\n'.join(downregulated)) # save list of downregulated gene names
		fout.write('\n\nMatching genesets:\n\n')
		fout.write('\n'.join(dr_genesets))
	with open(os.path.join(pop['output'], dname, 'upregulated_genes.txt'),'w') as fout:
		fout.write('Upregulated genes for sample: %s relative to the reference sample\n\n' % sample)
		fout.write('\n'.join(upregulated)) # save list of upregulated gene names
		fout.write('\n\nMatching genesets:\n\n')
		fout.write('\n'.join(ur_genesets))
	
	if renderhists == True: # if variable is True, then start histogram rendering
	#   dname = 'diffexp/%d_%s/hists/' % (refcomp, samplename) # define directory name
		dname = 'diffexp/%d_%s_%s_%d/hists/' % (refcomp, reftype, samplename,testcomp) # define directory name

		try:
			shutil.rmtree(os.path.join(pop['output'], dname))
		except:
			pass
		mkdir(os.path.join(pop['output'], dname)) # create directory if needed
		for lbl,i in zip(labels, lidx): # for each gene index in final list
			gname = genes[i]

			arrref = subref[i,:]
			arrtest = subtest[i,:]
			maxref, maxtest = np.max(arrref), np.max(arrtest)
			max_ = max(maxref,maxtest)

			nbins = 20
			bref, beref = np.histogram(arrref, bins=nbins, range=(0,max_))
			btest, betest = np.histogram(arrtest, bins=nbins, range=(0,max_))
			bref = bref/len(arrref)
			btest = btest/len(arrtest)

			width = beref[-1]/nbins
			plt.bar(beref[:-1], bref, label=xref, alpha=.3, width=width)
			plt.bar(beref[:-1], btest, label=xtest, alpha=0.3, width=width)
			plt.legend()
			plt.xlabel('Normalized counts')
			plt.ylabel('Fraction of cells in subpopulation')
			plt.title('Gene %s\nSubpopulation #%d of %s' % (gname, refcomp, xref))

			filename = '%s_%s' % (lbl, gname)
			plt.savefig(os.path.join(pop['output'], dname, '%s.pdf' % filename), dpi=200, bbox_inches='tight')
			plt.close()

	lidx = [genes[i] for i in lidx]
	plot_heatmap(pop, refcomp, lidx, clustersamples=False, clustercells=True, savename='refcomp%d_%s_%s' % (refcomp, reftype, sample), figsize=figsize, cmap='Purples', samplelimits=False, scalegenes=True, only=sample, equalncells=equalncells)
	return lidx

def diffexp_testcomp(pop, refcomp=0, sample='', nbins=20, cutoff=.5, renderhists=True, usefiltered='filtered'):
	'''
	Find differentially expressed genes between a reference subpopulation
	and the subpopulation of a sample that aligned to it

	Parameters
	----------
	refcomp : int
		Subpopulation number of the reference sample's GMM
	sample : str
		Name of the sample to compare
	nbins : int, optional
		Number of histogram bins to use
	nleft : int
		Number of underexpressed genes to retrieve
	nright : int
		Number of overexpressed genes to retrieve
	renderhists : bool
		Render histograms or not for the top differentially expressed genes
	usefiltered : str
		Either 'filtered', 'unfiltered', or 'refilter'. default: 'filtered'	
	'''
	xref = pop['ref'] # get reference sample label	
	reftype = pop['samples'][xref]['gmm_types'][refcomp]
	ncomps = pop['samples'][xref]['gmm'].n_components-1

	if sample not in pop['order']:
		raise Exception('Sample name not valid. Use show_samples(pop) to display valid sample names.')
	if refcomp > ncomps:
		raise Exception('Component number too high. Must be between 0 and %d' % ncomps)

	xtest = sample # test sample label

	try:
		arr = pop['samples'][xtest]['alignments'] # get alignments between reference and test
		irow = np.where(arr[:,0] == testcomp)[0][0] # get alignment that matches test subpopulation
		refcomp = int(arr[irow,1])# get ref subpopulation number
	except:
		raise Exception('Could not retrieve a matching alignment for sample %s, component %d' % (sample, refcomp))

	predictionref = pop['samples'][xref]['gmm'].predict(get_coeff(pop,xref)) # get ref cell assignments
	predictiontest = pop['samples'][xtest]['gmm'].predict(get_coeff(pop,xtest)) # get test cell assignments

	idxref = np.where(predictionref==refcomp)[0] # get matching indices
	idxtest = np.where(predictiontest==testcomp)[0] # get matching indices

	if usefiltered == 'filtered':
		Mref = pop['samples'][xref]['M_norm'] # get filtered reference sample matrix
		Mtest = pop['samples'][xtest]['M_norm'] # get filtered test sample matrix
		genes = np.array(pop['filtered_genes']) # get filtered gene labels
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest

	elif usefiltered == 'unfiltered': # Use just nonzero genes
		nzidx = pop['nzidx']
		Mref = pop['samples'][xref]['M'][nzidx,:] # get reference sample matrix
		Mtest = pop['samples'][xtest]['M'][nzidx,:]  # get test sample matrix
		genes = np.array(pop['genes']) # get gene labels
		genes = genes[nzidx] 
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest

	elif usefiltered =='refilter':
		# Only keep the genes that are present in >10% of cells
		nzidx = pop['nzidx']
		Mref = pop['samples'][xref]['M'][nzidx,:] # get reference sample matrix
		Mtest = pop['samples'][xtest]['M'][nzidx,:]  # get test sample matrix
		# Calculate the best set of genes for the current subset
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest
		Mtot = ss.hstack((subref,subtest))
		numexpr=np.sum(Mtot>0,axis=1)
		percexpr = numexpr/np.size(Mtot,1)
		gidx = np.where(percexpr>0.10)[0]
		subref = subref[gidx,:]
		subtest = subtest[gidx,:]
		genes = np.array(pop['genes']) # get original gene labels
		genes = genes[nzidx[gidx]] 
		print(len(genes))
	else: 
		raise Exception('The option usefiltered must be one of three strings: \'filtered\', \'unfiltered\', \'refilter\'')

	subref = subref.toarray() # from sparse matrix to numpy array for slicing efficiency
	subtest = subtest.toarray() # from sparse matrix to numpy array for slicing efficiency
	
	with Pool(pop['ncores']) as p:
		q = np.array(p.starmap(l1norm, [(ig, subref[ig,:], subtest[ig,:], nbins) for ig in range(subref.shape[0])])) # for each gene idx ig, call the l1norm function

	# reorder variables based on l1norm values order
	idx = np.argsort(q)
	q = q[idx]
	genes = genes[idx]
	subref = subref[idx,:]
	subtest = subtest[idx,:]

	# render l1norm values
	samplename = sample.replace('/','') # remove slash char to not mess up the folder path
	# dname = 'diffexp/%d_%s/' % (refcomp, samplename) # define directory name
	dname = 'diffexp/%d_%s_%s_%d/' % (refcomp, reftype, samplename,testcomp) # define directory name
	mkdir(os.path.join(pop['output'], dname)) # create directory if needed
	x = np.arange(len(q))
	y = q
	plt.scatter(x, y, s=.1, alpha=1)
	plt.axhline(y=cutoff, color='red', linewidth=.5, label='Cutoff')
	plt.axhline(y=-cutoff, color='red', linewidth=.5)
	plt.xticks([])	
	plt.ylabel('l1-norm')
	plt.xlabel('Genes')
	plt.legend()
	filename = 'l1norm_values'
	filename = os.path.join(pop['output'], dname, '%s.pdf' % filename)
	plt.savefig(filename, dpi=200, bbox_inches='tight')
	plt.close()

	downregulated_idx = np.where(np.array(q)<-cutoff)[0] # get indices of genes with low l1-norm values
	upregulated_idx = np.where(np.array(q)>cutoff)[0] # get indices of genes with high l1-norm values
	downregulated = [genes[i] for i in downregulated_idx] # get gene labels
	upregulated = [genes[i] for i in upregulated_idx] # get gene labels
	if len(downregulated+upregulated) == 0:
		raise Exception('Cutoff value did not retrieve any gene. Please modify cutoff based on %s' % filename)

	# gsea
	currpath = os.path.abspath(os.path.dirname(__file__)) # get current path of this file to find the genesets
	geneset = 'c5bp' # name of the geneset file
	d = load_dict(os.path.join(currpath, "gsea/%s.npy" % geneset)) # load geneset dictionar
	ngenesets = 20

	dr_genesets = enrichment_analysis(pop, d, downregulated, len(pop['genes']), ngenesets) # find genesets pvalues for the list of downregulated genes
	ur_genesets = enrichment_analysis(pop, d, upregulated, len(pop['genes']), ngenesets) # find genesets pvalues for the list of upregulated genes

	lidx = np.concatenate([downregulated_idx,upregulated_idx])
	labels = ['downregulated']*len(downregulated_idx)+['upregulated']*len(upregulated_idx)

	with open(os.path.join(pop['output'], dname, 'downregulated_genes.txt'),'w') as fout:
		fout.write('Downregulated genes for sample %s relative to the reference sample\n\n' % sample)
		fout.write('\n'.join(downregulated)) # save list of downregulated gene names
		fout.write('\n\nMatching genesets:\n\n')
		fout.write('\n'.join(dr_genesets))
	with open(os.path.join(pop['output'], dname, 'upregulated_genes.txt'),'w') as fout:
		fout.write('Upregulated genes for sample: %s relative to the reference sample\n\n' % sample)
		fout.write('\n'.join(upregulated)) # save list of upregulated gene names
		fout.write('\n\nMatching genesets:\n\n')
		fout.write('\n'.join(ur_genesets))
	
	if renderhists == True: # if variable is True, then start histogram rendering
		dname = 'diffexp/%d_%s_%s_%d/hists/' % (refcomp, reftype, samplename,testcomp) # define directory name
		# dname = 'diffexp/%d_%s/hists/' % (refcomp, samplename) # define directory name
		try:
			shutil.rmtree(os.path.join(pop['output'], dname))
		except:
			pass
		mkdir(os.path.join(pop['output'], dname)) # create directory if needed
		for lbl,i in zip(labels, lidx): # for each gene index in final list
			gname = genes[i]

			arrref = subref[i,:]
			arrtest = subtest[i,:]
			maxref, maxtest = np.max(arrref), np.max(arrtest)
			max_ = max(maxref,maxtest)

			nbins = 20
			bref, beref = np.histogram(arrref, bins=nbins, range=(0,max_))
			btest, betest = np.histogram(arrtest, bins=nbins, range=(0,max_))
			bref = bref/len(arrref)
			btest = btest/len(arrtest)

			width = beref[-1]/nbins
			plt.bar(beref[:-1], bref, label=xref, alpha=.3, width=width)
			plt.bar(beref[:-1], btest, label=xtest, alpha=0.3, width=width)
			plt.legend()
			plt.xlabel('Normalized counts')
			plt.ylabel('Percentage of cells in subpopulation')
			plt.title('Gene %s\nSubpopulation #%d of %s' % (gname, refcomp, xref))

			filename = '%s_%s' % (lbl, gname)
			plt.savefig(os.path.join(pop['output'], dname, '%s.pdf' % filename), dpi=200, bbox_inches='tight')
			plt.close()

	lidx = [genes[i] for i in lidx]
	plot_heatmap(pop, refcomp, lidx, clustersamples=False, clustercells=True,savename='refpop%d_%s_%s_heatmap' % (refcomp,reftype, sample),figsize=(15,15), cmap='Purples', samplelimits=False, scalegenes=True, only=sample, equalncells=True)
	return lidx

def all_diffexp(pop, refcomp=0, sample='', nbins=20, cutoff=.5, renderhists=True, usefiltered='filtered'):
	'''
	Find differentially expressed genes between a reference subpopulation
	and ALL subpopulations within a sample that align to it

	Parameters
	----------
	refcomp : int
		Subpopulation number of the reference sample's GMM
	sample : str
		Name of the sample to compare
	nbins : int, optional
		Number of histogram bins to use
	nleft : int
		Number of underexpressed genes to retrieve
	nright : int
		Number of overexpressed genes to retrieve
	renderhists : bool
		Render histograms or not for the top differentially expressed genes
	usefiltered : str
		Either 'filtered', 'unfiltered', or 'refilter'. Default: 'filtered'	

	'''
	xref = pop['ref'] # get reference sample label
	celltypes = pop['samples'][xref]['gmm_types']
	reftype = celltypes[refcomp]
	ncomps = pop['samples'][xref]['gmm'].n_components-1

	if sample not in pop['order']:
		raise Exception('Sample name not valid. Use show_samples(pop) to display valid sample names.')
	if refcomp > ncomps:
		raise Exception('Component number too high. Must be between 0 and %d' % ncomps)

	xtest = sample # test sample label

	try:
		arr = pop['samples'][xtest]['alignments'] # get alignments between reference and test
		irow = np.where(arr[:,1] == refcomp) # get alignment that match reference subpopulation
		itest = int(arr[irow, 0]) # get test subpopulation number
	except:
		raise Exception('Could not retrieve a matching alignment between sample %s and reference component %d' % (sample, refcomp))

	refcoeff = get_coeff(pop,xref)
	testcoeff = get_coeff(pop,xtest)
	predictionref = pop['samples'][xref]['gmm'].predict(refcoeff) # get ref cell assignments
	predictiontest = pop['samples'][xtest]['gmm'].predict(testcoeff) # get test cell assignments

	idxref = np.where(predictionref==refcomp)[0] # get matching indices
	idxtest = np.where(predictiontest==itest)[0] # get matching indices

	if usefiltered == 'filtered':
		Mref = pop['samples'][xref]['M_norm'] # get filtered reference sample matrix
		Mtest = pop['samples'][xtest]['M_norm'] # get filtered test sample matrix
		genes = np.array(pop['filtered_genes']) # get filtered gene labels
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest

	elif usefiltered == 'unfiltered': # Use just nonzero genes
		nzidx = pop['nzidx']
		Mref = pop['samples'][xref]['M'][nzidx,:] # get reference sample matrix
		Mtest = pop['samples'][xtest]['M'][nzidx,:]  # get test sample matrix
		genes = np.array(pop['genes']) # get gene labels
		genes = genes[nzidx] 
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest

	elif usefiltered =='refilter':
		# Only keep the genes that are present in >10% of cells
		nzidx = pop['nzidx']
		Mref = pop['samples'][xref]['M'][nzidx,:] # get reference sample matrix
		Mtest = pop['samples'][xtest]['M'][nzidx,:]  # get test sample matrix
		# Calculate the best set of genes for the current subset
		subref = Mref[:,idxref] # subset cells that match subpopulation refcomp
		subtest = Mtest[:,idxtest] # subset cells that match subpopulation itest
		Mtot = ss.hstack((subref,subtest))
		numexpr=np.sum(Mtot>0,axis=1)
		percexpr = numexpr/np.size(Mtot,1)
		gidx = np.where(percexpr>0.10)[0]
		subref = subref[gidx,:]
		subtest = subtest[gidx,:]
		genes = np.array(pop['genes']) # get original gene labels
		genes = genes[nzidx[gidx]] 
	else: 
		raise Exception('The option usefiltered must be one of three strings: \'filtered\', \'unfiltered\', \'refilter\'')


	subref = subref.toarray() # from sparse matrix to numpy array for slicing efficiency
	subtest = subtest.toarray() # from sparse matrix to numpy array for slicing efficiency
	
	with Pool(pop['ncores']) as p:
		q = np.array(p.starmap(l1norm, [(ig, subref[ig,:], subtest[ig,:], nbins) for ig in range(subref.shape[0])])) # for each gene idx ig, call the l1norm function
	q_raw = q
	genes_raw = genes

	# resort the genes according to L1 norm value and update multiple variables
	idx = np.argsort(q)
	q = q[idx]
	genes = genes[idx]
	subref = subref[idx,:]
	subtest = subtest[idx,:]

	downregulated_idx = np.where(np.array(q)<-cutoff)[0] # get indices of genes with low l1-norm values
	upregulated_idx = np.where(np.array(q)>cutoff)[0] # get indices of genes with high l1-norm values
	downregulated = [genes[i] for i in downregulated_idx] # get gene labels
	upregulated = [genes[i] for i in upregulated_idx] # get gene labels
	
	if len(downregulated+upregulated) == 0:
		# raise Exception('Cutoff value did not retrieve any gene. Please modify cutoff')
		lidx = [];
		print('Cutoff value did not retrieve any genes for '+ xtest)
		return  q_raw, genes_raw, lidx, upregulated, downregulated

	# gsea
	currpath = os.path.abspath(os.path.dirname(__file__)) # get current path of this file to find the genesets
	geneset = 'c5bp' # name of the geneset file
	d = load_dict(os.path.join(currpath, "gsea/%s.npy" % geneset)) # load geneset dictionar
	ngenesets = 20

	dr_genesets = enrichment_analysis(pop, d, downregulated, len(pop['genes']), ngenesets) # find genesets pvalues for the list of downregulated genes
	ur_genesets = enrichment_analysis(pop, d, upregulated, len(pop['genes']), ngenesets) # find genesets pvalues for the list of upregulated genes

	lidx = np.concatenate([downregulated_idx,upregulated_idx])
	labels = ['downregulated']*len(downregulated_idx)+['upregulated']*len(upregulated_idx)

	samplename = sample.replace('/','') # remove slash char to not mess up the folder path
	# dname = 'diffexp/%d_%s/' % (refcomp, samplename) # define directory name
	dname = 'diffexp/refpop%d_%s_%s/' % (refcomp, reftype, samplename) 
	mkdir(os.path.join(pop['output'], dname)) # create directory if needed
	with open(os.path.join(pop['output'], dname, 'downregulated_genes.txt'),'w') as fout:
		fout.write('Downregulated genes for sample %s relative to the reference sample\n\n' % sample)
		fout.write('\n'.join(downregulated)) # save list of downregulated gene names
		fout.write('\n\nMatching genesets:\n\n')
		fout.write('\n'.join(dr_genesets))
	with open(os.path.join(pop['output'], dname, 'upregulated_genes.txt'),'w') as fout:
		fout.write('Upregulated genes for sample: %s relative to the reference sample\n\n' % sample)
		fout.write('\n'.join(upregulated)) # save list of upregulated gene names
		fout.write('\n\nMatching genesets:\n\n')
		fout.write('\n'.join(ur_genesets))

	# render l1norm values
	x = np.arange(len(q))
	y = q
	plt.scatter(x, y, s=.1, alpha=1)
	plt.axhline(y=cutoff, color='red', linewidth=.5, label='Cutoff')
	plt.axhline(y=-cutoff, color='red', linewidth=.5)
	plt.xticks([])	
	plt.ylabel('l1-norm')
	plt.xlabel('Genes')
	plt.legend()
	filename = 'l1norm_values'
	plt.savefig(os.path.join(pop['output'], dname, '%s.pdf' % filename), dpi=200, bbox_inches='tight')
	plt.close()

	if renderhists == True: # if variable is True, then start histogram rendering
		# dname = 'diffexp/%d_%s/hists/' % (refcomp,samplename) # define directory name
		dnamehist = dname + 'hists/' # define directory name
		try:
			shutil.rmtree(os.path.join(pop['output'], dnamehist))
		except:
			pass
		mkdir(os.path.join(pop['output'], dnamehist)) # create directory if needed
		for lbl,i in zip(labels, lidx): # for each gene index in final list
			gname = genes[i]

			arrref = subref[i,:]
			arrtest = subtest[i,:]
			maxref, maxtest = np.max(arrref), np.max(arrtest)
			max_ = max(maxref,maxtest)

			# Format data fo histogram plot:
			nbins = 20
			bref, beref = np.histogram(arrref, bins=nbins, range=(0,max_))
			btest, betest = np.histogram(arrtest, bins=nbins, range=(0,max_))
			bref = bref/len(arrref)
			btest = btest/len(arrtest)

			width = beref[-1]/nbins
			ax1=plt.subplot(1, 2, 1)
			plt.bar(beref[:-1], bref, label=xref[0:10], alpha=.3, width=width)
			plt.bar(beref[:-1], btest, label=xtest[0:10], alpha=0.3, width=width)
			plt.legend()
			plt.xlabel('Normalized log(counts)')
			plt.ylabel('Cell Fraction')
			plt.title('%s\n %s-%d of %s' % (gname, reftype, refcomp, xref)) 
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=1)

			# Format data for violinplot:
			labels1 = [xref]*len(arrref)
			labels2 = [xtest]*len(arrtest)
			fakey = [1]*(len(arrtest)+len(arrref))

			v1 = np.concatenate([arrref,arrtest])
			v2 = np.concatenate([labels1,labels2])
			arrdf = pd.DataFrame(data = list([v1,v2]))
			arrdf = pd.DataFrame.transpose(arrdf)
			arrdf.rename(columns = {0:'values',1:'sample'},inplace=True)
			arrdf['values']=arrdf['values'].astype('float64')
			arrdf['sample']=arrdf['sample'].astype('category')
			arrdf['y'] = fakey
			arrdf['y'] = arrdf['y'].astype('float64')

			ax2=plt.subplot(1, 2, 2)
			sns.violinplot(y='values' ,x='y',data=arrdf,hue='sample',split=True, orient='v')
			plt.ylabel('Normalized log(counts)')
			plt.title(gname)
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)

			filename = '%s_%s' % (lbl, gname)
			plt.savefig(os.path.join(pop['output'], dnamehist, '%s.pdf' % filename),bbox_inches='tight')
			plt.close()

	lidx = [genes[i] for i in lidx]
	plot_heatmap(pop, refcomp, lidx, clustersamples=False, clustercells=True, savename='refpop%d_%s_%s_heatmap' % (refcomp,reftype, sample), figsize=(15,15), cmap='Purples', samplelimits=False, scalegenes=True, only=sample, equalncells=True)
	return q_raw, genes_raw, lidx, upregulated, downregulated

def all_samples_diffexp(pop, nbins=20, cutoff=[], renderhists=True, usefiltered='filtered', tailthresh=0.001, plotL1 = False, plotRibbon = False):
	'''
	Compute differentially expressed genes for all cell types and all samples. 

	Uses pop['deltas'][currtype]['orderedsamples'] to retrieve the sample order

	Puts differential expression outputs in pop['diffexp']

	Parameters
	----------
	deltaobj : dict,
		contains many keys including sample ordering for different cell types
	nbins : int, optional
		Number of histogram bins to use
	cutoff : float
		L1 norm cutoff. If this is empty, we recalculate it empirically from the control samples. 
	renderhists : bool
		Render histograms or not for the top differentially expressed genes. Default: True
	usefiltered : str
		Either 'filtered', 'unfiltered', or 'refilter'. Default: 'filtered'	
	tailthresh : float
		P-value threshold for total density "weight" within the distribution of L1 norm values for all control samples. 
		This number is used to generate a L1norm threshold to determine which differentially expressed genes are significant
		(p-val < tailthresh)

	Outputs
	----------
	pop['diffexp']: dict
		contains the following objects: 

	pop['diffexp']['de_df'] : dataframe
		table reporting on all genes up and downregulated across samples and cell types
		column names are: sample, celltype, sign, ngenes, and genes

	pop['diffexp'][celltype] : dict
		Dictionary containing information for each cell type, including L1norm values
	'''
	dname = 'diffexp/'
	deobj={}
	samples = pop['order']
	controlstring = pop['controlstring']
	if controlstring==None:
		raise Exception('Did not supply controlstring during load. Can be set now by executing: pop[\'controlstring\']=X')
	# deltaobj = pop['deltas']

	ref = pop['ref'] # get reference sample name
	celltypes = pop['samples'][ref]['gmm_types']

	# check that nbis is the same length as gmm_types
	if len(nbins)!=len(celltypes):
		raise Exception('nbins vector must be same length as celltypes')

	# get control values
	x = range(len(samples))
	controls = [i for i in pop['samples'].keys() if controlstring in i]
	# remove reference control sample
	controls.pop(controls.index(ref))
	
	for y in range(len(celltypes)) : 
		currtype = celltypes[y];
		curr_nbin = nbins[y]

		# Determine cutoff from control samples if it is not supplied
		if np.isscalar(cutoff) and cutoff >= 0 and cutoff <= 2: 
			currcutoff = cutoff
		elif not cutoff : 
			# First calculate control values
			print('******************************************************')
			print('Using controls to determine an L1 cutoff for '+currtype)
			print('******************************************************')
			all_control_q = []

			for x in controls: 
				print(x + ' ' + currtype)
				# calculate the differentially expressed genes
				q_raw, genes_raw, lidx, upregulated, downregulated = all_diffexp(pop, refcomp=y, sample=x, nbins=curr_nbin, cutoff=0.5, renderhists=renderhists, usefiltered=usefiltered)
				all_control_q.append(q_raw)

			all_control_q = np.concatenate(all_control_q)
			ctrlbins = 100
			max_ = np.max(all_control_q)
			min_ = np.min(all_control_q)

			bn, be = np.histogram(all_control_q, bins=ctrlbins, range=(-max_,max_))
			bn = bn /len(all_control_q) 

		    # find the threshold at which less than 0.001% of the genes in the controls 
		    # sample have this level of differential change
		    # sweep from 0.45 to 0.65:
			testrange = np.linspace(0.3,0.65,36)
			for i in range(len(testrange)):
				currcutoff = testrange[i]

				tailidx = np.where(abs(be) > currcutoff)[0]
				tailweight = np.sum(bn[tailidx[:-1]])
				if tailweight < tailthresh :
					break # keep current qthresh and don't try others

			print(currtype + ' cutoff is: ' + str(currcutoff))
			print('******************************************************')
			print('Now calculating for the remaining samples...')
			print('******************************************************')
		else: 
			raise Exception ("You must supply a cutoff between 0 and 2 or leave it blank: cutoff=[]")

		# Run over rest of samples with cutoff (either supplied or cell-type specific)
		deobj[currtype] = {}
		deobj[currtype]['samples'] = {}  
		# deobj[currtype]['orderedsamples'] = deltaobj[currtype]['orderedsamples']
		deobj[currtype]['orderedsamples'] = pop['order']
		
		all_q = []            

		for x in samples:
			print('Calculating for ' + x + ' ' + currtype + '...')

			# calculate the differentially expressed genes
			if checkalignment(pop, y, x): 			
				q_raw, genes_raw, lidx, upregulated, downregulated = all_diffexp(pop, refcomp=y, sample=x, nbins=curr_nbin, cutoff=currcutoff, renderhists=renderhists, usefiltered=usefiltered)
				numgenes = len(lidx)
			else: 
				print('alignment does not exist')
				lidx = []
				downregulated = []
				upregulated = []
				q_raw = np.zeros((len(genes_raw),))
			# create entry and load sparse matrix

			deobj[currtype]['samples'][x]={}
			deobj[currtype]['samples'][x]['numgenes'] = len(lidx)
			deobj[currtype]['samples'][x]['numgenes_down'] = len(downregulated)
			deobj[currtype]['samples'][x]['numgenes_up'] = len(upregulated)
			deobj[currtype]['samples'][x]['genes'] = lidx
			deobj[currtype]['samples'][x]['genes_down'] = downregulated
			deobj[currtype]['samples'][x]['genes_up'] = upregulated
			all_q.append(q_raw)

		deobj[currtype]['all_l1norm'] = np.array(all_q)
		deobj[currtype]['all_genes'] = genes_raw
		deobj[currtype]['all_samples'] = samples
		deobj[currtype]['cutoff'] = currcutoff # currcutoff is cell type specific

	# Put all the computed values into a single dataframe, and also compute shared genes
	samplelist = []
	ctlist = []
	signlist = []
	ngeneslist = []
	genelist = []

	for x in samples: 
		for currtype in celltypes :
			# get downregulated genes
			n_down = deobj[currtype]['samples'][x]['numgenes_down']
			genes_down = deobj[currtype]['samples'][x]['genes_down']
			samplelist.append(x)
			ctlist.append(currtype)
			signlist.append('down')
			ngeneslist.append(n_down)
			genelist.append( ','.join(genes_down))
			# get downregulated genes
			n_up = deobj[currtype]['samples'][x]['numgenes_up']
			genes_up = deobj[currtype]['samples'][x]['genes_up']
			samplelist.append(x)
			ctlist.append(currtype)
			signlist.append('up')
			ngeneslist.append(n_up)    
			genelist.append( ','.join(genes_up))

		# Calculate shared genes as the intersection of affected genes in all cell types
		shared_up = deobj[celltypes[0]]['samples'][x]['genes_up']
		shared_down = deobj[celltypes[0]]['samples'][x]['genes_down']

		# iterate through other cell types to get intersecting lists
		for currtype in celltypes[1:] :
			shared_down = np.intersect1d(shared_down, deobj[currtype]['samples'][x]['genes_down'])
			shared_up = np.intersect1d(shared_up, deobj[currtype]['samples'][x]['genes_up'])
			
		samplelist.append(x)
		ctlist.append('shared')
		signlist.append('down')
		ngeneslist.append(len(shared_down))
		genelist.append( ','.join(shared_down))

		samplelist.append(x)
		ctlist.append('shared')
		signlist.append('up')
		ngeneslist.append(len(shared_up))
		genelist.append( ','.join(shared_up))

	d = {'sample': samplelist, 'celltype': ctlist, 'sign' : signlist, 'ngenes': ngeneslist, 'genes': genelist}
	de_df = pd.DataFrame(data=d)
	deobj['de_df'] = de_df
	de_df.to_csv(os.path.join(pop['output'], dname, 'all_degenes_by_celltype.csv')) # save dataframe in a single csv file
	pop['diffexp'] = deobj

	# Now run all samples thorugh L1 heatmap
	if plotL1: 
		for x in samples:
			plot_L1_heatmap(pop, x, dname)

	# Now plot ribbon plot for numbers of genes that have changed
	if plotRibbon:
		ribboncols = sns.color_palette('muted')
		plot_ribbon_ngenes(pop, colors = ribboncols)

def calc_p_value(controlvals, testvals, tail = 1) : 
	'''
	Calculates the p-value using a one-sample t-test with FDR correction. 
	In this case values for control replicates are considered the "sample", 
	tested against the testvalues individually (i.e. the values for each perturbation)

	Parameters
	----------
	controlvals : list, floats
		values for the control samples
	testvals : list, float
		values for all test samples 
	tail : int
		single(1) or double-tailed (2)
	
	Output
	----------
	pvals: list, floats
	CI_min : float
		minimum of 95% confidence interval 
	CI_max : 
		maximum of 95% confidence interval

	'''
	N = len(controlvals)
	control_mean = sum(controlvals)/len(controlvals)
	control_std = np.std(controlvals)

	# calculate the t-statistic
	t_val = abs(testvals - control_mean)/(control_std/np.sqrt(N))

	## Compare with the critical t-value
	# Degrees of freedom
	df = len(controlvals)-1 # number of samples = control samples -1

	# p-value after comparison with the t 
	if tail == 1:
		pvals_raw = 1 - stats.t.cdf(t_val, df=df) 
	elif tail == 2: 
		pvals_raw = (1-stats.t.cdf(t_val, df=df))*2
	else: 
		raise Exception('tail must be 1 or 2')

	# FDR correction
	ranked_pvals = rankdata(pvals_raw)
	pvals = pvals_raw * len(pvals_raw) / ranked_pvals
	pvals[pvals > 1] = 1

	# Calculate CI min and CI max
	CI_min = control_mean - 1.96*control_std/np.sqrt(N)
	CI_max = control_mean + 1.96*control_std/np.sqrt(N)

	return pvals, CI_min, CI_max

'''
Auxiliary functions for building models with cell types
'''

def remove_celltypes(pop, ctlist): 
	'''
	Remove specified cell types from the pop object

	Parameters
	----------
	pop : dict
	    PopAlign object        
	ctlist : str
		list of cell types
	'''

	# remove data from main section first
	# ['meta']
	# ['pca']: ['proj'] ['mines'] ['maxes']
	# ['umap']
	# ['tsne']
	# ['onmf']

	newpop = pop # make a new pop object
 	# from each sample: remove cells from 'M', 'cell_type', 'indices', 'M_norm', 'pcaproj'
	for ct in ctlist: 
		
		# remove cells from top level of pop object 
		allcelltypes = cat_data(newpop,'cell_type')
		allkeepidx = np.where(np.array(allcelltypes) != ct)[0]
		newpop['meta'] = newpop['meta'].iloc[allkeepidx]
		try:
			newpop['umap'] = newpop['umap'][allkeepidx,:]
		except: 
			print(ct + ' not removed for umap: no umap coordinates currently stored.')
		try: 
			newpop['tsne'] = newpop['tsne'][allkeepidx,:]
		except: 
			print(ct + ' not removed for tsne: no tsne coordinates currently stored.')
		try:
			newpop['pca']['proj'] = newpop['pca']['proj'][allkeepidx,:]
		except: 
			print(ct + ' not removed for pca: no pca coefficients currently stored.')

		# remove cells from each sample:
		for x in newpop['order']: 
			currtypes = newpop['samples'][x]['cell_type']
			keepidx = np.where(np.array(currtypes) != ct)[0]
			newpop['samples'][x]['M'] = newpop['samples'][x]['M'][:,keepidx]
			newpop['samples'][x]['cell_type'] = [currtypes[i] for i in keepidx]
			try:
				newpop['samples'][x]['M_norm'] = newpop['samples'][x]['M_norm'][:,keepidx]
			except: 
				if x == newpop['order'][0]:
					print(ct + 's not removed for M_norm in each sample: no normalized gene expression data stored')
			try: 
				newpop['samples'][x]['pcaproj'] = newpop['samples'][x]['pcaproj'][keepidx,:]
			except: 
				if x == newpop['order'][0]:
					print(ct + 's not removed for pca in each sample: no pca coefficients currently stored.')
			try: 
				newpop['samples'][x]['C'] = newpop['samples'][x]['C'][keepidx,:]
			except: 
				if x == newpop['order'][0]:
					print(ct + 's not removed for onmf in each sample: no onmf coefficients currently stored.')

	# redefine indices for each sample 
	start = 0
	end = 0
	for i,x in enumerate(newpop['order']): # for each sample
		numcells = np.shape(newpop['samples'][x]['M'])[1]
		end = start+numcells # update start and end cell indices
		newpop['samples'][x]['indices'] = (start,end) # update start and end cell indices
		start = end # update start and end cell indices

	# recalculate pca max and min	
	newproj = newpop['pca']['proj']
	newpop['pca']['maxes'] = newproj.max(axis=0) # store PCA projection space limits
	newpop['pca']['mines'] = newproj.min(axis=0) # store PCA projection space limits

	# only if all things have been replaced, set pop to newpop
	pop = newpop

def save_celltypes_in_meta(pop, meta_in, meta_out):
	'''
	Save the labeled cell types into the metadata file
	And also put it the pop object

	Parameters
	----------
	pop : dict
	    PopAlign object        
	meta_in : str
		original meta file name
	meta_out : str
	    file name that ends in csv
	'''

	# concatenate all coefficient matrices together
	allC = get_cat_coeff(pop)

	# use the gmm to classify all of the data
	classes = pop['gmm'].predict(allC)

	dicttypes = pop['gmm_types']
	celltypes = [dicttypes[i] for i in classes]

	# add new column to metadata with cell types
	newmeta = pop['meta']
	newmeta['cell_type'] = celltypes

	# load original metadata file and add cell_types column
	meta = pd.read_csv(meta_in)

	# if 'cell_type' column does not already exist, add new column in meta where all cell_types are None
	if 'cell_type' not in list(meta):
		meta['cell_type'] = None

	# insert the identified cell types from the newmeta object into the original metadata table
	for i in newmeta.index:
		bc = newmeta.loc[i].cell_barcode
		currcelltype = newmeta.loc[i].cell_type
		meta.at[i,'cell_type'] = currcelltype

	# save the meta data file
	meta.to_csv(meta_out) # save dataframe in a single csv file

	# Store newmeta into pop object
	pop['meta'] = newmeta

	# Store cell types back into individual samples
	for x in pop['order']:
		currtypes = newmeta[newmeta.sample_id == x].cell_type.values
		pop['samples'][x]['cell_type'] = currtypes

'''

Auxiliary functions for calculating abundance and divergence changes
'''

def calc_abund_scores(fname, maintypes=['B-cells', 'Myeloid', 'T cells'], col = 'CD3', value=1, controlstring='CONTROL'):
		
	meta = pd.read_csv(fname, header=0) # load metadata file

	# Find sample names that obey the metadata filter 
	if (value != None) and (col != None):
		tmp_only = meta[meta[col]==value]['sample_id'].dropna().unique()
	elif (value != None) and (col == None):
		raise Exception('col and value arguments must be specified together, or both equal to None')
	elif (value == None) and (col != None):
		raise Exception('col and value arguments must be specified together, or both equal to None')
	else:
		tmp_only =  meta['sample_id'].dropna().unique() # get unique list of sample names
	
	only = tmp_only.tolist()
	try: 
		only.remove('unknown')
	except: 
		print('')

	# store index of each barcode in a dictionary to quickly retrieve indices for a list of barcodes
	barcodes = meta.cell_barcode.tolist()
	bc_idx = {} 
	for i, bc in enumerate(barcodes):
		bc_idx[bc] = i

	# accumulate cell type counts for each sample
	accum_idx = [] # accumulate index values for subsetted samples
	order = list()
	all_counts = np.zeros((len(only),3))
	for i,y in enumerate(only): # go through the sample_id values to split the data and store it for each individual sample
		x = str(y)
		if x != 'unknown':
			sample_bcs = meta[meta.sample_id == x].cell_barcode.values # get the cell barcodes for sample defined by sample_id
			idx = [bc_idx[bc] for bc in sample_bcs] # retrieve list of matching indices
			accum_idx = accum_idx + idx

			# Get cell type abundances
			currtypes = meta.cell_type.loc[idx].tolist()
			curr_count = [currtypes.count(x) for x in maintypes]
			all_counts[i,:] = curr_count
			order.append(x)

	# Calculate proportions
	totcells = np.array(all_counts.sum(axis=1)).flatten()
	all_props = all_counts*0
	for i in range(len(all_props)): # for each column i
		all_props[i,:] = all_counts[i,:] #/ totcells[i] # divide data values by matching column sum

	# Calculate values for control samples
	# controlstring = pop['controlstring']
	controlidx = [i for i in range(0,len(order)) if controlstring in order[i]]

	control_props = all_props[controlidx,:]
	control_averages = control_props.mean(axis=0)
	control_stds = control_props.std(axis=0)

	# Generate z scores
	scores = 0* all_props
	for i in range(np.shape(all_props)[1]): 
# 		scores[:,i] = (all_props[:,i] - control_averages[i]) / control_averages[i] # tweak this
# 		scores[:,i] = (all_props[:,i] - control_averages[i]) / control_stds[i] # tweak this
		scores[:,i] = np.log2(all_props[:,i] / control_averages[i]) # tweak this

	snames = [x + '_score' for x in maintypes]
	# Create scores dataframe
	scores = pd.DataFrame(scores, index = order, columns=snames)

	# generate vector specifying whether sample is a control
	v = []
	for x in order:
		if controlstring in x:
			v.append('True')
		else: 
			v.append('False')
	# Add other columns
	scores['totcells'] = totcells # total cells
	scores['control'] = v
	cnames = [x + '_counts' for x in maintypes]
	for i in range(len(cnames)): 
		scores[cnames[i]] = all_counts[:,i]

	# get scores for controls (assume normally distributed)
	control_scores = scores[snames]
	control_scores = control_scores.iloc[controlidx,:]

	for i in range(len(snames)):
		pvals, CI_min, CI_max = calc_p_value(control_scores[snames[i]], scores[snames[i]], tail = 2)
		colname = maintypes[i] + '_pval'
		scores[colname] = pvals
	return scores


import sys
if not sys.warnoptions:
	import warnings
	warnings.simplefilter("ignore")
