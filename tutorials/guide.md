# PopAlign user guide

This guide covers the worflow and data structures of the single cell RNA-seq analysis PopAlign framework.

## Installing PopAlign

The user can install PopAlign with the following command:
```
pip install git+https://github.com/thomsonlab/popalign
```
We recommand to install install the PopAlign package and its required dependencies in a dedicated environment (created with Anaconda for example). This package is built to run in Python 3.

## Importing PopAlign in Python

After installing the PopAlign package with pip, the user can import it with the following line in their Python code file:
```python
import popalign as PA
```

## Load data

The user can load data with two distinct methods depending on the data format.

* Format 1:
Each sample has its own matrix. Only the paths to the matrix files and the path to the gene file are required. The paths to the matrix files need to be stored in a dictionary where the keys are the sample names and the values are the paths to their respective .mtx file.
```python
mysamples = {
	'sample1' : 'path/to/sample1.mtx',
	'sample2' : 'path/to/sample2.mtx',
}
mygenes = 'path/to/genes.tsv'
```

The function to be used with that data format is:
```python
pop = PA.load_samples(samples=mysamples, 
	genes=mygenes,
	outputfolder='output_samples',
	existing_obj=None)
```
It returns a dictionary that contains the loaded data, genes and various information. That object is used throughout the entire analysis.
The `outputfolder` parameter defines where the results of the analysis will be saved. Its default value is `output`. 
If the user hasn't loaded any samples, `existing_obj` should be `None`. If samples have already been loaded with `load_samples` or `load_multiplexed`, the object that these functions returned should be given as the argument for this parameter.

* Format 2:
All the samples are stored in one unique matrix file. This format requires the paths to the .mtx file, the gene list, the barcode list and a metadata file. The meta data file must be a .csv file that includes two columns: `cell_barcode` and `sample_id`. The first column contains the cell barcodes and the second column contains the matching sample names for those cells. The cells from the different samples will be selected in the matrix based on their barcode index.
```python
mymatrix = 'path/to/matrix.mtx'
mybarcodes = 'path/to/barcodes.tsv'
mygenes = 'path/to/genes.tsv'
mymetadata = 'path/to/metadata.csv'
```

The function to be used with that data format is:
```python
pop = PA.load_multiplexed(matrix=mymatrix, 
	barcodes=mybarcodes, 
	metafile=mymetadata, 
	genes=mygenes,
	outputfolder='output_multiplexed',
	only=[],
	col=None,
	value=None)
```
It returns a dictionary that contains the loaded data, genes and various information. That object is used throughout the entire analysis.
The `outputfolder` parameter defines where the results of the analysis will be saved. Its default value is `output`. 

If the user hasn't loaded any samples, `existing_obj` should be `None`. If samples have already been loaded with `load_samples` or `load_multiplexed`, the object that these functions returned should be given as the argument for this parameter.

The sample gene expression matrices are stored individually. To access the gene expression matrix of `sample1`, use the following:
```python
pop['samples']['sample1']['M']
```

The gene labels are stored in the `pop` object:
```python
pop['genes']
```

## Normalize data

The data is normalized to account for sequencing depth differences. This is done by dividing each cell (column in a gene expression matrix) by its transcript count sum. 

![](https://latex.codecogs.com/svg.latex?M_{j}=\frac{M_{j}}{\sum&space;M_{j}})

The values are then multiplied by a scaling factor. The scaling factor value can be either set by the user or computed automatically. When computed automatically, multiple factors are tested and the factor that minimizes the difference between the mean of the original data `D` prior to normalization and the mean of the normalized, scaled data is used.

![](https://latex.codecogs.com/svg.latex?D_{scaled}=argmin(\left&space;|&space;mean(D)-mean(D_{scaled_{i}})&space;\right&space;|))

The normalization function is:
```python
PA.normalize(pop, scaling_factor=None)
```
It normalizes the gene expression matrices in-place, i.e. the matrices are directly updated.
The `scaling_factor` parameter can be used to set the scaling factor value. It defaults to None, which means the scaling factor is computed automatically, then applied to the column normalized data.

## Gene filtering

The process of filtering genes is divided into two steps: plotting the genes to select the most variable genes, and then filtering the genes once they have been selected with the first step.

The function to plot the genes is:
```python
PA.plot_gene_filter(pop, offset=1)
```

The generated scatter plot shows the logged gene means against the logged coefficient of variation (standard deviation over mean). A line is fitted to the data with linear regression. The expected slope is -1/2 but the intercept value depends on the data points. The line can be moved thanks to the `offset` parameter. The log10 of the argument passed to this parameter will be added to the line intercept. The default offset value is 1, which doesn't move the line since log10(1) is 0. If the offset value is greater than 1, the line will move up, selecting less genes. If the offset value is less than one, the line will move down, keeping more genes. Everytime this function is ran (with a different offset value for example), the indices of genes to keep is updated in the `pop` object under:
```python
pop['filter_idx']
```

Once the user has selected genes with the `plot_gene_filter` function, they can filter the genes with:
```python
PA.filter(pop, remove_ribsomal=True)
```
This function filters out the genes and then logs the data values in all gene expression matrices. The filtered matrices are stored under a different entry. The filtered matrix of `sample1` is accessed with:
```python
pop['samples']['sample1']['M_norm']
```
The non-filtered matrices are also logged in-place.

If the `remove_ribsomal` parameter is `True`, genes that start with `RPS` or `RPL` are removed from the filtered genes. If `False`, those ribosomal genes are kept in the filtered matrices. The filtered gene list can be found under:
```python
pop['filtered_genes']
```

## (Optional) Remove red blood cells

If the user is using blood data, they can decide to remove the possible red blood cells with the `removeRBC` function. This function uses different marker genes for human samples (HBB, HBA1, HBA2) and mouse samples (HBB-BT, HBB-BS, HBA-A1, HBA-A2). The function is used as follows:
```python
PA.removeRBC(pop, species='human')
```
The `species`parameter can either be `human` or `mouse`.

## Perform dimensionality reduction

Dimensionality reduction allows to drastically reduce the number of descriptive variables. The variables in scRNAseq data are the genes. Even though the genes have been filtered in one of previous steps (consider it a cleaning step to keep the most informative variables), the data is still in a high-dimensional space (hundreds or thousands of genes) and still has high complexity, which makes it harder to analyze and understand. Dimensionality reduction creates macro variables that encapsulates the original variables and could be seen as features or programs.

The first dimensionality reduction method to be used is orthogonal nonnegative matrix factorization (oNMF). This version of NMF applies orthogonality constraints on the feature space. It factors a gene expression matrix D of size (m genes, n cells) into two matrices W (m genes, k features) and H (k features, n cells) so that W.H approximates D. W is called the feature space. The function to run oNMF is:
```python
PA.onmf(pop, ncells=5000, nfeats=[5,7], nreps=3, niter=500)
```
`ncells` is an integer indicates how many cells should be randomly sampled from the loaded data to build the feature spaces. If that number is greater than the number of cells in the data, it will be adjusted down to the total number of cells. If the user wants to check the total number of cells in the loaded data, they can use `PA.print_ncells(pop)`. The `ncells` argument defaults to 2000.

`nfeats` is an integer or a list of integers. It indicates how many features to build with oNMF. If a list of integers is provided, the algorithm will try each value individually.

`nreps` (integer) is how many times to repeat each possible `k` (number of features) from `nfeats`. For example, if `nfeats` is [5,7,9] and `nreps` is 2, 6 distinct feature spaces will be built (two feature spaces with 5 features, two feature spaces with 7 features and two feature spaces with 9 features).

After the feature spaces have been computed in parallel, each feature space is scaled by scaling its features by their respective l2-norm. This helps uniformizing the range of values of the projected data in a feature space.

The goal is to select the best representing feature space among all the computed feature spaces. PopAlign selects the feature space that minimizes the mean square error between the original data and the reconstructed version of the data using the feature space. To do so, the normalized gene expression data matrix D is projected onto each individual feature space W<sub>j</sub>. The projected data can be notated H<sub>j</sub>. The final feature space W is selected so that:

![](https://latex.codecogs.com/svg.latex?W&space;=&space;argmin(\frac{1}{n}\sum_{i}^{n}(D-Wj.Hj)^{2}))

Gene set enrichment analysis (GSEA) is run for each feature from the feature space.

