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
If the user hasn't loaded any samples, `existing_obj` should be `None`. If samples have already been loaded with `load_samples` or `load_screen`, the object that these functions returned should be given as the argument for this parameter.

* Format 2:
All the samples are stored in one unique matrix file. This format requires the paths to the .mtx file, the gene list, the barcode list and a metadata file. The meta data file must be a .csv file that includes two columns: `cell_barcode` and `sample_id`. The cells from the different samples will be selected in the matrix based on their barcode index.
```python
mymatrix = 'path/to/matrix.mtx'
mybarcodes = 'path/to/barcodes.tsv'
mygenes = 'path/to/genes.tsv'
mymetadata = 'path/to/metadata.csv'
```

The function to be used with that data format is:
```python
pop = PA.load_screen(matrix=mymatrix, 
	barcodes=mybarcodes, 
	metafile=mymetadata, 
	genes=mygenes,
	outputfolder='output_screen')
```
It returns a dictionary that contains the loaded data, genes and various information. That object is used throughout the entire analysis.
The `outputfolder` parameter defines where the results of the analysis will be saved. Its default value is `output`. 
If the user hasn't loaded any samples, `existing_obj` should be `None`. If samples have already been loaded with `load_samples` or `load_screen`, the object that these functions returned should be given as the argument for this parameter.

The sample gene expression matrices are stored individually. To access the gene expression matrix of `sample1`, use the following:
```python
pop['samples']['sample1']['M']
```

The gene labels are stored in the `pop` object:
```python
pop['genes']
```

## Normalize data

The data is normalized to account for sequencing depth differences. This is done by dividing each cell (column in a gene expression matrix) by its transcript count sum. The values are then multiplied by a scaling factor. The scaling factor value can be either set by the user or computed automatically. When computed automatically, multiple factors are tested and the factor that minimizes the difference between the mean of the original data prior to normalization and the mean of the normalized, scaled data is used.

The normalization function is:
```python
PA.normalize(pop, scaling_factor=None)
```
It normalizes the gene expression matrices in-place, i.e. the matrices are directly updated.
The `scaling_factor` parameter can be used to set the scaling factor value. It defaults to None, which means the scaling factor is computed automatically, then applied to the column normalized data.

## Gene filtering

The process of filtering genes is divided into two steps: plotting the genes to select the most variable genes, and then filtering the genes once they have been selected with the first step.

Plotting the genes