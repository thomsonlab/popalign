

```python
import sys
sys.path.append('../popalign/')
import popalign as PA
import importlib
```


```python
PA.__file__
```




    '../popalign/popalign.py'




```python
# Load data (example)
LOAD = 'samples'

if LOAD == 'samples':
    mysamples = {
        'CTRL' : '../data/samples/PBMC.mtx',
        'GMCSF_1ng/ml' : '../data/samples/GMCSF.mtx',
        'IFNG_1ng/ml' : '../data/samples/IFNG.mtx',
        'IL2_10ng/ml' : '../data/samples/IL2.mtx',
        'CD40L_20ng/ml' : '../data/samples/CD40L.mtx',
    }
    mygenes = '../data/samples/genes.tsv'
    pop = PA.load_samples(samples=mysamples, 
                          genes=mygenes)
    
elif LOAD == 'screen':
    mymatrix = 'data/screen/drug_screen/pbmcmult4cd3minus.mtx'
    mybarcodes = 'data/screen/drug_screen/barcodes.tsv'
    mygenes = 'data/screen/drug_screen/features.tsv'
    mymetadata = 'data/screen/drug_screen/meta.csv'
    pop = PA.load_screen(matrix=mymatrix, 
                         barcodes=mybarcodes, 
                         metafile=mymetadata, 
                         genes=mygenes)
```


```python
# Perform column normalization
# Find best normalization factor
PA.normalize(pop)
```

    Performing column normalization
    Finding best normalization factor



```python
# Plot genes (log cv ~ log cv) and filtering line --use multiple times to find best offset (usually between .7 and 1.5)
PA.plot_gene_filter(pop, offset=1.1)
```

    1490 genes selected



![png](output_4_1.png)



```python
# Gene filter the data with the last offset used in the previous step
PA.filter(pop)
```

    Removing ribosomal genes
    Filtering genes ang logging data



```python
# Remove red blood cells from the data
PA.removeRBC(pop, 'human')
```

    CTRL 1349 cells kept out of 1377
    GMCSF_1ng/ml 1424 cells kept out of 1444
    IFNG_1ng/ml 1125 cells kept out of 1144
    IL2_10ng/ml 1633 cells kept out of 1654
    CD40L_20ng/ml 2326 cells kept out of 2358



```python
# Generate multiple feature spaces and pick the best one based on reconstruction error
# Run GSEA on each feature
# Generate QC plots
PA.onmf(pop, ncells=500, nfeats=[5,7], nreps=3, niter=300)
```

    Computing W matrices
    ......... Iteration #0
    ......... Iteration #0
    ......... Iteration #0
    ......... Iteration #0
    ......... Iteration #0
    ......... Iteration #0
    ......... Iteration #100
    ......... Iteration #100
    ......... Iteration #100
    ......... Iteration #100
    ......... Iteration #100
    ......... Iteration #100
    ......... Iteration #200
    ......... Iteration #200
    ......... Iteration #200
    ......... Iteration #200
    ......... Iteration #200
    ......... Iteration #200
    ......... Iteration #299
    Orthogonal NMF performed with 300 iterations
    
    ......... Iteration #299
    ......... Iteration #299
    Orthogonal NMF performed with 300 iterations
    Orthogonal NMF performed with 300 iterations
    
    
    ......... Iteration #299
    Orthogonal NMF performed with 300 iterations
    
    ......... Iteration #299
    Orthogonal NMF performed with 300 iterations
    
    ......... Iteration #299
    Orthogonal NMF performed with 300 iterations
    
    Computing reconstruction errors
    Progress: 1 of 6
    Progress: 2 of 6
    Progress: 3 of 6
    Progress: 4 of 6
    Progress: 5 of 6
    Progress: 6 of 6
    Retrieving W with lowest error
    Starting gene set enrichment analysis
    GSEA progress: 1 of 7
    GSEA progress: 2 of 7
    GSEA progress: 3 of 7
    GSEA progress: 4 of 7
    GSEA progress: 5 of 7
    GSEA progress: 6 of 7
    GSEA progress: 7 of 7



```python
# Build a Gaussian Mixture model for each sample
# Type the models subpopulations
importlib.reload(PA)
PA.build_gmms(pop, ks=(5,20), nreps=2, reg_covar=False, rendering='grouped', types=None)
```

    Building model for CTRL (1 of 5)
    Building model for GMCSF_1ng/ml (2 of 5)
    Building model for IFNG_1ng/ml (3 of 5)
    Building model for IL2_10ng/ml (4 of 5)
    Building model for CD40L_20ng/ml (5 of 5)
    Rendering models



```python
# Calculate all the subpopulations entropies for each samples
PA.entropy(pop)
```


![png](output_9_0.png)



```python
# Align subpopulations of each sample against a reference model's subpopulations
importlib.reload(PA)
PA.align(pop, ref='CTRL', method='conservative')
```


```python
# Rank each sample against a reference sample's model
PA.rank(pop, ref='CTRL', k=100, niter=200, mincells=50)
```


![png](output_11_0.png)



```python
# Build a unique GMM for the samples concatenated together
PA.build_unique_gmm(pop, ks=(5,20), nreps=3, reg_covar=False, types=None)
```


```python
# Generate a query plot
importlib.reload(PA)
PA.plot_query(pop)
```


![png](output_13_0.png)



```python
# Interactive 3D visualization of the data in feature space
import plotly
plotly.offline.init_notebook_mode()
PA.plotfeatures(pop)
```

```python
"""
typelist = list(types.keys())
genelist = np.concatenate([types[t] for t in typelist])

gmm = pop['samples']['CTRL']['gmm'] # get gmm
prediction = gmm.predict(pop['samples']['CTRL']['C']) # prediction the cells assignments for that sample
types = PA.default_types()
genes = pop['genes']

df = pd.DataFrame(columns=range(gmm.n_components), index=typelist) # create empty dataframe

for t in types: # for each cell type in the dictionary
    gidx = [np.where(genes==x)[0][0] for x in types[t] if x in genes] # get the indices of the valid genes for that cell type
    for i in range(gmm.n_components): # for each component of the sample
        cidx = np.where(prediction==i)[0] # get the matching cell indices
        sub = pop['samples']['CTRL']['M'][:,cidx] # subset the normalized data
        sub = sub[gidx,:] # subset the desired genes
        df.at[t,i] = sub.mean() # update the dataframe with the mean of those cells for those genes
"""
```


```python
"""
from sklearn import preprocessing

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x.T)
df = pd.DataFrame(columns=range(gmm.n_components), index=typelist, data=x_scaled.T)
"""
```
