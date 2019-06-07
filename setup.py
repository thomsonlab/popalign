from distutils.core import setup

setup(
    name="popalign",
    version="0.1",
    packages=[
        "popalign",
    ],
    install_requires=[
        "numpy",
        "pandas",
	"scipy",
	"sklearn",
	"plotly",
	"matplotlib",
	"seaborn",
	"adjustText",
	"ipywidgets"	
    ]
)
