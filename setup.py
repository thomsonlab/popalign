from distutils.core import setup

setup(
    name="popalign",
    version="0.1",
    author="Paul Rivaud",
    author_email='paulrivaud.info@gmail.com",
    keywords="popalign",
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
