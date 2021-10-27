master_doc = 'README'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]
project = 'alsm'
intersphinx_mapping = {
    'python': ('http://docs.python.org/2', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('http://matplotlib.sourceforge.net/', None),
}
