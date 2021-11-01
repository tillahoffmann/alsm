master_doc = 'README'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]
project = 'alsm'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'pystan': ('https://pystan.readthedocs.io/en/latest/', None),
}
napoleon_custom_sections = [('Returns', 'params_style')]
html_theme = 'nature'
