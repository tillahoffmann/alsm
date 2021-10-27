from setuptools import find_packages, setup


setup(
    name='alsm',
    packages=find_packages(),
    version='0.1.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'pystan>=3',
    ],
    extras_require={
        'tests': [
            'flake8',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
        ]
    }
)
