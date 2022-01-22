from setuptools import setup,find_packages
setup(
    name='main',
    description='package for style transfer',
    version= 1.0,
    author='amine ammor',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)