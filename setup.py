from setuptools import setup, find_packages

setup(
    name='force_field_models'
    , version='0.2.2'
    , packages=find_packages()
    , package_data={'': ['*.yaml', '*.csv']}
    , include_package_data=True
    , url='https://github.com/deeporiginbio/smartdatafactory-experiments'
)