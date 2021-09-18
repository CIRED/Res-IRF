from setuptools import setup

with open("documentation/source/README.md", 'r') as f:
    long_description = f.read()

setup(
   name='Res-IRF',
   version='4.0',
   description='Building Energy Model',
   license="GNU GENERAL PUBLIC LICENSE",
   long_description=long_description,
   author='L.G. Giraudet, L. Vivier',
   author_email='vivier@centre-cired.fr',
   url="https://github.com/lucas-vivier/Res-IRF",
   packages=['project'],
   install_requires=['pandas', 'numpy', 'scipy'],
   scripts=[
           ]
)
