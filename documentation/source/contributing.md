# Contributing


## Updating Documentation

### Overview

Documentation is available: https://lucas-vivier.github.io/Res-IRF/

1. Documentation is run with Sphinx.   
Sphinx concatenate Markdown (.md) files and pick up docstring information (modules and functions information) in .py files and returns .html files.  

2. Documentation is available online thanks to GitHub pages services.  
GitHub pages recognize .html files and create a static website.

Documentation is organized in 2 different folders:
- documentation : folder used by Sphinx:
    - _documentation/source_ contains source (mostly .md files),
    - _documentation/build_ contains .html files.
- docs : used by GitHub pages to create a static website.

### Quickstart

### Update content (modification of .md file or docstring in .py file)
- Run sphinx make file
- Copy and paste _documentation/build_ folder in _docs_ folder
- Visit the documentation website to confirm update has been considered correctly.

### Add content (adding .md file)
- Need to add file name in _documentation/source/index.rst_ file 







