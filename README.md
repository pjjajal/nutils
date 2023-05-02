# nn-utils — a collection of utilities for studying PyTorch modules
## Local Installation Instructions
1) Update 'build' module
`python -m pip install --upgrade build`
2) Use 'build' module to create an installable .whl file
`python -m build`
3) Install your .whl file so you can access 'nutils' as a module in your Python environment. Replace <filename> with the generated .whl file
`python -m pip install dist/<filename>.whl --no-deps`
4) Open a Python interpreter and execute `import nutils` - it should work!