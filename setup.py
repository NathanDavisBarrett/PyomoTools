from setuptools import setup, find_packages

setup(
    name='PyomoTools',
    version='0.2.5',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pyomo',
        'pandas',
        'pwlf',
        'matplotlib',
        'numpy',
        'pytest',
        'XlsxWriter',
        'openpyxl',
        'highspy'
      ]
)
