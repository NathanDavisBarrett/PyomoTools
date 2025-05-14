from setuptools import setup, find_packages

import subprocess
import sys
from warnings import warn

def install_optional_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        warn(f"Warning: {package} could not be installed. Continuing without it.")

# Attempt to install the optional package
install_optional_package('pypoman')

setup(
    name='PyomoTools',
    version='0.3.0',
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
        'highspy',
        'scipy',
        'pyyaml'
      ],
      extras_require={
        'optional': ['pypoman']
      }
)
