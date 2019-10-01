from setuptools import find_packages
from setuptools import setup


__version__ = '0.0.1'
REQUIRED_PACKAGES = [
    'tf_yarn'
]


setup(
    name="tf_collective_all_reduce",
    version="0.0.1",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=False
)
