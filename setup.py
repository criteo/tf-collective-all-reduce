from setuptools import find_packages
from setuptools import setup


__version__ = '0.0.1'
REQUIRED_PACKAGES = [
    'tensorflow == 1.12.2'
]


setup(
    name="tf-collective-ops",
    version="0.0.1",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    zip_safe=False
)
