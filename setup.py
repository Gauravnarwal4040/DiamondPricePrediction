from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path : str) -> list[str]:
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n",'') for i in requirements]
        return requirements

setup(
    name = 'DiamondPricePrediction',
    version = '0.0.1',
    author = 'Gaurav',
    author_email = "gauravnarwal4040@gmail.com",
    install_requires = get_requirements('requirements.txt'),
    packages = find_packages()


)