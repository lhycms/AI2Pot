from pathlib import Path
from setuptools import setup, find_packages


ai2pot_root_dir:str = Path(__file__).parent.absolute()


### Part . Set up ai2pot
setup(
    name="ai2pot",
    version="v1.0",
    author="Liu Hanyu",
    author_email="domainofbuaa@gmail.com",
    url="https://github.com/lhycms/AI2Pot",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
            "cmake==3.26.4"
            "pymatgen==2024.7.18",
            "ase==3.23.0",
            "numpy==1.26.4",
            "pybind11==2.11.1",
            #"torch-2.4.0+cpu"
    ],
    extras_requires={
        "presentation": ["chgnet"]
    }
)
