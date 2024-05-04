import os
import subprocess
from pathlib import Path
from setuptools import setup, find_packages


ai2pot_root_dir:str = Path(__file__).parent.absolute()


### Part . Set up nblist
nblist_name:str = "nblist"
nblist_bind_dir:str = os.path.join(ai2pot_root_dir, "source", "nblist", "bind")
nblist_bind_gen_dir:str = os.path.join(nblist_bind_dir, "gen")
subprocess.call([
    "mkdir", "-p",
    nblist_bind_gen_dir
])
subprocess.call([
    "python",
    "{0}".format( os.path.join(nblist_bind_dir, "setup.py") ),
    "build_ext",
    "--build-lib={0}".format( nblist_bind_gen_dir )
])


### Part . Set up ai2pot
setup(
    name="ai2pot",
    version="v1.0",
    author="Liu Hanyu && LONXUN QUANTUM",
    author_email="domainofbuaa@gmail.com",
    url="https://github.com/lhycms/AI2Pot",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
            "pymatgen>=2022.11.7",
            "numpy>=1.23.5",
            "prettytable>=3.5.0",
            #"dpdata>=0.2.13",
            "click>=8.1.3",
            "joblib>=1.2.0",
            "h5py>=3.8.0",
            "pybind11>=2.11.1",
    ]
)
