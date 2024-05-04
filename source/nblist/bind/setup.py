import os
import subprocess
from distutils.core import Extension, setup
from typing import List
from pathlib import Path
import numpy as np


nblist_name:str = "nblist"
nblist_root_dir:str = Path(__file__).parent.absolute().parent.absolute()
nblist_include_dir:str = os.path.join(nblist_root_dir, "include")
nblist_src_dir:str = os.path.join(nblist_root_dir, "src")
nblist_bind_dir:str = os.path.join(nblist_root_dir, "bind")
nblist_bind_gen_dir:str = os.path.join(nblist_root_dir, "bind", "gen")

print(nblist_root_dir)
class NblistSwigExecutor(object):
    def run(self, cxx_wrap_file:str, nblist_bind_gen_dir:str):
        subprocess.call(
            [
                "swig",
                "-c++", "-python",
                "-o", cxx_wrap_file,
                "-outdir", nblist_bind_gen_dir,
                "{0}.i".format(os.path.join(nblist_bind_dir, nblist_name))
            ]
        )


class NblistFilesFinder(object):
    def find_cc_lst(self):
        cc_lst:List[str] = []
        for tmp_file_name in os.listdir(nblist_src_dir):
            cc_lst.append( os.path.join(nblist_src_dir, tmp_file_name) )
        return [] #cc_lst

    def find_wrap_cxx_file(self):
        return os.path.join(nblist_bind_gen_dir, "{0}_wrap.cxx".format(nblist_name))



nblist_cc_lst:List[str] = NblistFilesFinder().find_cc_lst()
nblist_wrap_cxx_file:str = NblistFilesFinder().find_wrap_cxx_file()
NblistSwigExecutor().run(
    cxx_wrap_file=nblist_wrap_cxx_file, 
    nblist_bind_gen_dir=nblist_bind_gen_dir)

nblist_module = Extension(
    name="_nblist",
    sources=[
        *nblist_cc_lst,
        os.path.join(nblist_bind_dir, "{0}_bind.cc".format(nblist_name)),
        nblist_wrap_cxx_file
        ],
    include_dirs=[
        nblist_include_dir,
        nblist_bind_dir,
        np.get_include()
    ]
)


setup(
    name="nblist",
    ext_modules=[
        nblist_module,
    ],
    py_modules=[
        "nblist"
    ]
)
