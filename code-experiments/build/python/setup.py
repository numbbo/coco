import numpy as np
from setuptools import Extension, setup

extensions = []
extensions.append(Extension(name="cocoex.interface",
                            sources=["src/cocoex/coco.c", "src/cocoex/interface.pyx"],
                            include_dirs=[np.get_include()]))

setup(ext_modules=extensions)
