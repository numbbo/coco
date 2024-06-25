import numpy as np
from setuptools import Extension, setup

extensions = []
extensions.append(Extension(name="cocoex.interface",
                            sources=["src/cocoex/coco.c", "src/cocoex/interface.pyx"],
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                            include_dirs=[np.get_include()]))
extensions.append(Extension(name="cocoex.function",
                            sources=["src/cocoex/coco.c", "src/cocoex/function.pyx"],
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                            include_dirs=[np.get_include()]))

setup(ext_modules=extensions)
