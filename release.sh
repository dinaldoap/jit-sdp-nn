#!/bin/bash

cd jitsdp
pyinstaller main.py  --onefile --name jitsdp \
                                    --hidden-import='pkg_resources.py2_warn' \
                                    --hidden-import='sklearn.utils._cython_blas' \
                                    --hidden-import='sklearn.neighbors._typedefs' \
                                    --hidden-import='sklearn.neighbors._quad_tree' \
                                    --hidden-import='sklearn.tree._utils' \
                                    --hidden-import='scipy.special.cython_special' \
                                    --hidden-import='skmultiflow.metrics._confusion_matrix'