#!/bin/bash

cd jitsdp
pyinstaller __main__.py  --onefile --name jitsdp \
                                    --hidden-import='pkg_resources.py2_warn' \
                                    --hidden-import='sklearn.utils._cython_blas' \
                                    --hidden-import='sklearn.neighbors._typedefs' \
                                    --hidden-import='sklearn.neighbors._quad_tree' \
                                    --hidden-import='sklearn.tree._utils'

echo '# -*- mode: python ; coding: utf-8 -*-' >> jitsdp_fixed.spec
echo 'import sys' >> jitsdp_fixed.spec
echo 'sys.setrecursionlimit(10000)' >> jitsdp_fixed.spec
cat jitsdp.spec >> jitsdp_fixed.spec

pyinstaller jitsdp_fixed.spec