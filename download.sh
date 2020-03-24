#!/bin/bash

# project dir
project=jit-sdp-nn
# store dir
store=mlruns-server
# clean up
rm -rf ${store}
# compress
ssh gpu tar -C ${project} -zcvf ${project}/mlruns.tar.gz mlruns/
# download
scp -rC gpu:${project}/mlruns.tar.gz .
# create dir
mkdir ${store}
# decompress
tar -C ${store} -zxvf mlruns.tar.gz
# clean up
rm mlruns.tar.gz
ssh gpu rm ${project}/mlruns.tar.gz
# start mlflow
mlflow ui --backend-store-uri ${store}