#!/bin/bash

# project dir
project=jit-sdp-nn
# store dir
store=mlruns-server
# compress
ssh gpu "cd ${project}/mlruns && tar -zcvf ../mlruns.tar.gz *"
# download
scp -rC gpu:${project}/mlruns.tar.gz .
# recreate store
rm -rf ${store}
mkdir ${store}
# decompress
tar -C ${store} -zxvf mlruns.tar.gz
# change paths
find mlruns-server -name meta.yaml | xargs sed --in-place --expression "s/file:\/\/\/.*\/mlruns\//file:\/\/\/workspace\/${store}\//g"
# clean up
rm mlruns.tar.gz
ssh gpu rm ${project}/mlruns.tar.gz
# start mlflow
mlflow ui --backend-store-uri ${store}