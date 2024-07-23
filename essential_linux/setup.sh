#!/bin/bash
rm -r shape_proj_util
git clone https://github.com/kaifengZheng/shape_proj_util.git

folderpath=$1
mkdir $folderpath
cp -r config.toml NanoGene.py shape_proj_util ../$folderpath
