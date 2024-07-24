#!/bin/bash
filename=$1
mkdir ../$filename
cp -r config.toml NanoGene.py shape_proj_util ../$filename
