#!/bin/bash

rm -R public/*
hugo
cd public 
git add *
git commit -a -m "Build $(date)"
git push -f origin master
