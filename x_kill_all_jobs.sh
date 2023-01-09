#!/bin/bash

for j in $(squeue --user="$(id -u -n)" -n hex-generator --noheader --format=%i) ; do
  scancel $j
done

for j in $(squeue --user="$(id -u -n)" -n hex-trainer --noheader --format=%i) ; do
  scancel $j
done