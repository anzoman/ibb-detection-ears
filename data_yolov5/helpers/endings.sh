#!/bin/bash

counter=1
for filename in ./test/*; do
    mv "$filename" "${filename}.png"
    counter=$((counter+1))
done
