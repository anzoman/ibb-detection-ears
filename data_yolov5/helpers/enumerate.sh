#!/bin/bash

counter=1
for filename in ./test/*; do
    mv "$filename" "$(printf '%04d' "$counter" 2> /dev/null)"
    counter=$((counter+1))
done
