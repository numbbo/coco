#!/bin/bash
FILES=./mixint_f_??.png
for f in $FILES
do
	echo "$f"
    FULL_FILENAME=$f
    FILENAME=${FULL_FILENAME##*/}
    convert -resize 2039 -quality 10 $f ${FILENAME%%.*}_small.png
done