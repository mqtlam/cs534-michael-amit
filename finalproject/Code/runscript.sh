#! /bin/bash

clear
echo
for((i=300;i<364;i++))
do
	a="video"
	b=$(printf "%04d" $i)
	c=".avi"
	d="$a$b$c"
	./wideDetector $d
done
echo
echo "Done!!"
