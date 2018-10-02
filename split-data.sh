#!/bin/bash
cd selfies-training-data/1
COUNTER=0
for FILE in *.jp*; do
	REMAINDER=$(( $COUNTER % 4))
	if [[ "$REMAINDER" == 0 ]] ; then
		mv "$FILE" ../../selfies-test-data/1/
	fi
	((COUNTER++))
done
