#!/bin/sh

NPARTS="1000 2000 3000 4000 5000 7500 10000 15000 20000"
LOOPS=10000
LOGEXT=dat

/bin/rm -f external.$LOGEXT internal.$LOGEXT
for particles in $NPARTS
do
    ./external $LOOPS $particles | grep -v CPU >> external.$LOGEXT
    ./internal $LOOPS $particles | grep -v CPU >> internal.$LOGEXT
done

# time to say 'Good bye' ;-)
#
exit 0

