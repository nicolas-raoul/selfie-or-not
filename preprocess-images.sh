set -x
SIZE=100

mkdir data/thumbnails-${SIZE}
mkdir data/thumbnails-${SIZE}/training
mkdir data/thumbnails-${SIZE}/training/0
mkdir data/thumbnails-${SIZE}/training/1
mkdir data/thumbnails-${SIZE}/test
mkdir data/thumbnails-${SIZE}/test/0
mkdir data/thumbnails-${SIZE}/test/1

mogrify -path data/thumbnails-${SIZE}/training/0/ -auto-orient -format png -resize ${SIZE}x${SIZE}\! -set colorspace Gray -separate -average data/real/training/0/*
mogrify -path data/thumbnails-${SIZE}/training/1/ -auto-orient -format png -resize ${SIZE}x${SIZE}\! -set colorspace Gray -separate -average data/real/training/1/*
mogrify -path data/thumbnails-${SIZE}/test/0/ -auto-orient -format png -resize ${SIZE}x${SIZE}\! -set colorspace Gray -separate -average data/real/test/0/*
mogrify -path data/thumbnails-${SIZE}/test/1/ -auto-orient -format png -resize ${SIZE}x${SIZE}\! -set colorspace Gray -separate -average data/real/test/1/*
