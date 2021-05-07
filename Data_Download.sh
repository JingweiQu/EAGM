#!/bin/bash

if [ ! -d "./data/PascalVOC" ]; then
mkdir ./data/PascalVOC
fi
if [ ! -d "./data/Willow" ]; then
mkdir ./data/Willow
fi
if [ ! -d "./data/CMUHouse" ]; then
mkdir ./data/CMUHouse
fi

cd ./data/PascalVOC
echo -e "\e[35mDownloading Berkeley annotations for PascalVOC\e[0m"
if [ ! -f "voc2011_keypoints_Feb2012.tgz" ]; then
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz
fi
tar xzf voc2011_keypoints_Feb2012.tgz
echo -e "\e[35m... Done\e[0m"

echo -e "\e[35mDownloading PascalVOC\e[0m"
if [ ! -f "VOCtrainval_25-May-2011.tar" ]; then
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar
fi
tar xf VOCtrainval_25-May-2011.tar
mv TrainVal/VOCdevkit/VOC2011/JPEGImages ./
rm -rf TrainVal
echo -e "\e[35m... Done\e[0m"

cd ../Willow
echo -e "\e[35mDownloading Willow Object\e[0m"
if [ ! -f "WILLOW-ObjectClass_dataset.zip" ]; then
wget http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip
fi
unzip -q WILLOW-ObjectClass_dataset.zip
mv WILLOW-ObjectClass/* ./
rmdir WILLOW-ObjectClass
echo -e "\e[35m... Done\e[0m"

cd ../CMUHouse
echo -e "\e[35mDownloading CMU House Sequence\e[0m"
unzip -q images.zip
echo -e "\e[35m... Done\e[0m"