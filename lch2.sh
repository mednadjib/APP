#!/bin/bash
for i in `seq 705 8 82783`;
do

for j in {0..7}
do
index=$(expr $i \+ $j )
python -c 'from ROI.Texture import texture; texture().__init__()' '/home/imene/MY-APPLICATION/data/coco'  'MSCOCO' 'trainval2014,minival2014,test2017' $index  &
#python -c 'from ROI.Texture import texture; texture().__init__()' ' /home/imene/CornerNet-master/data/pascal_voc0712'  'PASCAL_VOC' 'trainval0712,test07' $index  &
done
wait 
done