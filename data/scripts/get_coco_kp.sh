#!/bin/bash
# Example usage: bash data/scripts/get_coco_kp.sh

# Make dataset directories
mkdir -p data/datasets/coco/images

# Download/unzip annotations
d='data/datasets/coco' # unzip directory
f1='annotations_trainval2017.zip'
f2='image_info_test2017.zip'
url=http://images.cocodataset.org/annotations/
for f in $f1 $f2; do
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f && unzip -q $f -d $d && rm $f &
done

# Download/unzip images
d='data/datasets/coco/images' # unzip directory
url=http://images.cocodataset.org/zips/
f1='train2017.zip' # 19G, 118k images
f2='val2017.zip'   # 1G, 5k images
f3='test2017.zip'  # 7G, 41k images (optional)
for f in $f1 $f2 $f3; do
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f && unzip -q $f -d $d && rm $f &
done
wait # finish background tasks
