#!/bin/bash
printf -v name "sleepVolunteer%03d.bag" $1
echo "Recording to file $name"
rosbag record -O $name --lz4 /kinect2/sd/image_depth/compressed /kinect2/sd/image_ir/compressed /kinect2/hd/image_color/compressed