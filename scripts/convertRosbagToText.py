#!/usr/bin/env python

import sys
import rosbag

if len(sys.argv) != 3:
    print("Invalid number of arguments!")
    print("USAGE: {} <input_rosbag_path> <output_text_path>".format(sys.argv[0]))
    sys.exit(1)

bag_path = sys.argv[1]
with rosbag.Bag(bag_path) as bag:
    for _, _msg, _ in bag.read_messages(topics=["smplpp/motion"]):
        msg = _msg

text_path = sys.argv[2]
with open(text_path, mode='w') as f:
    for data in msg.data_list:
        f.write("{}\n".format(list(data.theta)))
