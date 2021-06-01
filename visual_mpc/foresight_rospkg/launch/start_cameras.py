#!/usr/bin/env python
import argparse
import os
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="orders and launches cameras")
    # parser.add_argument("--cam_order", type=int, nargs='+', default=[i for i in range(3)],
    #                     help="list of camera video stream providers")
    # parser.add_argument("--topic_names", type=str, nargs='+',
    #                     default=['front', 'left', 'right_side'],
    #                     help="list of camera topic names")
    parser.add_argument("--cam_order", type=int, nargs='+', default=[0,4],
                        help="list of camera video stream providers")
    parser.add_argument("--topic_names", type=str, nargs='+',
                        default=['left', 'right'],
                        help="list of camera topic names")
    parser.add_argument('--visualize', action='store_true', default=False, help="if flag supplied image_view will show")
    parser.add_argument('--flip-vertical', action='store_true', default=False, help="if flag supplied will flip vertically")
    parser.add_argument('--flip-horizontal', action='store_true', default=False, help="if flag supplied will flip horizontal")



    args = parser.parse_args()

    assert len(args.cam_order) == len(args.topic_names), "Number of providers should equal number of topics"

    base_call = "roslaunch foresight_rospkg camera.launch video_stream_provider:={} camera_name:={} visualize:={} fps:=20 flip_vertical:={} flip_horizontal:={} buffer_queue_size:=20 &"
    visualize_str = "false"
    if args.visualize:
        visualize_str = "true"

    for provider, cam_name in zip(args.cam_order, args.topic_names):
        os.system(base_call.format(provider, cam_name, visualize_str, args.flip_vertical, args.flip_horizontal))
        time.sleep(5)
