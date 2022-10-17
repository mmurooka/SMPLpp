#! /usr/bin/env python

import numpy as np

import rospy
from tf import transformations
from geometry_msgs.msg import TransformStamped, Pose
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

def toPoseMsg(pos):
    pose_msg = Pose()
    pose_msg.position.x = pos[0]
    pose_msg.position.y = pos[1]
    pose_msg.position.z = pos[2]
    pose_msg.orientation.w = 1.0
    return pose_msg

class IkTargetManager(object):
    """"Node to manage IK targets."""
    def __init__(self):
        # setup marker
        self.setupInteractiveMarker()

        # setup publisher
        self.transform_pub = rospy.Publisher("ik_target_pose", TransformStamped, queue_size=1)

    def setupInteractiveMarker(self):
        # make server
        self.im_server = InteractiveMarkerServer("ik_target_manager")

        # add marker
        self.im_server.insert(
            self.makeInteractiveMarker(
                name="LeftHand",
                pos=[0.5, 0.5, 1.0]),
            self.interactivemarkerFeedback)
        self.im_server.insert(
            self.makeInteractiveMarker(
                name="RightHand",
                pos=[0.5, -0.5, 1.0]),
            self.interactivemarkerFeedback)
        self.im_server.insert(
            self.makeInteractiveMarker(
                name="LeftFoot",
                pos=[0.0, 0.2, 0.0]),
            self.interactivemarkerFeedback)
        self.im_server.insert(
            self.makeInteractiveMarker(
                name="RightFoot",
                pos=[0.0, -0.2, 0.0]),
            self.interactivemarkerFeedback)

        # apply to server
        self.im_server.applyChanges()

    def makeInteractiveMarker(self, name, pos):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.name = name
        int_marker.pose = toPoseMsg(pos)
        int_marker.scale = 0.3

        def makeAxisMarker(quat, color):
            marker_msg = Marker()
            marker_msg.type = Marker.CYLINDER
            pos_msg = marker_msg.pose.position
            pos_msg.x, pos_msg.y, pos_msg.z = \
                np.matmul(transformations.quaternion_matrix(quat)[0:3, 0:3], np.array([0.0, 0.0, 0.05]))
            ori_msg = marker_msg.pose.orientation
            ori_msg.x, ori_msg.y, ori_msg.z, ori_msg.w = quat
            marker_msg.scale.x = 0.02;
            marker_msg.scale.y = 0.02;
            marker_msg.scale.z = 0.1;
            color_msg = marker_msg.color
            color_msg.r, color_msg.g, color_msg.b, color_msg.a = color
            return marker_msg

        control = InteractiveMarkerControl()
        control.name = "axis"
        control.always_visible = True
        control.markers = [
            makeAxisMarker(transformations.quaternion_from_euler(0, np.pi/2, 0), [0.9, 0.1, 0.1, 1.0]),
            makeAxisMarker(transformations.quaternion_from_euler(-np.pi/2, 0, 0), [0.1, 0.9, 0.1, 1.0]),
            makeAxisMarker(transformations.quaternion_from_euler(0, 0, 0), [0.1, 0.1, 0.9, 1.0])]
        int_marker.controls.append(control)

        # move_x control
        control = InteractiveMarkerControl()
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        ori = control.orientation
        ori.x, ori.y, ori.z, ori.w = transformations.quaternion_from_euler(0, 0, 0)
        int_marker.controls.append(control)

        # move_y control
        control = InteractiveMarkerControl()
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        ori = control.orientation
        ori.x, ori.y, ori.z, ori.w = transformations.quaternion_from_euler(0, 0, np.pi/2)
        int_marker.controls.append(control)

        # move_z control
        control = InteractiveMarkerControl()
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        ori = control.orientation
        ori.x, ori.y, ori.z, ori.w = transformations.quaternion_from_euler(0, -np.pi/2, 0)
        int_marker.controls.append(control)

        # rotate_x control
        control = InteractiveMarkerControl()
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        ori = control.orientation
        ori.x, ori.y, ori.z, ori.w = transformations.quaternion_from_euler(0, 0, 0)
        int_marker.controls.append(control)

        # rotate_y control
        control = InteractiveMarkerControl()
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        ori = control.orientation
        ori.x, ori.y, ori.z, ori.w = transformations.quaternion_from_euler(0, 0, np.pi/2)
        int_marker.controls.append(control)

        # rotate_z control
        control = InteractiveMarkerControl()
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control.orientation_mode = InteractiveMarkerControl.FIXED
        ori = control.orientation
        ori.x, ori.y, ori.z, ori.w = transformations.quaternion_from_euler(0, -np.pi/2, 0)
        int_marker.controls.append(control)

        return int_marker

    def interactivemarkerFeedback(self, feedback):
        # set message
        transform_msg = TransformStamped()
        transform_msg.header = feedback.header
        transform_msg.child_frame_id = feedback.marker_name
        transform_msg.transform.translation.x = feedback.pose.position.x
        transform_msg.transform.translation.y = feedback.pose.position.y
        transform_msg.transform.translation.z = feedback.pose.position.z
        transform_msg.transform.rotation = feedback.pose.orientation

        # publish message
        self.transform_pub.publish(transform_msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("ik_target_manager", anonymous=False)

    manager = IkTargetManager()

    manager.spin()
