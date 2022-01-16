#!/usr/bin/env python

import math
import rospy
import numpy as np
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64, Bool, Float64MultiArray


class DataProcess(object):
    def __init__(self):
        self.node_name = 'data_process'
        rospy.init_node(self.node_name)

        self.r = rospy.Rate(5)
        self.tyre_radius = 0.3671951254
        self.yaw_rate = 0
        self.acc_x = 0
        self.acc_y = 0

        # Subscribers
        self.sub1 = rospy.Subscriber("/catvehicle/imu", Imu, self.imu_callback)
        self.sub2 = rospy.Subscriber("/catvehicle/joint_states", JointState, self.joint_callback)
        # Publishers
        self.pub1 = rospy.Publisher("/catvehicle/processed_data", Float64MultiArray, queue_size=1)

        rospy.loginfo("Waiting for topics...")

    def imu_callback(self, imu_data):
        # read yaw rate from imu sensor
        self.yaw_rate = imu_data.angular_velocity.z

        # Read the linear acceleration in two direction
        self.acc_x = imu_data.linear_acceleration.x
        self.acc_y = imu_data.linear_acceleration.y
        return


    def joint_callback(self, joint_states):
        try:
            vel_x = joint_states.velocity[0] * self.tyre_radius
            steering_angle = joint_states.position[2]
            processed_data = Float64MultiArray()
            processed_data.data = [self.acc_x, self.acc_y, self.yaw_rate, vel_x, steering_angle]
            self.pub1.publish(processed_data)
        except(IndexError):
            pass


if __name__ == '__main__':
    try:
        DataProcess()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
