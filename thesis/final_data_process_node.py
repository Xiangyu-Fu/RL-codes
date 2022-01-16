#!/usr/bin/env python

import math
import rospy
import message_filters
import numpy as np
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64, Bool, Float64MultiArray


class DataProcess(object):
    def __init__(self):
        # init node
        self.node_name = 'data_process'
        rospy.init_node(self.node_name)

        # set publish rate
        self.r = rospy.Rate(10)
        self.tyre_radius = 0.3671951254

        # Subscribers
        imu_data = message_filters.Subscriber("/catvehicle/imu", Imu)
        joint_data = message_filters.Subscriber("/catvehicle/joint_states", JointState)

        sync = message_filters.ApproximateTimeSynchronizer([imu_data, joint_data], 10, 1)
        sync.registerCallback(self.data_callback)

        # Publishers
        self.pub1 = rospy.Publisher("/SSAE/processed_data", Float64MultiArray, queue_size=1)

        rospy.loginfo("data processing...")

    def data_callback(self, imu_data, joint_states):
        try:
            # get sensor data from value
            yaw_rate = imu_data.angular_velocity.z  # read yaw rate from imu sensor
            acc_x = imu_data.linear_acceleration.x  # Read the linear acceleration in two direction
            acc_y = imu_data.linear_acceleration.y

            vel_x = joint_states.velocity[0] * self.tyre_radius
            steering_angle = joint_states.position[2]
            processed_data = Float64MultiArray()
            processed_data.data = [acc_x, acc_y, yaw_rate, vel_x, steering_angle]
            self.pub1.publish(processed_data)
            self.r.sleep()
        except(IndexError):
            pass


if __name__ == '__main__':
    try:
        DataProcess()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass