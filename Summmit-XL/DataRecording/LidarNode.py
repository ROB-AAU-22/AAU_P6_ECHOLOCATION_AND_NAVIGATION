import rospy
from sensor_msgs.msg import LaserScan
import os
import json


class LaserScanListener:
    def __init__(self, directory):
        self.laser_data = None
        self.directory = directory  # Directory where the JSON files will be stored
        self.angle_increment = 0.00436333334073  # Assuming this is the correct increment value

    def laser_callback(self, msg):
        """Callback function that receives LaserScan messages."""
        self.laser_data = list(msg.ranges)
        rospy.signal_shutdown("Laser data received")  # Shutdown the node after data is received

    def receive_laser_scan(self):
        """Start the ROS node, subscribe to the LaserScan topic, and receive the data."""
        rospy.init_node('laser_listener', anonymous=True)
        rospy.loginfo("Laser Scan Listener node initialized.")
        rospy.Subscriber('/robot/front_laser/scan_filtered', LaserScan, self.laser_callback)

        # Block until data is received
        rospy.spin()

        if self.laser_data is None:
            raise RuntimeError("Failed to receive laser scan data.")
        return self.laser_data

    def get_next_json_filename(self):
        """Get the next available filename for a JSON file."""
        i = 0
        while True:
            filename = os.path.join(self.directory, "laser_data_{}.json".format(i))
            if not os.path.exists(filename):
                return filename
            i += 1

    def write_data_to_json(self, laser_data, angle_data, json_filename):
        """Write the laser scan data and corresponding angle data to a JSON file."""
        # Create a list of dictionaries with laser range and its corresponding angle
        data_dict = [{"laser_range": laser_data[i], "angle": angle_data[i]} for i in range(len(laser_data))]

        directory = os.path.dirname(json_filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write to JSON file
        with open(json_filename, 'w') as file:
            json.dump(data_dict, file, indent=4)

    def angle_measurement(self, laser_data):
        """Calculate the angle of the laser scan."""
        # Assuming laser_data is a list of ranges, calculate the angles based on the number of ranges
        angles = []
        num_ranges = len(laser_data)
        for i in range(num_ranges):
            angle = i * self.angle_increment
            angles.append(angle)
        return angles


if __name__ == '__main__':
    directory = 'Lidar_data'
    laser_listener = LaserScanListener(directory)  # Create listener object
    try:
        laser_data = laser_listener.receive_laser_scan()  # Get the laser scan data
        print("Laser data received:", laser_data)

        angle_data = laser_listener.angle_measurement(laser_data)  # Calculate the angle of the laser scan
        print("Angle data:", angle_data)

        json_filename = laser_listener.get_next_json_filename()  # Get the next JSON filename
        laser_listener.write_data_to_json(laser_data, angle_data,
                                          json_filename)  # Write laser data and angles to JSON file

        print("Laser data has been written to '{}'".format(json_filename))
    except RuntimeError as e:
        print(e)