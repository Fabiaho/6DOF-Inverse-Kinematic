#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64, String
import pandas as pd
import signal
import sys



'''
- TODO: Die richtigen Topics abbnonieren gerade nur test topics


'''


class DataCollector:
    def __init__(self):
        # Hier werden die Listen initialisiert, in denen die Daten gespeichert werden
        self.times = []
        self.float_data = []
        self.string_data = []

        # Subscriber für zwei Beispiel-Topics
        self.sub_float = rospy.Subscriber("/example_float_topic", Float64, self.float_callback)
        self.sub_string = rospy.Subscriber("/example_string_topic", String, self.string_callback)
        
        # Beim Beenden des Nodes (z.B. mit Ctrl+C) soll die save_data Funktion aufgerufen werden
        signal.signal(signal.SIGINT, self.shutdown_hook)
        signal.signal(signal.SIGTERM, self.shutdown_hook)

    def float_callback(self, msg):
        # Aktuelle ROS-Zeit und empfangene Nachricht speichern
        self.times.append(rospy.get_time())
        self.float_data.append(msg.data)
        # Für das Beispiel werden hier Strings einfach mit leeren Einträgen ergänzt,
        # damit die Listen gleich lang bleiben.
        self.string_data.append("")

    def string_callback(self, msg):
        # Falls man mit variabler Länge umgehen möchte, könnte man z.B. die Zeit nicht
        # separat führen oder zwei DataFrames erstellen. Hier aber einheitlich:
        self.times.append(rospy.get_time())
        self.string_data.append(msg.data)
        # Leeren Eintrag für Float-Daten:
        self.float_data.append(float('nan'))

    def save_data(self):
        # Erstellung eines DataFrames aus den gesammelten Daten
        df = pd.DataFrame({
            'time': self.times,
            'float_value': self.float_data,
            'string_value': self.string_data
        })

        # Speichern als CSV
        df.to_csv('/home/fhtw_user/catkin_ws/src/fhtw/data_generator/scripts/ros_collected_data.csv', index=False)
        rospy.loginfo("Daten wurden als CSV gespeichert.")

    def shutdown_hook(self, signum, frame):
        rospy.loginfo("Shutdown signal empfangen, Daten werden gespeichert.")
        self.save_data()
        # Beenden von ROS sauber durchführen
        rospy.signal_shutdown("Beende den Node.")

if __name__ == '__main__':
    rospy.init_node('data_collector_node', anonymous=True)
    collector = DataCollector()
    rospy.spin()


'''
For testign this generator
rostopic pub /example_float_topic std_msgs/Float64 "data: 1.23"
rostopic pub /example_string_topic std_msgs/String "data: 'Hallo ROS'"

- After str+C this should generate a CSV file
'''