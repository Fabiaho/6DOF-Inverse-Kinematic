# Tips

- Logs are under hidden folder: /home/fhtw_user/.ros
- Open multiple terminals inside of the docker:

```bash
docker exec -it fhtw_ros bash
source /opt/ros/noetic/setup.bash
cd /home/fhtw_user/catkin_ws
source devel/setup.bash #(nur im catkin_ws möglich)
#(In jedem neuen Terminal muss source devel/setup.bash wieder gesetzt werden um die Ros befehle durchzuführen)
```
