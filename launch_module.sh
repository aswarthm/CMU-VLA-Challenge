#!/bin/bash

cd ai_module
cd genai-server && python3.9 app.py &
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics > /dev/null 2>&1 &
colcon build && source install/local_setup.bash && ros2 launch dummy_vlm dummyvlm.launch.py 

