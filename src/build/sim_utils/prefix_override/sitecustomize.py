import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/me597/ros2/sim_ws_Fall2025/src/install/sim_utils'
