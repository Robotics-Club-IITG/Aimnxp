# AIM NXP Warehouse Challenge Solution (2025)

This repository contains the complete `cranium` workspace for my solution to the **AIM NXP 2025 "Warehouse Treasure Hunt & Object Recognition"** challenge.

This project, based on the NXP B3RB robot, involves autonomous navigation, object recognition, and strategic pathfinding in a simulated warehouse environment using ROS 2 Humble and Gazebo.



---

### ## My Contributions

This repository contains the full workspace with all modifications implemented to solve the challenge. While the base framework was provided, my work focused on developing the core logic and tuning the system for performance.

My key modifications are primarily within the `b3rb_ros_aim_india` package and the `b3rb_nav2` configurations:

* **Central State Machine:** Implemented the main logic in `b3rb_ros_warehouse.py` to manage the robot's state (e.g., `EXPLORING`, `NAVIGATING_TO_SHELF`, `DECODING_QR`, `RECOGNIZING_OBJECTS`).
* **Shelf Detection:** Developed a map-based algorithm to parse the `/map` occupancy grid, identify shelf footprints, and calculate their precise coordinates for navigation.
* **QR Code Decoding:** Implemented logic within the `camera_image_callback` to process the camera feed using OpenCV and `pyzbar` to reliably detect and decode QR codes on the shelves.
* **Heuristic Navigation:** Wrote the logic to parse the heuristic angle from the decoded QR string and calculate the goal pose for the next (hidden) shelf.
* **Nav2 Tuning:** Modified the configuration files in `b3rb/b3rb_nav2/config/` (like `nav2.yaml` and `slam.yaml`) to optimize the robot's navigation, path planning, and recovery behaviors for the specific warehouse layout.
* **Object Recognition Logic:** Integrated the provided YOLO model's output from the `/shelf_objects` topic, associating recognized objects with the correct shelf and publishing the final data.

---

### ## Original Framework & Installation

This solution is built on top of the **CogniPilot (AIRY Release)** framework provided by NXP.

**1. Install the Original Stack:**
Before using this repository, you must first install the complete CogniPilot framework by following the setup instructions from the original competition repository.
* **Original Repo:** `https://github.com/NXPHoverGames/NXP_AIM_INDIA_2025`

**2. Run My Solution:**
After installing the base framework, you can use my solution by:
* Cloning this repository to replace the `cranium` workspace.
* Building the workspace:
    ```bash
    cd ~/cognipilot/cranium/
    colcon build
    source install/setup.bash
    ```
* Launching the simulation:
    ```bash
    ros2 launch b3rb_gz_bringup sil.launch.py world:=nxp_aim_india_2025/warehouse_1
    ```
