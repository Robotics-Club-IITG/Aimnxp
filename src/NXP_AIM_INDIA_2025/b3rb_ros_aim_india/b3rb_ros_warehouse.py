# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import SetParameters

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist # For direct command velocity

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

from pyzbar.pyzbar import decode

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)
		
		self.row_count = 2
		self.col_count = shelf_count
		

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
					      relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		self.boxes[row][col].delete(1.0, tk.END)
		self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')
		# --- Shelf Detection State Variables (Ensure these are in your __init__) ---
		self.map_fully_explored = False # New state variable
		self.shelves_detected_on_map = [] # To store detected shelf world coordinates
		self.current_shelf = []
		self.current_shelf_idx = 0 # To keep track of which shelf to visit next
		self.task_status=0
		self.first_shelf_found=0
		self.x_shelf=0
		self.y_shelf=0
		self.obstacled_avoided= False
		self.corner_wall_status=0
		self.current_height=0
		self.current_width=0
		self.corner_wall_navigation=False
		self.wrong_shelf=False
		self.last_x=0.0
		self.last_y=0.0
		self.visited_shelves = [] 
		self.temp_visited_shelves = []

		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')
		# --- MODIFIED: Nav2 Parameter Clients (only controller_server) ---
		self.controller_client = self.create_client(SetParameters, '/controller_server/set_parameters')

		# Wait for the service to be available (important for startup robustness)
		self.get_logger().info('Waiting for /controller_server/set_parameters service...')
		while not self.controller_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().warn('controller_server/set_parameters service not available, waiting again...')
		self.get_logger().info('controller_server/set_parameters service available.')

		# --- MODIFIED: Define Tolerance Sets for Different Phases (only controller_server params) ---
		self.tolerance_configs = {
			"exploration": {
				# Controller Server Parameters (general_goal_checker and progress_checker from your YAML)
				"general_goal_checker.xy_goal_tolerance": 0.4, # More relaxed than default 0.25
				"general_goal_checker.yaw_goal_tolerance": 0.8, # More relaxed than default 0.1 (radians, ~45 degrees)
				"progress_checker.required_movement_radius": 0.5, # Default was 0.5. Keep it for exploration to prevent false stucks.
				"progress_checker.movement_time_allowance": 15.0, # Default was 10.0. Give more time for progress during exploration.
				"FollowPath.desired_linear_vel": 0.8, # Existing value or initial exploration speed

			},
			"precision": {
				# Controller Server Parameters
				"general_goal_checker.xy_goal_tolerance": 0.1, # Very tight for final approach (default was 0.25)
				"general_goal_checker.yaw_goal_tolerance": 0.1, # Tight for orientation (default was 0.1)
				"progress_checker.required_movement_radius": 0.05, # Much tighter for precision (default was 0.5)
				"progress_checker.movement_time_allowance": 5.0, # Less time, fail faster if truly stuck (default was 10.0)
				"FollowPath.desired_linear_vel": 0.8, # Existing value or initial exploration speed
			}
		}
		self.current_tolerance_phase = None


		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		# Add subscription for local costmap
		self.subscription_local_map = self.create_subscription(
			OccupancyGrid,
			'/local_costmap/costmap', # Usually this is the topic for local costmap
			self.local_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)
		
		# Publisher for direct cmd_vel (for manual obstacle avoidance if needed)
		self.publisher_cmd_vel = self.create_publisher(
			Twist,
			'/cmd_vel',
			QOS_PROFILE_DEFAULT
		)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = \
			self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = \
			self.get_parameter('initial_angle').get_parameter_value().double_value

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None
		self.local_map_curr = None # New variable for local map

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10
		self.current_navigation_goal = None # Store the actual goal for re-sending
		self.avoiding_obstacle = False # New state variable for obstacle avoidance

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0

		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		self.next_angle_deg = 0.0
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0
			self.qr_col_count = 0

		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()
		self.last_published_qr = None
		self.last_published_qr_index = 0  # Start at 0, so QR 1 is first allowed
		# Track max sum of objects for each shelf (column)
		self.max_object_sum_per_col = [0] * self.shelf_count
		# Store latest max objects and QR for each shelf
		self.latest_object_name_per_col = [[] for _ in range(self.shelf_count)]
		self.latest_object_count_per_col = [[] for _ in range(self.shelf_count)]
		self.latest_qr_per_col = [None] * self.shelf_count

		self.qr_gui_published_for_current_shelf = False

	def set_nav2_tolerances(self, phase_name):
		"""
		Sets Nav2 parameters for a given phase (e.g., "exploration" or "precision").
		Focuses only on controller_server parameters.
		"""
		if phase_name == self.current_tolerance_phase:
			self.get_logger().debug(f"Tolerances already set for '{phase_name}'. Skipping redundant call.")
			return

		if phase_name not in self.tolerance_configs:
			self.get_logger().error(f"Unknown tolerance phase: {phase_name}. Cannot set parameters.")
			return

		params_to_set = self.tolerance_configs[phase_name]
		self.get_logger().info(f"Initiating change of Nav2 parameters for controller_server to phase: {phase_name}")

		controller_params_msgs = []

		for param_key, param_value in params_to_set.items():
			# These keys (e.g., "general_goal_checker.xy_goal_tolerance") are directly
			# the parameter names for the controller_server's set_parameters service.
			controller_params_msgs.append(Parameter(name=param_key, value=param_value).to_parameter_msg())
			
		# Send request asynchronously and attach a callback for the result
		if controller_params_msgs:
			req_controller = SetParameters.Request()
			req_controller.parameters = controller_params_msgs
			self.controller_client.call_async(req_controller).add_done_callback(
				lambda future, node_name="controller_server": self._set_parameters_callback(future, node_name)
			)
		else:
			self.get_logger().warn("No controller parameters to set for this phase.")
		
		self.current_tolerance_phase = phase_name

	def _set_parameters_callback(self, future, node_name):
		"""Callback for parameter setting service (general for any node)."""
		try:
			response = future.result()
			if response is None:
				self.get_logger().error(f"Service call to {node_name}/set_parameters failed: No response.")
				return

			all_successful = True
			for result in response.results:
				if not result.successful:
					all_successful = False
					self.get_logger().error(f"Failed to set parameter '{result.name}' on {node_name}: {result.reason}")
			
			if all_successful:
				self.get_logger().info(f"Successfully updated all specified parameters on {node_name}.")
			else:
				self.get_logger().warn(f"Some parameters failed to set on {node_name}.")

		except Exception as e:
			self.get_logger().error(f"Exception while receiving parameter set response from {node_name}: {e}")

	# --- END NEW METHODS ---

	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)

	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates.

		Args:
			message: ROS2 message containing the simple map data.

		Returns:
			None
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		self.world_center = self.get_world_coord_from_map_coord(
			map_info.width / 2, map_info.height / 2, map_info
		)

	def global_map_callback(self, message):
		"""Callback function to handle global map updates.

		Args:
			message: ROS2 message containing the global map data.

		Returns:
			None
		"""
		
		self.global_map_curr = message
		

	def local_map_callback(self, message):
		"""Callback function to handle local costmap updates.
		Used for local obstacle detection.

		Args:
			message: ROS2 message containing the local costmap data.

		Returns:
			None
		"""
		self.local_map_curr = message
	def free_space(self,x,y,global_map_arr,height,width):
		counter=0
		y0=y
		for x in range(x-15,x+15):
			y=y0
			for y in range(y-15,y+15):
				if y>height-2 or x>width-2 or x<0 or y<0:
					
					self.logger.info(f'out of size {x} , {y}')
					return 2
				if global_map_arr[y,x]>1:
					counter+=1
		if counter>5:

			return 0
		else:
			self.logger.info('free space found')
			return 1

		


	def avoid_local_obstacle(self):
		"""
		Analyzes the local costmap to detect immediate obstacles.
		Returns True if an obstacle is detected within a close proximity, False otherwise.
		
		"""

		if self.global_map_curr is None or self.goal_completed==False:
			return False

		map_info = self.global_map_curr.info
		width = map_info.width
		height = map_info.height
		resolution = map_info.resolution
		open_counter=0
		open_counter_2=0
		quadrant=0
		

		# Convert 1D map data to 2D numpy array
		global_map_array = np.array(self.global_map_curr.data).reshape((height, width))

		# Define a region of interest (ROI) directly in front of the robot
		# Assuming robot is at the center of the local costmap.
		# Nav2's local costmap often has the robot at the center (width/2, height/2).
		robot_map_x,robot_map_y = self.get_map_coord_from_world_coord(self.buggy_pose_x,self.buggy_pose_y,map_info)
		x_shelf_map,y_shelf_map=self.get_map_coord_from_world_coord(self.x_shelf,self.y_shelf,map_info)
		x_shelf_map=int(x_shelf_map)
		y_shelf_map=int(y_shelf_map)
		robot_map_x=int(robot_map_x)
		robot_map_y=int(robot_map_y)

		if robot_map_x>width/2 and robot_map_y>height/2:
			quadrant=1
		elif robot_map_x<width/2 and robot_map_y>height/2:
			quadrant=2
		elif robot_map_x<width/2 and robot_map_y<height/2:
			quadrant=3
		elif robot_map_x>width/2 and robot_map_y<height/2:
			quadrant=4
		if self.obstacled_avoided==True:

			x0 = robot_map_x
			y0 = robot_map_y
			self.logger.info(f'{x0},{y0}')

			# Determine coefficients A, B, C for the line equation Ax + By + C = 0
			# based on the angle and a point on the line.

			A: float
			B: float
			C: float

			# Check if the line is vertical (angle is close to pi/2 or 3*pi/2, etc.)
			# Use a small epsilon to account for floating-point inaccuracies
			epsilon = 1e-9
			if abs(math.cos(np.deg2rad(self.next_angle_deg))) < epsilon:
				# Case: Vertical line (x = x_shelf)
				# Equation: 1*x + 0*y - x_shelf = 0
				A = 1.0
				B = 0.0
				C = -x_shelf_map
			else:
				# Case: Non-vertical line
				# Calculate slope from angle: m = tan(angle)
				slope = math.tan(np.deg2rad(self.next_angle_deg))
				# Equation: y - y_shelf = slope * (x - x_shelf)
				# Rearrange to slope*x - y - slope*x_shelf + y_shelf = 0
				A = slope
				B = -1.0
				C = - (slope * x_shelf_map) + y_shelf_map

			# Calculate the denominator (A^2 + B^2)
			# This will be non-zero as A and B cannot both be zero for a valid line.
			denominator = A**2 + B**2
			self.logger.info(f'{A},{B},{C}')

			# Apply the formulas for the intersection point (x_Q, y_Q)
			x_Q = (B**2 * x0 - A * B * y0 - A * C) / denominator
			y_Q = (A**2 * y0 - A * B * x0 - B * C) / denominator
			x_Q=int(x_Q)
			y_Q=int(y_Q)
			x_world,y_world=self.get_world_coord_from_map_coord(x_Q,y_Q,map_info)
			self.logger.info(f'{x_world},{y_world},{x_Q},{y_Q}')
			if self.free_space(x_Q,y_Q,global_map_array,height,width)==1:
				self.obstacled_avoided=False
				goal=self.create_goal_from_map_coord(x_Q,y_Q,map_info,np.deg2rad(self.next_angle_deg))
				self.send_goal_from_world_pose(goal)
				self.logger.info("recovery succesfull")
				return
			else:
				self.logger.info('recovery failed, trying normal obsctacle avoidance')

				check_distance_meters = 2
				map_info = self.global_map_curr.info
				width = map_info.width
				height = map_info.height

				# Convert meters to map cells
				check_dist_cells = int(check_distance_meters / resolution)
				if self.next_angle_deg>180:
					check_dist_cells=-check_dist_cells
				self.logger.info(f'{check_dist_cells}')
				
				obstacle_detected = False
				next_angle_deg=0
				grey_counter=0
				x=x_shelf_map
				y=y_shelf_map
				if abs(math.cos(np.deg2rad(self.next_angle_deg))) < epsilon:
					while grey_counter<10:
						
						if y>height-3 or x>width-3 or y<5 or x<5 :

							break
						if global_map_array[y,x]==-1:
							x_map,y_map=self.get_world_coord_from_map_coord(x,y,map_info)
							self.logger.info(f'{x_map},{y_map}')
							grey_counter+=1
						else:
							grey_counter=0
						if self.next_angle_deg>0 and self.next_angle_deg<180:
							y+=1
						else:
							y-=1
						while global_map_array[y,x]!=100:

							if self.next_angle_deg > 0 and self.next_angle_deg < 180:
								y += 1
							else:
								y -= 1
							if y >height-2 or y < 3 or x > width-2 or x < 3:

								break

						next_angle_deg=np.rad2deg(np.arctan2(y-robot_map_y,x-robot_map_x))
		
					
				else:
					slope=math.tan(np.deg2rad(self.next_angle_deg))
					y=int(slope*(x-x_shelf_map) + y_shelf_map)
					while grey_counter<10:
						
						if y>height-3 or x>width-3 or y<5 or x<5 :
		
							break
						if global_map_array[y,x]==-1:
							x_map,y_map=self.get_world_coord_from_map_coord(x,y,map_info)
	
							grey_counter+=1
						else:
							grey_counter=0
						if self.next_angle_deg>90 and self.next_angle_deg<270:
							x-=1
						else:
							x+=1
						while global_map_array[y,x]!=100:
							y = int(slope * (x - x_shelf_map) + y_shelf_map)

							if self.next_angle_deg > 90 and self.next_angle_deg < 270:
								x -= 1
							else:
								x += 1
							if y >height-2 or y < 3 or x > width-2 or x < 3:

								break

						next_angle_deg=np.rad2deg(np.arctan2(y-robot_map_y,x-robot_map_x))
						self.logger.info(f'{x},{robot_map_x},{y},{robot_map_y},{next_angle_deg}')
				
					

				for y in range(min(robot_map_y,robot_map_y+check_dist_cells),max(robot_map_y,robot_map_y+check_dist_cells)):
					x=int((y-robot_map_y)/math.tan(np.deg2rad(next_angle_deg))+robot_map_x)
					if y>height-2 or x>width-2:
						continue
					if global_map_array[y, x] > 80 :
						self.logger.info(f"obstacle detected line angle:")
						obstacle_detected=True
						x_world,y_world=self.get_world_coord_from_map_coord(x,y,map_info)

						break
				if obstacle_detected==True and self.free_space(x,y,global_map_array,height,width)==0:
					xobstacle=x
					yobstacle=y
					xdown=x
					xup_outofbound=False
					xdown_outofbound=False
					y_down=y

					epsilon = 1e-6 
					
					while open_counter<20 and open_counter_2<20:
						if xup_outofbound and xdown_outofbound:
							break
						x+=1
						xdown-=1
						
						angle_rad = np.deg2rad(self.next_angle_deg)
						if abs(np.cos(angle_rad)) < epsilon:
							y = yobstacle + (x - xobstacle) * 0 
							y_down = yobstacle + (xdown - xobstacle) * 0
						else:
							slope = math.tan(angle_rad)
							y = int(-1 / slope * (x - xobstacle) + yobstacle)
							y_down = int(-1 / slope * (xdown - xobstacle) + yobstacle)

						if x>width-2 or y>height-2 or y<0:
							self.logger.info('xup or y out of bounds')
							xup_outofbound=True
						else:
							if global_map_array[y,x]<=0:
								open_counter+=1
								
							else:
								open_counter=0
						if  xdown<0 or y_down>height-2 or y_down<0:
							self.logger.info('xdown or y out of bounds')
							xdown_outofbound=True
						else:
							if global_map_array[y_down,xdown]<=0:
								open_counter_2+=1
							else:
								open_counter_2 = 0
					if xup_outofbound==True and xdown_outofbound==True:
						self.logger.info('no open space found')
						return False
					else:
						if open_counter_2>open_counter:
							x=xdown
							y=y_down
						x,y=self.get_world_coord_from_map_coord(x,y,map_info)
						self.logger.info(f'{x},{y}')
						goal=self.create_goal_from_world_coord(x,y,self.next_angle_deg)
						self.send_goal_from_world_pose(goal)
						self.obstacled_avoided=True

					return True
				else:		
					self.logger.info(f"no Obstacle detected in local costmap at map coord and initiating blind movement ")
					current_x = self.buggy_pose_x
					current_y = self.buggy_pose_y
							
					# Use the initial_angle for the direction of blind movement
					# Convert initial_angle from degrees to radians for goal creation
					blind_move_yaw_radians = np.deg2rad(next_angle_deg)
							
					# Calculate a point 5 meters directly ahead in the direction of blind_move_yaw_radians
					move_distance = 1.5 # meters
					target_x = current_x + move_distance * math.cos(blind_move_yaw_radians)
					target_y = current_y + move_distance * math.sin(blind_move_yaw_radians)
							
					# Create and send a new goal
					blind_move_goal = self.create_goal_from_world_coord(target_x, target_y, blind_move_yaw_radians)
					self.send_goal_from_world_pose(blind_move_goal)
					self.current_navigation_goal = blind_move_goal # Store this goal
					self.set_nav2_tolerances("exploration") # Keep exploration tolerances for blind movement
					self.task_status=1
					return False
				
		else:			
		
			check_distance_meters = 2
			map_info = self.global_map_curr.info
			width = map_info.width
			height = map_info.height

			# Convert meters to map cells
			check_dist_cells = int(check_distance_meters / resolution)
			if self.next_angle_deg>180:
				check_dist_cells=-check_dist_cells
			self.logger.info(f'{check_dist_cells}')
			
			obstacle_detected = False

			for y in range(min(robot_map_y,robot_map_y+check_dist_cells),max(robot_map_y,robot_map_y+check_dist_cells)):
				x=int((y-robot_map_y)/math.tan(np.deg2rad(self.next_angle_deg))+robot_map_x)
				if y>height-2 or x>width-2 or y<0 or x<0:
					continue
				if global_map_array[y, x] > 80 :
					self.logger.info(f"obstacle detected line angle:")
					obstacle_detected=True
					x_world,y_world=self.get_world_coord_from_map_coord(x,y,map_info)

					break
			free_counter_y=0

			if self.free_space(x,y,global_map_array,height,width)==2 or self.corner_wall_navigation==True	:
				self.logger.info(f'corner wall detected in quadrant {quadrant}')
				
				
				if self.corner_wall_status==0:
					if self.corner_wall_navigation==False:
						self.initial_quadrant=quadrant
					if quadrant==1 and abs(self.current_width-width)>20 and self.corner_wall_navigation==True:
						if abs(self.current_height-height)>5 and self.corner_wall_navigation==True:
							self.corner_wall_status+=1
						
						self.corner_wall_navigation=False
						quadrant=4
					if quadrant==2 and abs(self.current_height-height)>20 and self.corner_wall_navigation==True:
						if abs(self.current_width-width)>5 and self.corner_wall_navigation==True:
							self.corner_wall_status+=1
							
						self.corner_wall_navigation=False
						quadrant=3
					if quadrant==3 and abs(self.current_width-width)>20 and self.corner_wall_navigation==True:
						if abs(self.current_height-height)>5 and self.corner_wall_navigation==True:
							self.corner_wall_status+=1
							
						self.corner_wall_navigation=False
						quadrant=2
					if quadrant==4 and abs(self.current_height-height)>20 and self.corner_wall_navigation==True:
						if abs(self.current_width-width)>5 and self.corner_wall_navigation==True:
							self.corner_wall_status+=1
							
						self.corner_wall_navigation=False
						quadrant=1


					if quadrant ==1:
						if abs(self.current_height-height)>5 and self.corner_wall_navigation==True:
							self.corner_wall_status+=1
							return
						self.corner_wall_navigation=True
						self.current_height=height
						self.current_width=width		
						x=width-1
						y=0
						while self.free_space(x,y,global_map_array,height,width)!=1:
							x=width-1-y
							y+=1

						goal=self.create_goal_from_map_coord(x,y,map_info)
						self.send_goal_from_world_pose(goal)
					if quadrant ==2:
						if abs(self.current_width-width)>5 and self.corner_wall_navigation==True:
							self.corner_wall_status+=1
							return
						self.corner_wall_navigation=True
						self.current_height=height
						self.current_width=width		
						x=width-1
						y=height-1
						while self.free_space(x,y,global_map_array,height,width)!=1:
							x=width-height+y
							y-=1

						goal=self.create_goal_from_map_coord(x,y,map_info)
						self.send_goal_from_world_pose(goal)
					if quadrant ==3:
						if abs(self.current_height-height)>5 and self.corner_wall_navigation==True:
							self.corner_wall_status+=1
							return
						self.corner_wall_navigation=True
						self.current_height=height
						self.current_width=width		
						x=0
						y=height-1
						while self.free_space(x,y,global_map_array,height,width)!=1:
							x=height-1-y
							y-=1

						goal=self.create_goal_from_map_coord(x,y,map_info)
						self.send_goal_from_world_pose(goal)
					if quadrant ==4:
					
						if abs(self.current_width-width)>5 and self.corner_wall_navigation==True:
							self.corner_wall_status+=1
							return
						self.corner_wall_navigation=True
						self.current_height=height
						self.current_width=width		
						x=0
						y=0
						while self.free_space(x,y,global_map_array,height,width)!=1:
							x=y
							y+=1

						goal=self.create_goal_from_map_coord(x,y,map_info)
						self.send_goal_from_world_pose(goal)
				elif self.corner_wall_status==1:
	
					if quadrant==self.initial_quadrant :
						self.corner_wall_navigation==False
						self.obstacled_avoided=True
						return
					else:
						if quadrant==1:
							self.corner_wall_navigation=True
							self.current_height=height
							self.current_width=width		
							x=0
							y=height-1
							while self.free_space(x,y,global_map_array,height,width)!=1:
								
								if (x-robot_map_x)**2 + (y-robot_map_y)**2 <=70/(resolution**2):
									x=height-1-y
								else:
									x=int(height-2-y+((width-1)/2))
								y-=1

							goal=self.create_goal_from_map_coord(x,y,map_info)
							self.send_goal_from_world_pose(goal)

						if quadrant ==3:

							self.corner_wall_navigation=True
							self.current_height=height
							self.current_width=width		
							x=width-1
							y=0
							while self.free_space(x,y,global_map_array,height,width)!=1:
								
								if (x-robot_map_x)**2 + (y-robot_map_y)**2 <=70/(resolution**2):
									x=width-1-y
								else:
									x=int(-y-1+((width-1)/2))
								
								y+=1

							goal=self.create_goal_from_map_coord(x,y,map_info)
							self.send_goal_from_world_pose(goal)
						if quadrant ==4:

							self.corner_wall_navigation=True
							self.current_height=height
							self.current_width=width		
							x=width-1
							y=height-1

							while self.free_space(x,y,global_map_array,height,width)!=1:
								
								if (x-robot_map_x)**2 + (y-robot_map_y)**2 <=70/(resolution**2):
									y=-width+height+x
								else:
									y=int(x-width+((height-1)/2))
								
								x-=1
							
							

							xmap,ymap=self.get_world_coord_from_map_coord(x,y,map_info)

							goal=self.create_goal_from_map_coord(x,y,map_info)
							self.send_goal_from_world_pose(goal)

						if quadrant ==2:
							self.logger.info(f'{abs(self.current_width-width)}')

							self.corner_wall_navigation=True
							self.current_height=height
							self.current_width=width		
							x=0
							y=0
							while self.free_space(x,y,global_map_array,height,width)!=1:
								
								if (x-robot_map_x)**2 + (y-robot_map_y)**2 <=70/(resolution**2):
									x=y
								else:
									y=int(x-1+((height-1)/2))
								x+=1

							goal=self.create_goal_from_map_coord(x,y,map_info)
							self.send_goal_from_world_pose(goal)
							
			



						
					

				

			elif obstacle_detected==True and self.free_space(x,y,global_map_array,height,width)==0:
				xobstacle=x
				yobstacle=y
				xdown=x
				xup_outofbound=False
				xdown_outofbound=False
				y_down=y
				
				while open_counter<20 and open_counter_2<20:
					if xup_outofbound==True and xdown_outofbound==True:
						break
					x+=1
					xdown-=1
					
					y=int(-1/(math.tan(np.deg2rad(self.next_angle_deg)))*(x-xobstacle)+yobstacle)
					if x>width-2 or y>height-2 or y<0:
						self.logger.info('xup or y out of bounds')
						xup_outofbound=True
					else:
						if global_map_array[y,x]<=0:
							open_counter+=1
							
						else:
							open_counter=0
					y_down=int(-1/(math.tan(np.deg2rad(self.next_angle_deg)))*(xdown-xobstacle)+yobstacle)
					if  xdown<0 or y_down>height-2 or y_down<0:
						self.logger.info('xdown or y out of bounds')
						xdown_outofbound=True
					else:
						if global_map_array[y_down,xdown]<=0:
							open_counter_2+=1
						else:
							open_counter_2 = 0
				if xup_outofbound==True and xdown_outofbound==True:
					self.logger.info('no open space found')
					return False
				else:
					if open_counter_2>open_counter:
						x=xdown
						y=y_down
					x,y=self.get_world_coord_from_map_coord(x,y,map_info)
					self.logger.info(f'{x},{y}')
					goal=self.create_goal_from_world_coord(x,y,self.next_angle_deg)
					self.send_goal_from_world_pose(goal)
					self.obstacled_avoided=True

				return True
			else:		
				self.logger.info(f"no Obstacle detected in local costmap at map coord and initiating blind movement ")
				current_x = self.buggy_pose_x
				current_y = self.buggy_pose_y
						
				# Use the initial_angle for the direction of blind movement
				# Convert initial_angle from degrees to radians for goal creation
				blind_move_yaw_radians = np.deg2rad(self.next_angle_deg)
						
				# Calculate a point 5 meters directly ahead in the direction of blind_move_yaw_radians
				move_distance = 1.5 # meters
				target_x = current_x + move_distance * math.cos(blind_move_yaw_radians)
				target_y = current_y + move_distance * math.sin(blind_move_yaw_radians)
						
				# Create and send a new goal
				blind_move_goal = self.create_goal_from_world_coord(target_x, target_y, blind_move_yaw_radians)
				self.send_goal_from_world_pose(blind_move_goal)
				self.current_navigation_goal = blind_move_goal # Store this goal
				self.set_nav2_tolerances("exploration") # Keep exploration tolerances for blind movement
				self.task_status=1
				return False


	def send_task(self):
		"""
		Manages the sequential tasks for the robot:
		1. Initial pose setting (task_status 0)
		2. Shelf localization (task_status 1) - MODIFIED TO INCLUDE BLIND MOVEMENT
		3. Navigation to the first identified shelf (task_status 2)
		4. Navigation to the next side of the first shelf (task_status 3)
		5. Stop after first shelf and its side (task_status 4)
		
		Includes logic for obstacle avoidance during navigation.
		"""
		if not self.goal_completed or self.avoiding_obstacle:
			# If a goal is already active and we're not avoiding an obstacle, do nothing.
			return


		if self.task_status == 0:
			self.logger.info("Sending initial pose goal (Task 0).")
			# Convert initial_angle from degrees to radians for goal creation
			target_yaw_radians = np.deg2rad(self.initial_angle)
			self.set_nav2_tolerances("exploration") 
			goal = self.create_goal_from_world_coord(0.0, 0.0, target_yaw_radians) # Use 0,0 and initial angle as starting point
			self.send_goal_from_world_pose(goal)
			self.current_navigation_goal = goal # Store the goal
			# Set initial tolerances
			# Task status will increment to 1 upon goal completion.

		elif self.task_status == 1:
			self.logger.info("Locating shelves on map (Task 1).")
			self.locate_shelves_on_map() # This updates self.current_shelf and increments task_status if successful

			if len(self.current_shelf) == 0:
				# No shelves detected, so initiate blind movement at initial angle
				
				self.logger.info("No shelves detected. Initiating blind movement at initial angle.")
				self.avoid_local_obstacle()

			else:
				# Check if the current shelf has already been visited
				# A small tolerance for floating-point comparisons
				# Ensure that current_shelf[0][0] and current_shelf[0][1] are the x and y coordinates of the shelf center
				shelf_center_x = self.current_shelf[0][0]
				shelf_center_y = self.current_shelf[0][1]
				is_visited = False
				for visited_x, visited_y in self.visited_shelves:
					if abs(shelf_center_x - visited_x) < 0.1 and abs(shelf_center_y - visited_y) < 0.1: # 0.1m tolerance
						self.logger.info(f"Shelf at ({shelf_center_x:.2f}, {shelf_center_y:.2f}) has already been visited. Skipping.")
						is_visited = True
						# If already visited, clear current_shelf and try to locate new shelves
						self.current_shelf = [] 
						break
				
				if is_visited:
					# Stay in task 1 to re-evaluate and find a new unvisited shelf
					return
				
				self.logger.info("Shelves detected! Proceeding to Task 2 (navigation to first shelf).")
				# If shelves were detected and not visited, locate_shelves_on_map would have incremented task_status to 2.
				# So, this 'else' block will only be hit if shelf detection succeeded and task_status is now 2.
				# No need to send goal here, task_status 2 block will handle it in the next cycle.

		elif self.task_status == 2:
			if len(self.current_shelf) == 0:				
				
				self.logger.info("First shelf not found yet, re-attempting shelf localization or waiting (Task 2).")
				# This should ideally not happen if task_status became 2 because a shelf was found.
				# But as a safeguard: if current_shelf is somehow empty here, revert to task 1.
				self.task_status = 1
				return
			else:
				# Calculate approach point for the first shelf
				self.first_shelf_found=1
				shelf_data = self.current_shelf[0]
				# Approach point 2.5 meters in front of the shelf's midpoint,
				# and orient 180 degrees from the shelf's orientation.
				self.logger.info(f"{self.current_shelf[0][0]-2.5*np.sin(self.current_shelf[0][3])},{self.current_shelf[0][1]+2.5*np.cos(self.current_shelf[0][3])}")
				target_x1 = self.current_shelf[0][0]-2.5*np.sin(self.current_shelf[0][3])
				target_y1 = self.current_shelf[0][1]+2.5*np.cos(self.current_shelf[0][3])
				target_yaw1 = float(self.current_shelf[0][2]-np.pi) # Target orientation to face the shelf
				target_x2 = self.current_shelf[0][0]+2.5*np.sin(self.current_shelf[0][3])
				target_y2 = self.current_shelf[0][1]-2.5*np.cos(self.current_shelf[0][3])
				target_yaw2 = float(self.current_shelf[0][2])
				current_x=self.buggy_pose_x
				current_y=self.buggy_pose_y
				if (target_x1-current_x)**2 + (target_y1-current_y)**2 <(target_x2-current_x)**2 + (target_y2-current_y)**2:
					pass
				else:
					target_x1=target_x2
					target_y1=target_y2
					target_yaw1=target_yaw2
				self.logger.info(f"Sending goal to first shelf approach (Task 2): ({target_x1:.2f}, {target_y1:.2f}) at yaw {np.rad2deg(target_yaw1):.2f} deg.")
				goal = self.create_goal_from_world_coord(target_x1, target_y1, target_yaw1)
				self.send_goal_from_world_pose(goal)
				self.current_navigation_goal = goal
				self.set_nav2_tolerances("precision") # Switch to precision tolerances
		elif self.task_status == 3:
			if len(self.current_shelf) == 0:
				self.logger.info("Shelf data missing for Task 3. Cannot plan next move.")
				self.task_status = 1 # Revert to task 1 if data is somehow lost
				return
			shelf_data = self.current_shelf[0]

			goal = self.create_goal_from_world_coord(self.current_shelf[0][0]+2.5*np.cos(self.current_shelf[0][3]),self.current_shelf[0][1]+2.5*np.sin(self.current_shelf[0][3]),float(self.current_shelf[0][2]+np.deg2rad(90)))
			self.send_goal_from_world_pose(goal)
			self.current_navigation_goal = goal
			self.set_nav2_tolerances("precision") # Maintain precision tolerances
		elif self.task_status == 4:
				if len(self.current_shelf) == 0:
					self.logger.info("Shelf data missing for Task 3. Cannot plan next move.")
					self.task_status = 1 # Revert to task 1 if data is somehow lost
					return
				self.task_status=1

		else:
			self.logger.info(f"Unhandled task_status: {self.task_status}. No goal sent.")

	def locate_shelves_on_map(self):

		if self.wrong_shelf==True:
			self.x_shelf=self.last_x
			self.y_shelf=self.last_y
		else:
			if len(self.current_shelf)==0 and self.first_shelf_found==0:
				self.x_shelf=0
				self.y_shelf=0
				self.next_angle_deg=self.initial_angle
			elif len(self.current_shelf)==0:
				pass
			else:
					self.x_shelf=self.current_shelf[0][0]
					self.y_shelf=self.current_shelf[0][1]
			
		self.shelves_detected_on_map = []  # Clear previous detections
		self.current_shelf = []            # To store information about the identified "first shelf"

		if self.global_map_curr is None:
			self.logger.warn("Cannot locate shelves: global map is not available.")
			# Do NOT set task_status to 1 here; it's handled by send_task if current_shelf is empty
			return

		map_info = self.global_map_curr.info
		height, width = map_info.height, map_info.width

		# Step 1: Reshape the 1D map data into a 2D NumPy array
		map_array = np.array(self.global_map_curr.data).reshape((height, width))
		self.logger.info(f"Processing map with dimensions: {width}x{height}")

		# Step 2: Create a Binary Image of Occupied Spaces
		binary_map = np.zeros_like(map_array, dtype=np.uint8)
		binary_map[map_array ==100] = 255
		# The map data in ROS is Y-up, but OpenCV processes images Y-down, so we flip it
		binary_map = np.flipud(binary_map)
		self.logger.info("Created binary map of occupied spaces (occupied=255, free/unknown=0).")
		
		# --- Tuning Parameters ---
		MIN_LINE_LENGTH = 15
		MAX_LINE_LENGTH = 30
		CROP_BORDER_PIXELS = 5
		HOUGH_THRESHOLD = 20
		MAX_LINE_GAP = 20
		MAX_PARALLEL_DIST = 35 # Max distance between lines to be in the same cluster
		MAX_PARALLEL_ANGLE_DIFF = 10 # Max angle difference in degrees

		# Step 3: Crop the map to remove edge noise
		height, width = binary_map.shape
		cropped_map = binary_map[CROP_BORDER_PIXELS:height - CROP_BORDER_PIXELS, CROP_BORDER_PIXELS:width - CROP_BORDER_PIXELS]

		# Publish debug visualization of the cropped map
		self.publish_debug_image(self.publisher_qr_decode, cropped_map)
		self.logger.info("Published cropped map for debugging.")

		# Step 4: Apply Probabilistic Hough Line Transform to the cropped map
		lines = cv2.HoughLinesP(cropped_map, 1, np.pi / 180, HOUGH_THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
		
		detected_lines = []
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]
				length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
				if MIN_LINE_LENGTH <= length <= MAX_LINE_LENGTH:
					detected_lines.append(line[0])

		# Step 5: Cluster nearby and parallel lines to represent each shelf as a single line
		line_clusters = []
		for line1 in detected_lines:
			line1_mid = ((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2)
			line1_angle = np.degrees(np.arctan2(line1[3] - line1[1], line1[2] - line1[0]))
			
			found_cluster = False
			for cluster in line_clusters:
				rep_line = cluster[0]
				rep_line_mid = ((rep_line[0] + rep_line[2]) / 2, (rep_line[1] + rep_line[3]) / 2)
				rep_line_angle = np.degrees(np.arctan2(rep_line[3] - rep_line[1], rep_line[2] - rep_line[0]))

				dist = np.sqrt((line1_mid[0] - rep_line_mid[0])**2 + (line1_mid[1] - rep_line_mid[1])**2)
				angle_diff = abs(line1_angle - rep_line_angle)
				if angle_diff > 180: angle_diff = 360 - angle_diff
				
				if dist < MAX_PARALLEL_DIST and angle_diff < MAX_PARALLEL_ANGLE_DIFF:
					cluster.append(line1)
					found_cluster = True
					break
			
			if not found_cluster:
				line_clusters.append([line1])

		# Step 6: Select a single representative line from each cluster and convert to world coordinates
		final_shelf_lines_in_map_coords = []
		if line_clusters:
			for cluster in line_clusters:
				if cluster:
					# Find the longest line in the cluster to represent it
					longest_line = max(cluster, key=lambda l: np.sqrt((l[2] - l[0])**2 + (l[3] - l[1])**2))
					final_shelf_lines_in_map_coords.append(longest_line)

		if not final_shelf_lines_in_map_coords:
			self.logger.info("No shelves detected based on line criteria.")
			# Do NOT set task_status to 1 here; it's handled by send_task if current_shelf is empty
		else:
			self.logger.info(f"Detected {len(final_shelf_lines_in_map_coords)} distinct shelf footprints.")
			
			# Prepare a colored map for visualization
			colored_map = cv2.cvtColor(cropped_map, cv2.COLOR_GRAY2BGR)

			for line in final_shelf_lines_in_map_coords:
				x1, y1, x2, y2 = line
				angle_shelf = np.arctan2(y2 - y1, x2 - x1)	
				angle_shelf=(np.pi)-angle_shelf
				if angle_shelf==np.pi:
					angle_shelf=0	
				# Calculate the midpoint of the line in cropped map coordinates
				cx_img, cy_img = (x1 + x2) / 2, (y1 + y2) / 2
				
				# Convert midpoint to full image coordinates by adding the crop border back
				cx_full_img = int(cx_img + CROP_BORDER_PIXELS)
				cy_full_img = int(cy_img + CROP_BORDER_PIXELS)
				cx_full_img = cx_full_img-0.25*np.sin(angle_shelf)
				cy_full_img = cy_full_img +0.25*np.cos(angle_shelf)
				
				# Convert image coordinates to world coordinates.
				# Remember binary_map was flipped, so we use height - 1 - cy_full_img
				world_x, world_y = self.get_world_coord_from_map_coord(cx_full_img, height - 1 - cy_full_img, map_info)

				# Calculate orientation
				
				angle_robot_target = angle_shelf + np.pi/2 # Angle of approach (90 deg to the shelf line)
				angle_rad = np.arctan2(world_y-self.y_shelf, world_x-self.x_shelf)
				angle_shelf_shelfn = np.degrees(angle_rad)
				if angle_shelf_shelfn < 0:
						angle_shelf_shelfn += 360
				# Store the detected shelf
				self.logger.info(f"Identified potential shelf at map (img) coordinates ({cx_full_img}, {cy_full_img}), world coordinates ({world_x:.2f}, {world_y:.2f}) at an angle ({angle_shelf_shelfn}) with shelf at {np.rad2deg(angle_shelf)} deg")
				
				# Logic to identify the "first shelf" based on initial_angle
				# This part is crucial for defining your sequence.
				# The current implementation checks if the shelf's angle from origin is close to initial_angle.
				shelf_added = False
				for i in range(1, 7): # Check within +/- 4 degrees
					if (abs(self.next_angle_deg - angle_shelf_shelfn) < i or abs(self.next_angle_deg - angle_shelf_shelfn) > 360 - i) and not (abs(world_x - self.x_shelf) < 0.5 and abs(world_y - self.y_shelf) < 0.5):
						
						# FIXED: Check both visited lists properly
						is_already_visited = False
						# Check visited_shelves
						for vx, vy in self.visited_shelves:
							if abs(world_x - vx) < 0.5 and abs(world_y - vy) < 0.5:  # Increased tolerance
								is_already_visited = True
								self.logger.info(f"Shelf at ({world_x:.2f}, {world_y:.2f}) found in visited_shelves. Skipping.")
								break
						
						# Check temp_visited_shelves if not already found in visited_shelves
						if not is_already_visited:
							for vx, vy in self.temp_visited_shelves:
								if abs(world_x - vx) < 0.5 and abs(world_y - vy) < 0.5:  # Increased tolerance
									is_already_visited = True
									self.logger.info(f"Shelf at ({world_x:.2f}, {world_y:.2f}) found in temp_visited_shelves. Skipping.")
									break
                
						if not is_already_visited:
							self.logger.info(f"New shelf found and assigned to current_shelf. Angle: {angle_shelf_shelfn}, Position: ({world_x:.2f}, {world_y:.2f})")
							self.current_shelf.append((world_x, world_y, angle_robot_target, angle_shelf, angle_shelf_shelfn))
							
							self.logger.info(f"Added shelf to temp_visited_shelves: ({world_x:.2f}, {world_y:.2f})")
							
							shelf_added = True
							break
				
				if not shelf_added:
					self.logger.info("Not the first shelf, adding to general detected shelves list.")
					self.shelves_detected_on_map.append((world_x, world_y, angle_robot_target, angle_shelf, angle_shelf_shelfn))
								
				# Draw the line and circle for visualization on the final image
				cv2.line(colored_map, (x1, y1), (x2, y2), (0, 0, 255), 2)
				cv2.circle(colored_map, (int(cx_img), int(cy_img)), 5, (0, 255, 0), -1)

			self.logger.info(f"Total shelves detected and filtered: {len(self.shelves_detected_on_map)}")
			self.publish_debug_image(self.publisher_qr_decode, colored_map)
			self.logger.info("Published final map with detected shelves to debug topic.")
			
			# Only increment task_status if a new, unvisited shelf was successfully identified
			if len(self.current_shelf) > 0:
				self.task_status += 1 # Move to next task only if a valid shelf was found and processed.


	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def camera_image_callback(self, message):
		"""Callback function to handle incoming camera images.

		Args:
			message: ROS2 message of the type sensor_msgs.msg.CompressedImage.

		Returns:
			None
		"""
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		cv2.imshow("Input Image", image)
		cv2.waitKey(1)

		current_col_index = self.table_col_count
		if current_col_index >= self.shelf_count:
			self.get_logger().warn(f"current_col_index {current_col_index} out of bounds for shelf_count {self.shelf_count}. Resetting to 0.")
			current_col_index = 0

		decoded_objects = decode(image)
		for obj in decoded_objects:
			data = obj.data.decode("utf-8")
			print(f"QR raw data: {data} (len={len(data)})")
			if len(data) == 30:
				qr_index = int(data.split('_')[0])
				if qr_index != self.last_published_qr_index + 1:
					self.qr_gui_published_for_current_shelf = False

		if self.max_object_sum_per_col[current_col_index] > 1:
			decoded_objects = decode(image)
			for obj in decoded_objects:
				data = obj.data.decode("utf-8")
				print(f"QR raw data: {data} (len={len(data)})")
				if len(data) == 30:
					qr_index = int(data.split('_')[0])
					if qr_index == self.last_published_qr_index + 1:

						# This is the next QR in sequence
						self.task_status+=1
						self.cancel_current_goal()

						self.qr_code_str = data
						self.get_logger().info(f"QR Code detected and accepted in order: {data}")
						self.next_angle_deg = float(self.qr_code_str[2:7]) # Extract and convert to float
						self.get_logger().info(f"Extracted next_angle: {self.next_angle_deg}")
						self._publish_shelf_data_with_qr(qr_index, self.qr_code_str, current_col_index)  # <-- Pass col here
						self.last_published_qr_index = qr_index
						self.qr_gui_published_for_current_shelf = True # Flag set here
		
		if self.qr_gui_published_for_current_shelf == False:
			if len(self.current_shelf) > 0:
				shelf_x, shelf_y, _, _, _ = self.current_shelf[0] 
				self.wrong_shelf=True
				
				self.temp_visited_shelves.append((shelf_x, shelf_y))

	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	def shelf_objects_callback(self, message):
		"""Callback function to handle shelf objects updates.

		Args:
			message: ROS2 message containing shelf objects data.

		Returns:
			None
		"""
		self.shelf_objects_curr = message

		# Determine which shelf/column this detection is for
		col = self.table_col_count
		if col >= self.shelf_count:
			col = 0

		current_object_sum = sum(list(message.object_count))

		# --- Object Data Logic (Max Sum) ---
		if current_object_sum >= self.max_object_sum_per_col[col]:
			self.max_object_sum_per_col[col] = current_object_sum
			self.latest_object_name_per_col[col] = list(message.object_name)
			self.latest_object_count_per_col[col] = list(message.object_count)

	def _publish_shelf_data_with_qr(self, qr_index, qr_data_str, col=None):
		current_qr = getattr(self, 'qr_code_str', "")
		qr_publishable = False
		if col is None:
			col = 0  # fallback to 0 if not provided
		# Only allow QR 1 if nothing published yet, or QR n if QR n-1 was published
		if qr_index == 1:
			if (self.latest_qr_per_col[col] != current_qr):
				qr_publishable = True
		else:
			prev_qr = f"{qr_index-1}"
			found_prev = False
			for q in self.latest_qr_per_col:
				if q and str(q).startswith(prev_qr):
					found_prev = True
					break
			if found_prev and (self.latest_qr_per_col[col] != current_qr):
				qr_publishable = True
		if qr_publishable:
			self.latest_qr_per_col[col] = current_qr
			publish_needed = True
			self.last_published_qr_index = qr_index
		else:
			pass

		# Use 'col' directly for all data access and GUI updates
		shelf_data_message = WarehouseShelf()
		shelf_data_message.object_name = list(self.latest_object_name_per_col[col])
		shelf_data_message.object_count = list(self.latest_object_count_per_col[col])
		shelf_data_message.qr_decoded = qr_data_str

		self.get_logger().info(f"Publishing combined data for shelf column {col} (triggered by QR {qr_data_str}):")
		self.get_logger().info(f"  Objects: {shelf_data_message.object_name}, Counts: {shelf_data_message.object_count}")
		self.get_logger().info(f"  QR: {shelf_data_message.qr_decoded}")

		self.publisher_shelf_data.publish(shelf_data_message)

		if PROGRESS_TABLE_GUI and box_app is not None:
			obj_str = ""
			for name, count in zip(shelf_data_message.object_name, shelf_data_message.object_count):
				obj_str += f"{name}: {count}\n"
			try:
				box_app.change_box_text(0, col, "")
				box_app.change_box_text(1, col, "")
				box_app.change_box_text(0, col, obj_str)
				box_app.change_box_color(0, col, "cyan")
				box_app.change_box_text(1, col, shelf_data_message.qr_decoded)
				box_app.change_box_color(1, col, "yellow")
				self.table_col_count = col + 1
				if self.table_col_count >= self.shelf_count:
					self.table_col_count = 0
				self.qr_gui_published_for_current_shelf = True
				
				
				if self.qr_gui_published_for_current_shelf and len(self.current_shelf) > 0:

					self.temp_visited_shelves.clear()
					self.wrong_shelf=False
					shelf_x, shelf_y, _, _, _ = self.current_shelf[0]
					self.last_x=shelf_x
					self.last_y=shelf_y 
					self.visited_shelves.append((shelf_x, shelf_y))
					self.get_logger().info(f"Added shelf at ({shelf_x:.2f}, {shelf_y:.2f}) to visited_shelves list.")
				

			except Exception as e:
				self.logger.warn(f"***************************************GUI shelf object/QR update error durros2 launch b3rb_gz_bringup sil.launch.py world:=nxp_aim_india_2025/warehouse_2 warehouse_id:=2 shelf_count:=4 initial_angle:=040.6 x:=0.0 y:=-7.0 yaw:=1.57ing publish: {e}")
				self.qr_gui_published_for_current_shelf = True
				
				
				if self.qr_gui_published_for_current_shelf and len(self.current_shelf) > 0:

					self.temp_visited_shelves.clear()
					self.wrong_shelf=False
					shelf_x, shelf_y, _, _, _ = self.current_shelf[0]
					self.last_x=shelf_x
					self.last_y=shelf_y 
					self.visited_shelves.append((shelf_x, shelf_y))
					self.get_logger().info(f"Added shelf at ({shelf_x:.2f}, {shelf_y:.2f}) to visited_shelves list.")

		self.max_object_sum_per_col[col] = 0
		self.latest_object_name_per_col[col] = []
		self.latest_object_count_per_col[col] = []
		self.latest_qr_per_col[col] = None

	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)



	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			# Important: If you cancel, you need to decide what to do next.
			# If it's for obstacle avoidance, you'll want to re-send the goal later.
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)


	def goal_result_callback(self, future):
		"""
		Callback function executed when the navigation goal reaches a final result.

		Args:
			future (rclpy.Future): The future that is result of the navigation action.
		"""
		status = future.result().status
		# NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.
		if status == GoalStatus.STATUS_SUCCEEDED:
			self.logger.info("Goal completed successfully!")
			# Increment task status if the current goal was successful,
			# except when in blind exploration and no shelves were found yet.
			if not (self.task_status == 1 and len(self.current_shelf) == 0):
				self.task_status += 1 # Advance to next task
				self.logger.info(f"Advanced to Task {self.task_status} after successful navigation.")
				if self.task_status == 3 or self.task_status == 4: # Or other appropriate task status
					self.qr_gui_published_for_current_shelf = False
			else:
				self.logger.info("Blind exploration goal completed, but no shelves found. Remaining in Task 1.")
		else:
			self.logger.warn(f"Goal failed with status: {status}")
			# If a goal fails, it stays at the current task_status,
			# and send_task will re-evaluate and likely re-attempt the goal.
			
		self.goal_completed = True  # Mark goal as completed (regardless of success/failure)
		self.goal_handle_curr = None  # Clear goal handle.
		self.current_navigation_goal = None # Clear the stored goal as it's either done or failed.


	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
			self.current_navigation_goal = None # Clear stored goal
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling goal due to excessive recoveries ({number_of_recoveries}).")
			self.cancel_current_goal()  # Unblock by discarding the current goal.
			# If you cancel here, the goal_result_callback will be called with a CANCELLED status.
			# You might want to trigger `self.avoiding_obstacle = True` here if Nav2's recoveries aren't enough.


	def send_goal_from_world_pose(self, goal_pose):
		"""
		Sends a navigation goal to the Nav2 action server.

		Args:
			goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

		Returns:
			bool: True if the goal was successfully sent, False otherwise.
		"""
		if not self.goal_completed or self.goal_handle_curr is not None:
			self.logger.info("Goal already in progress or handling, not sending new goal.")
			return False

		# Store the goal that is being sent
		self.current_navigation_goal = goal_pose
		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			self.goal_completed = True # Revert to completed if server not available
			self.current_navigation_goal = None
			return False

		# Send goal asynchronously (non-blocking).
		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		goal_future.add_done_callback(self.goal_response_callback)

		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
				       yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	# Use a timer to periodically call send_task
	# This is better than calling it in map callbacks, which can be very frequent.
	# A 1-second timer for task management is usually sufficient.
	
	task_timer = warehouse_explore.create_timer(1.0, warehouse_explore.send_task)
	
	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()