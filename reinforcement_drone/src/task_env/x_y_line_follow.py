from typing_extensions import Self
import rospy
import numpy
from gym import spaces
from robot_env import drone_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3


class XYLineFollowEnv(drone_env.DroneEnv):

    register(
        id='XYLineFollowEnv-v0',
        entry_point='task_env.x_y_line_follow:XYLineFollowEnv',
        max_episode_steps=rospy.get_param('/drone/nepisodes'),
    )

    def __init__(self):
        self.stop_counter = 0
        # Only variable needed to be set here
        number_actions = rospy.get_param('/drone/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param(
            '/drone/linear_forward_speed')
        self.angular_turn_speed = rospy.get_param('/drone/angular_turn_speed')
        self.angular_speed = rospy.get_param('/drone/angular_speed')

        self.init_linear_speed_vector = Vector3()
        self.init_linear_speed_vector.x = rospy.get_param(
            '/drone/init_linear_speed_vector/x')
        self.init_linear_speed_vector.y = rospy.get_param(
            '/drone/init_linear_speed_vector/y')
        self.init_linear_speed_vector.z = rospy.get_param(
            '/drone/init_linear_speed_vector/z')

        self.init_angular_turn_speed = rospy.get_param(
            '/drone/init_angular_turn_speed')

        self.min_sonar_value = rospy.get_param('/drone/min_sonar_value')
        self.max_sonar_value = rospy.get_param('/drone/max_sonar_value')

        # Get WorkSpace Cube Dimensions
        self.work_space_x_max = rospy.get_param("/drone/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/drone/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/drone/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/drone/work_space/y_min")
        self.work_space_z_max = rospy.get_param("/drone/work_space/z_max")
        self.work_space_z_min = rospy.get_param("/drone/work_space/z_min")

        # Maximum RPY values
        self.max_roll = rospy.get_param("/drone/max_roll")
        self.max_pitch = rospy.get_param("/drone/max_pitch")
        self.max_yaw = rospy.get_param("/drone/max_yaw")

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/drone/desired_pose/x")
        self.desired_point.y = rospy.get_param("/drone/desired_pose/y")
        self.desired_point.z = rospy.get_param("/drone/desired_pose/z")

        self.desired_point_epsilon = rospy.get_param(
            "/drone/desired_point_epsilon")
        self.x_distance_epsilon = rospy.get_param('/drone/x_distance_epsilon')
        # We place the Maximum and minimum values of the X,Y,Z,R,P,Yof the pose

        high = numpy.array([
            self.work_space_x_max,
            self.work_space_y_max,
            self.max_roll,
            self.max_pitch,
            numpy.linalg.norm(self.work_space_x_min - 0),
            self.desired_point.x,
            self.desired_point.y,
        ])

        low = numpy.array([
            self.work_space_x_min,
            self.work_space_y_min,
            -1*self.max_roll,
            -1*self.max_pitch,
            numpy.linalg.norm(self.work_space_x_max - 0),
            self.desired_point.x,
            self.desired_point.y,
        ])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        # Rewards
        self.closer_to_point_reward = rospy.get_param(
            "/drone/closer_to_point_reward")
        self.not_ending_point_reward = rospy.get_param(
            "/drone/not_ending_point_reward")
        self.end_episode_points = rospy.get_param("/drone/end_episode_points")
        self.max_consequent_stops = rospy.get_param(
            '/drone/max_consequent_stops')
        self.stopped_punishment = rospy.get_param('/drone/stopped_punishment')
        self.bad_direction_punishment = rospy.get_param(
            '/drone/bad_direction_punishment')
        self.x_distance_punishment = rospy.get_param(
            'drone/x_distance_punishment')
        self.x_distance_reward = rospy.get_param(
            'drone/x_distance_reward')
        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(XYLineFollowEnv, self).__init__()

    def _set_init_pose(self):
        """
        Sets the Robot in its init linear and angular speeds
        and lands the robot. Its preparing it to be reseted in the world.
        """
        #raw_input("INIT SPEED PRESS")
        self.move_base(self.init_linear_speed_vector,
                       self.init_angular_turn_speed,
                       epsilon=0.05,
                       update_rate=10)
        # We Issue the landing command to be sure it starts landing
        #raw_input("LAND PRESS")
        # self.land()

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        #raw_input("TakeOFF PRESS")
        # We TakeOff before sending any movement commands
        self.takeoff()

        # For Info Purposes
        self.cumulated_reward = 0.0
        # We get the initial pose to mesure the distance from the desired point.
        gt_pose = self.get_gt_pose()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(
            gt_pose.position)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the drone
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        self.fix_all_driffting()
        # We convert the actions to speed movements to send to the parent class of Parrot
        linear_speed_vector = Vector3()
        angular_speed = 0.0

        if action == 0:  # STRAFE_LEFT
            linear_speed_vector.y = self.linear_forward_speed
            self.last_action = "STRAFE_LEFT"
            self.stop_counter = 0
        elif action == 1:  # STRAFE_RIGHT
            linear_speed_vector.y = -1*self.linear_forward_speed
            self.last_action = "STRAFE_RIGHT"
            self.stop_counter = 0
        elif action == 2:  # STOP
            self.stop_counter = self.stop_counter + 1
            # linear_speed_vector.y = -1*self.linear_forward_speed
            self.last_action = "STOP"
        elif action == 3:  # Forward
            linear_speed_vector.x = self.linear_forward_speed
            self.last_action = "FORWARD"
            self.stop_counter = 0
        elif action == 4:  # BACK
            linear_speed_vector.x = -1*self.linear_forward_speed
            self.last_action = "FORWARD"
            self.stop_counter = 0

        # We tell drone the linear and angular speed to set to execute
        self.move_base(linear_speed_vector,
                       angular_speed,
                       epsilon=0.05,
                       update_rate=60)

        rospy.logwarn("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        droneEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        gt_pose = self.get_gt_pose()

        # We get the orientation of the cube in RPY
        roll, pitch, _ = self.get_orientation_euler(gt_pose.orientation)

        """
        observations = [    round(gt_pose.position.x, 1),
                            round(gt_pose.position.y, 1),
                            round(gt_pose.position.z, 1),
                            round(roll, 1),
                            round(pitch, 1),
                            round(yaw, 1),
                            round(sonar_value,1)]
        """

        # Added distance from the cable as observation
        # We simplify a bit the spatial grid to make learning faster
        observations = [
            int(gt_pose.position.x),
            int(gt_pose.position.y),
            round(roll, 1),
            round(pitch, 1),
            self.get_x_distance_from_desired_line(gt_pose.position),
            self.desired_point.x,
            self.desired_point.y,

        ]

        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    def _is_done(self, observations):
        """
        The done can be done due to three reasons:
        1) It went outside the workspace
        2) It detected something with the sonar that is too close
        3) It flipped due to a crash or something
        4) It has reached the desired point
        """

        episode_done = False
        self.info = {}
        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]

        current_orientation = Point()
        current_orientation.x = observations[2]
        current_orientation.y = observations[3]

        distance_from_cable = observations[4]

        is_inside_workspace_now = self.is_inside_workspace(current_position)
        # drone_flipped = self.drone_has_flipped(current_orientation)
        has_reached_des_point = self.is_in_desired_position(
            current_position, self.desired_point_epsilon)

        rospy.logwarn(">>>>>> DONE RESULTS <<<<<")
        if not is_inside_workspace_now:
            self.info = {'message': 'outside_geofence'}
            rospy.logerr("is_inside_workspace_now=" +
                         str(is_inside_workspace_now))
            rospy.logerr(current_position)
        else:
            rospy.logwarn("is_inside_workspace_now=" +
                          str(is_inside_workspace_now))

        if has_reached_des_point:
            self.info = {'message': 'reached_goal'}
            rospy.logerr("has_reached_des_point="+str(has_reached_des_point))
        else:
            rospy.logwarn("has_reached_des_point="+str(has_reached_des_point))

        # We see if we are outside the Learning Space
        episode_done = not(
            is_inside_workspace_now) or has_reached_des_point

        if episode_done:
            rospy.logerr("episode_done====>"+str(episode_done))
        else:
            rospy.logwarn("episode_done====>"+str(episode_done))

        return episode_done

    def _compute_reward(self, observations, done):

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]

        distance_from_cable = observations[4]

        distance_from_des_point = self.get_distance_from_desired_point(
            current_position)
        distance_difference = distance_from_des_point - \
            self.previous_distance_from_des_point

        if not done:
            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward = self.closer_to_point_reward
            else:
                rospy.logwarn("INCREASE IN DISTANCE BAD")
                reward = self.bad_direction_punishment

            # Punishing for beeing close/fard to cable
            # Should I punish/Rewad improvements instead???
            # Should I also reward?
            # Here will go reading from sensor instead of current position
            # rospy.logerr(self.is_ideal_from_cable(current_position))
            # rospy.logerr(distance_from_cable)
            if self.is_ideal_from_cable(current_position):
                reward = reward + self.x_distance_reward
            else:
                reward = reward + self.x_distance_punishment

            # Punish for staying at same spot
            if self.stop_counter >= self.max_consequent_stops:
                reward = reward + self.stopped_punishment

        else:
            if self.is_in_desired_position(current_position, epsilon=self.desired_point_epsilon):
                reward = self.end_episode_points
            else:
                reward = -1*self.end_episode_points

        self.previous_distance_from_des_point = distance_from_des_point

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods

    def is_in_desired_position(self, current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False

        x_pos_max = self.desired_point.x + epsilon
        x_pos_min = self.desired_point.x - epsilon

        y_pos_max = self.desired_point.y + epsilon
        y_pos_min = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_max) and (x_current >= x_pos_min)
        y_pos_are_close = (y_current <= y_pos_max) and (y_current >= y_pos_min)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        rospy.logwarn("###### IS DESIRED POS ? ######")
        rospy.logwarn("current_position"+str(current_position))
        rospy.logwarn("y_pos_max"+str(y_pos_max) +
                      ",y_pos_min="+str(y_pos_min))
        rospy.logwarn("y_pos_are_close"+str(y_pos_are_close))
        rospy.logwarn("is_in_desired_pos"+str(is_in_desired_pos))
        rospy.logwarn("############")
        rospy.logwarn(f"{y_current} <= {y_pos_max}")
        rospy.logwarn(y_current <= y_pos_max)
        rospy.logwarn(f"{y_current} >= {y_pos_min}")
        rospy.logwarn(y_current >= y_pos_min)

        return is_in_desired_pos

    def is_inside_workspace(self, current_position):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_inside = False

        rospy.logwarn("##### INSIDE WORK SPACE? #######")
        rospy.logwarn("Y current_position"+str(current_position))
        rospy.logwarn("work_space_y_max"+str(self.work_space_y_max) +
                      ",work_space_y_min="+str(self.work_space_y_min))
        rospy.logwarn("############")

        if current_position.x >= self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y >= self.work_space_y_min and current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside

    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def is_ideal_from_cable(self, coordinates):
        # Here was code from the depth sensor (removed due to time and resources constrains (a lot of data))
        # I did simple calculation instead cable is at x=0 and ideal distance is -5 (arbitrary picked)
        x_current = coordinates.x
        x_pos_max = self.desired_point.x + self.x_distance_epsilon
        x_pos_min = self.desired_point.x - self.x_distance_epsilon
        return (x_current <= x_pos_max) and (x_current >= x_pos_min)

    def get_x_distance_from_desired_line(self, current_position):
        # Here was code from the depth sensor (removed due to time and resources constrains (a lot of data))
        # I did simple calculation instead cable is at x=0
        return numpy.linalg.norm(current_position.x - 0)

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y))
        b = numpy.array((p_end.x, p_end.y))

        distance = numpy.linalg.norm(a - b)

        return distance

    def fix_all_driffting(self):
        self.fix_yaw_drift(-0.1, 0.1)

        self.fix_vertical_drift(self.work_space_z_min + 1,
                                self.work_space_z_max - 1)

    def _get_info(self):
        return self.info
