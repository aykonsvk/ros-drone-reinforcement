#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from task_env.left_right_line_follow import LeftRightLineFollowEnv
import os
from functools import reduce

if __name__ == '__main__':

    rospy.init_node('left_right_qlearn',
                    anonymous=True, log_level=rospy.ERROR)

    env = gym.make("LeftRightLineFollowEnv-v0")

    rospack = rospkg.RosPack()
    rospack.list()

    pkg_path = rospack.get_path('reinforcement_drone')
    outdir = pkg_path + '/training_results/qlearn/left_right'
    # Remove all data start from scratch
    # env = wrappers.Monitor(env, outdir, force=True)
    # Dont remove but start from scratch
    # env = wrappers.Monitor(env, outdir, force=False)
    # resume the training
    env = wrappers.Monitor(env, outdir, force=False, resume=True)

    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/drone/alpha")
    Epsilon = rospy.get_param("/drone/epsilon")
    Gamma = rospy.get_param("/drone/gamma")
    epsilon_discount = rospy.get_param("/drone/epsilon_discount")
    nepisodes = rospy.get_param("/drone/nepisodes")
    nsteps = rospy.get_param("/drone/nsteps")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0
        # Check if it already exists a training policy, and load it if so
    qfile = os.path.join(outdir,"qlearn_states.npy")
    # qfile = "qlearn_states.npy"
    if (os.path.exists(qfile)):
        print("Loading from file:",qfile)
        qlearn.load(qfile)
    else:
        print("The File doesnt exist="+str(qfile))
    rospy.logfatal(qlearn.q)

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        steps_count = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logwarn("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were=>" + str(state))
            rospy.logwarn("# action that we took=>" + str(action))
            rospy.logwarn("# reward that action gave=>" + str(reward))
            rospy.logwarn("# episode cumulated_reward=>" +
                          str(cumulated_reward))
            rospy.logwarn(
                "# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)
        
            steps_count = steps_count+1

            if not (done):
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            # input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(
        reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
    
    qlearn.save(qfile)

    rospy.logfatal(qlearn.q)

    env.close()
