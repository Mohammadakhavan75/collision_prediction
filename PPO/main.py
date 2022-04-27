from asyncio.log import logger
from env.evn import Env
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pathlib
import numpy
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

if __name__ == '__main__':
    x = 58000 * 1.62 # 1.3
    y = 58000 * 1.62
    deltaT = 0.1
    arrival = 0
    agnetNumber = 2
    counter = 0
    best_score = 0
    score_history = []
    totalScore = []
    total_avg_score = []
    mean_episode_score = []
    actionsListEpisode=[]
    LoggerOn = True
    ismanouver = False
    breakEpisode = False
    maxDistfromPath = 0
    maxDistfromPathPerEpisode = 0
    ContinousLearning = False
    N = 20
    learn_iters = 0
    dtLogger = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    world = Env(x, y)
    agentList = world.initAgent(n_actions=18, random=False)
    _ = agentList.pop(2)
    episodes=1000
    logPath = f"./Log/{dtLogger}/"
    pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)
    for i in range(episodes):
        px, pxt, pxd = [], [], []
        py, pyt, pyd = [], [], []
        print("episode %f started!", i)
        done=[False for _ in agentList]
        observation, agentList = world.reset(agentList)
        observation = [ob/x for ob in observation]
        episodeScore = []
        actionsListEpisode = [[],[]]
        outputofModel = []
        distAgent = [[],[]]
        rewardsList = [[[],[],[],[],[]], [[],[],[],[],[]]]
        score = [0,0]
        stepCounter = 0
        maxDistfromPathPerEpisode = 0
        breakEpisode = False
        manouverStarted = False
        for ag in agentList:
            print(ag.getAttr())
        while not all(done) and not breakEpisode:
            counter += 1
            stepCounter += 1
            if stepCounter % 100 == 0:
                print(f"stepCounter is: {stepCounter} , episode is : {i}")
                print("Age attribute: ", agentList[0].getAttr())
                print("Tar attribute: ", agentList[1].getAttr())
                print(f"score: {score}")
                print(f"maxDistfromPath: {maxDistfromPath}, maxDistfromPathPerEpisode: {maxDistfromPathPerEpisode}")
                logPath = f"./Log/{dtLogger}/episode_{i}/"
                pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)

                plt.figure(figsize=(16, 10))
                plt.title("Path of Agnets")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.plot(px, py, color='b')
                plt.plot(pxt, pyt, color='k')
                plt.savefig(logPath + "pathCombine" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")
 
                plt.figure(figsize=(16, 10))
                plt.plot(distAgent[0], color='b')
                plt.plot(distAgent[1], color='r')
                plt.savefig(logPath + "distAgent_100_" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                plt.figure(figsize=(16, 10))
                plt.plot(info[agentList[0].id][0], color='b')
                plt.plot(info[agentList[0].id][1], color='r')
                plt.plot(info[agentList[0].id][3], color='y')
                plt.savefig(logPath + "R_Agent_0_100_" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                plt.figure(figsize=(16, 10))
                plt.plot(info[agentList[1].id][0], color='b')
                plt.plot(info[agentList[1].id][1], color='r')
                plt.plot(info[agentList[1].id][3], color='y')
                plt.savefig(logPath + "R_Agent_1_100_" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                plt.figure(figsize=(16, 10))
                plt.title("Actor Loss of Agnets")
                plt.xlabel("x")
                plt.ylabel("value")
                plt.plot(agentList[0].actorLoss, color='r')
                plt.savefig(logPath + "ActorLoss_Agent0_" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                plt.figure(figsize=(16, 10))
                plt.title("Actor Loss of Agnets")
                plt.xlabel("x")
                plt.ylabel("value")
                plt.plot(agentList[1].actorLoss, color='r')
                plt.savefig(logPath + "ActorLoss_Agent1_" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

            for j, agent in enumerate(agentList):
                if not agent.checkArrival():
                    if agent.id == agentList[0].id:
                        target = agentList[1]
                    else:
                        target = agentList[0]

                    if all(agent.sensor(agentList, ismanouver)):
                        ismanouver = True
                        manouverStarted = True
                        action, prob, val = agent.choose_action(observation)
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList)
                        observation_ = [ob/x for ob in observation_]
                        agent.store_transition(observation, action, prob, val, reward, done[0])
                        if stepCounter % N == 0:
                            agent.learn()
                            learn_iters += 1
                            
                        observation = observation_
                        score[j] += reward
                        totalScore.append(score[j])
                        episodeScore.append(score[j])
                        actionsListEpisode[0].append(action['accel'].numpy())
                        actionsListEpisode[1].append(action['angle'].numpy())
                        distAgent[agent.id].append(agent.distfromAgent(target))
                        if agent.id == 0:
                            px.append(agent.xPos)
                            py.append(agent.yPos)
                        if agent.id == 1:
                            pxt.append(agent.xPos)
                            pyt.append(agent.yPos)
                        if maxDistfromPathPerEpisode < agent.distfromPathLine():
                            maxDistfromPathPerEpisode = agent.distfromPathLine()
                        if maxDistfromPath < agent.distfromPathLine():
                            maxDistfromPath = agent.distfromPathLine()

                    elif agent.distfromPathLine() > 0.00001:
                        action, prob, val = agent.choose_action(observation)
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList)
                        observation_ = [ob/x for ob in observation_]
                        agent.store_transition(observation, action, prob, val, reward, done[0])
                        if stepCounter % N == 0:
                            agent.learn()
                            learn_iters += 1
                        
                        observation = observation_
                        score[j] += reward
                        totalScore.append(score[j])
                        episodeScore.append(score[j])
                        actionsListEpisode[0].append(action['accel'].numpy())
                        actionsListEpisode[1].append(action['angle'].numpy())
                        distAgent[agent.id].append(agent.distfromAgent(target))
                        if agent.id == 0:
                            px.append(agent.xPos)
                            py.append(agent.yPos)
                        if agent.id == 1:
                            pxt.append(agent.xPos)
                            pyt.append(agent.yPos)
                        if maxDistfromPathPerEpisode < agent.distfromPathLine():
                            maxDistfromPathPerEpisode = agent.distfromPathLine()
                        if maxDistfromPath < agent.distfromPathLine():
                            maxDistfromPath = agent.distfromPathLine()

                    else:
                        ismanouver = False
                        action = None
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList)
                        observation_ = [ob/x for ob in observation_]
                        score[j] += reward
                        observation = observation_

                        distAgent[agent.id].append(agent.distfromAgent(target))
                        if agent.id == 0:
                            px.append(agent.xPos)
                            py.append(agent.yPos)
                        if agent.id == 1:
                            pxt.append(agent.xPos)
                            pyt.append(agent.yPos)
                else:
                    done[agent.id] = True

                if agent.outofBound():
                    breakEpisode = True
                    print("agent goes out of bound!")

                if score[j] < -100000:
                    print("agent score goes lower than -100000")
                    breakEpisode = True
            
        print("episode %f finished!", i)

        score_history.append(score[j])
        avg_score = np.mean(score_history[-100:])
        total_avg_score.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models()

        if LoggerOn:
            logPath = f"./Log/{dtLogger}/episode_{i}/"
            pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)

            print("score is: ", score, "avg_score:", avg_score)
            print(f"maxDistfromPath: {maxDistfromPath}, maxDistfromPathPerEpisode: {maxDistfromPathPerEpisode}")
            print(f"max Colision {np.sum(info[0][4])}")

            plt.figure(figsize=(16, 10))
            plt.title("Total Average Score")
            plt.xlabel("step")
            plt.ylabel("value")
            plt.plot(total_avg_score)
            plt.savefig(logPath + "end_episode_total_avg_score_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.title("total Score")
            plt.xlabel("totals steps")
            plt.ylabel("value")
            plt.plot(totalScore)
            plt.savefig(logPath + "end_episode_totalScore_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.title("Episode Score")
            plt.xlabel("step")
            plt.ylabel("value")
            plt.plot(episodeScore)
            plt.savefig(logPath + "end_episode_episodeScore_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.title("Path of Agents")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(px, py, color='b', label='A1')
            plt.plot(pxt, pyt, color='k', label='A2')
            plt.savefig(logPath + "end_episode_pathCombine_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.title("Distance Between Agents")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(distAgent[0], color='b')
            plt.plot(distAgent[1], color='r')
            plt.savefig(logPath + "end_episode_distAgent_" + str(i) + ".png")
            plt.close("all")

            # plt.figure(figsize=(16, 10))
            # plt.title("Reward Agent 1 Going Left")
            # plt.xlabel("steps")
            # plt.ylabel("values")
            # plt.plot(info[0][3], color='y')
            # plt.savefig(logPath + "end_episode_R_Left_Agent_0_" + str(i) + ".png")
            # plt.close("all")

            # plt.figure(figsize=(16, 10))
            # plt.title("Reward Agent 2 Going Left")
            # plt.xlabel("steps")
            # plt.ylabel("values")
            # plt.plot(info[1][3], color='y')
            # plt.savefig(logPath + "end_episode_R_Left_Agent_1_" + str(i) + ".png")
            # plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.title("Reward Agent 1")
            plt.xlabel("steps")
            plt.ylabel("values")
            plt.plot(info[agentList[0].id][0], color='b' , label="Going Forward") 
            plt.plot(info[agentList[0].id][1], color='r', label="From Line")
            plt.plot(info[agentList[0].id][3], color='y', label="Going Left")
            plt.savefig(logPath + "end_episode_R_Agent_" + str(agentList[0].id)+ "_" + str(i) + "_" + str(stepCounter) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.title("Reward Agent 2")
            plt.xlabel("steps")
            plt.ylabel("values")
            plt.plot(info[agentList[1].id][0], color='b', label="Going Forward")
            plt.plot(info[agentList[1].id][1], color='r',  label="From Line")
            plt.plot(info[agentList[1].id][3], color='y', label="Going Left")
            plt.savefig(logPath + "end_episode_R_Agent_" + str(agentList[1].id)+ "_" + str(i) + "_" + str(stepCounter) + ".png")
            plt.close("all")

            with open(logPath + "end_episode_actionList" + str(i) + ".txt", 'w') as f:
                for aa in actionsListEpisode:
                    f.write("%s\n" % aa)



        


