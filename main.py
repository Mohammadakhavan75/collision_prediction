from asyncio.log import logger
from env.evn import Env
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pathlib
import numpy

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
    dtLogger = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    world = Env(x, y)
    # agentList = world.initAgent(agnetNumber)
    agentList = world.initAgent(random=False)
    # duplicateAgent = agentList[2]
    _ = agentList.pop(2)
    # world.initRender()
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
        # duplicateAgent.resetAttr()
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
        # print(duplicateAgent.getAttr())
        while not all(done) and not breakEpisode:
            counter += 1
            stepCounter += 1
            if stepCounter % 100 == 0:
                print(f"stepCounter is: {stepCounter} , episode is : {i}")
                print("Age attribute: ", agentList[0].getAttr())
                print("Tar attribute: ", agentList[1].getAttr())
                # print("Dup attribute: ", duplicateAgent.getAttr())
                print(f"score: {score}")
                print(f"maxDistfromPath: {maxDistfromPath}, maxDistfromPathPerEpisode: {maxDistfromPathPerEpisode}")
                logPath = f"./Log/{dtLogger}/episode_{i}/"
                pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)

                plt.figure(figsize=(16, 10))
                plt.plot(px, py, color='b')
                # plt.plot(pxd, pyd, color='r')
                plt.plot(pxt, pyt, color='k')
                plt.savefig(logPath + "pathCombine" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                plt.figure(figsize=(16, 10))
                plt.plot(px, py, color='b')
                # plt.plot(pxd, pyd, color='r')
                # plt.plot(pxt, pyt, color='k')
                plt.savefig(logPath + "pathAgent1" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                plt.figure(figsize=(16, 10))
                plt.plot(px, py, color='k')
                # plt.plot(pxd, pyd, color='r')
                # plt.plot(pxt, pyt, color='k')
                plt.savefig(logPath + "pathAgent2" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                # plt.figure(figsize=(16, 10))
                # plt.plot(px[-100:], py[-100:], color='b')
                # plt.plot(pxd[-100:], pyd[-100:], color='r')
                # # plt.plot(pxt[-100:], pyt[-100:], color='k')
                # plt.savefig(logPath + "pathlast_100_Combine" + str(i) + "_" + str(stepCounter) + ".png")
                # plt.close("all")
                
                # plt.figure(figsize=(16, 10))
                # plt.plot(actionsListEpisode[0], color='b')
                # plt.savefig(logPath + "actionAccel_100_" + str(i) + "_" + str(stepCounter) + ".png")
                # plt.close("all")

                # plt.figure(figsize=(16, 10))
                # plt.plot(actionsListEpisode[1], color='b')
                # plt.savefig(logPath + "actionAngle_100_" + str(i) + "_" + str(stepCounter) + ".png")
                # plt.close("all")
 
                plt.figure(figsize=(16, 10))
                plt.plot(distAgent[0], color='b')
                plt.plot(distAgent[1], color='r')
                plt.savefig(logPath + "distAgent_100_" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")


                plt.figure(figsize=(16, 10))
                plt.plot(info[0][0], color='b')
                plt.plot(info[0][1], color='r')
                plt.savefig(logPath + "R_Forward_FromLine_Agent_0_100_" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                plt.figure(figsize=(16, 10))
                plt.plot(info[1][0], color='b')
                plt.plot(info[1][1], color='r')
                plt.savefig(logPath + "R_Forward_FromLine_Agent_1_100_" + str(i) + "_" + str(stepCounter) + ".png")
                plt.close("all")

                # plt.figure(figsize=(16, 10))
                # plt.plot(info[0][0], color='b')
                # plt.plot(info[0][1], color='r')
                # plt.savefig(logPath + "R_Left_Agent_0_100_" + str(i) + "_" + str(stepCounter) + ".png")
                # plt.close("all")

                # plt.figure(figsize=(16, 10))
                # plt.plot(info[1][0], color='b')
                # plt.plot(info[1][1], color='r')
                # plt.savefig(logPath + "R_Left_Agent_1_100_" + str(i) + "_" + str(stepCounter) + ".png")
                # plt.close("all")

                # plt.figure(figsize=(16, 10))
                # plt.plot(px[-100:], py[-100:], color='b')
                # plt.plot(pxd[-100:], pyd[-100:], color='r')
                # plt.plot(pxt[-100:], pyt[-100:], color='k')
                # plt.savefig(logPath + "pathlast1000Combine" + str(i) + "_" + str(stepCounter) + ".png")
                # plt.close("all")

            for j, agent in enumerate(agentList):
                if not agent.checkArrival():
                    if agent.id == agentList[0].id:
                        target = agentList[1]
                    else:
                        target = agentList[0]

                    # print(f"agent {agent.id}, target {target.id}")
                    # if agent.id == 1:
                    #     action = None
                    #     observation_, reward, ـ, info = world.step(action, agent, agentList, deltaT, ismanouver)
                    #     observation = [agentList[0].xPos, agentList[0].yPos, agentList[0].speed['vx'], agentList[0].speed['vy'], agentList[0].accel['ax'], agentList[0].accel['ay'], agentList[1].xPos, agentList[1].yPos, agentList[1].speed['vx'], agentList[1].speed['vy'], agentList[1].accel['ax'], agentList[1].accel['ay']]
                    #     observation_ = [ob/x for ob in observation_]
                    #     observation = [ob/x for ob in observation]
                    #     pxt.append(agent.xPos)
                    #     pyt.append(agent.yPos)
                    #     continue
                    if all(agent.sensor(agentList, ismanouver)):
                        ismanouver = True
                        manouverStarted = True
                        # action = agent.choose_action(observation)
                        action = agent.choose_action_categorical(observation)
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList)
                        observation_ = [ob/x for ob in observation_]
                        score[j] += reward
                        totalScore.append(score[j])
                        episodeScore.append(score[j])
                        # agent.learn(observation, reward, observation_, False)
                        agent.learnCategorical(observation, reward, observation_, False)
                        observation = observation_
                        
                        actionsListEpisode[0].append(action['accel'].numpy())
                        actionsListEpisode[1].append(action['angle'].numpy())
                        distAgent[agent.id].append(agent.distfromAgent(target))
                        if agent.id == 0:
                            px.append(agent.xPos)
                            py.append(agent.yPos)
                        if agent.id == 1:
                            pxt.append(agent.xPos)
                            pyt.append(agent.yPos)
                        # if duplicateAgent.checkArrival() and not agent.outofBound():
                        #     duplicateAgent.directMove(deltaT)
                        #     pxd.append(duplicateAgent.xPos)
                        #     pyd.append(duplicateAgent.yPos)
                        if maxDistfromPathPerEpisode < agent.distfromPathLine():
                            maxDistfromPathPerEpisode = agent.distfromPathLine()
                        if maxDistfromPath < agent.distfromPathLine():
                            maxDistfromPath = agent.distfromPathLine()

                    elif agent.distfromPathLine() > 0.00001:
                        # action = agent.choose_action(observation)
                        action = agent.choose_action_categorical(observation)
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList)
                        observation_ = [ob/x for ob in observation_]
                        score[j] += reward
                        totalScore.append(score[j])
                        episodeScore.append(score[j])
                        # agent.learn(observation, reward, observation_, False)
                        agent.learnCategorical(observation, reward, observation_, False)
                        observation = observation_
                        
                        actionsListEpisode[0].append(action['accel'].numpy())
                        actionsListEpisode[1].append(action['angle'].numpy())
                        distAgent[agent.id].append(agent.distfromAgent(target))
                        if agent.id == 0:
                            px.append(agent.xPos)
                            py.append(agent.yPos)
                        if agent.id == 1:
                            pxt.append(agent.xPos)
                            pyt.append(agent.yPos)
                        # if duplicateAgent.checkArrival() and not agent.outofBound():
                        #     duplicateAgent.directMove(deltaT)
                        #     pxd.append(duplicateAgent.xPos)
                        #     pyd.append(duplicateAgent.yPos)
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

                        # if duplicateAgent.checkArrival() and not agent.outofBound():
                        #     duplicateAgent.directMove(deltaT)
                        #     pxd.append(duplicateAgent.xPos)
                        #     pyd.append(duplicateAgent.yPos)
                            
                else:
                    done[agent.id] = True

                if agent.outofBound():
                    breakEpisode = True
                    print("agent goes out of bound!")

                if score[j] < -2000000:
                    print("agent score goes lower than -20000")
                    breakEpisode = True
            
        print("episode %f finished!", i)

        score_history.append(score[j])
        avg_score = np.mean(score_history[-100:])
        total_avg_score.append(avg_score)
        mean_episode_score.append(np.mean(episodeScore))
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
            plt.plot(score_history)
            plt.savefig(logPath + "end_episode_score_history_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(total_avg_score)
            plt.savefig(logPath + "end_episode_total_avg_score_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(totalScore)
            plt.savefig(logPath + "end_episode_totalScore_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(episodeScore)
            plt.savefig(logPath + "end_episode_episodeScore_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(mean_episode_score)
            plt.savefig(logPath + "end_episode_mean_episode_score_" + str(i) + ".png")
            plt.close("all")

            # plt.figure(figsize=(16, 10))
            # plt.plot(px, py)
            # plt.savefig(logPath + "end_episode_pathAgent_" + str(i) + ".png")
            # plt.close("all")

            # plt.figure(figsize=(16, 10))
            # plt.plot(pxt, pyt)
            # plt.savefig(logPath + "end_episode_pathDuplicate_" + str(i) + ".png")
            # plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(px, py, color='b')
            # plt.plot(pxd, pyd, color='r')
            plt.plot(pxt, pyt, color='k')
            plt.savefig(logPath + "end_episode_pathCombine_" + str(i) + ".png")
            plt.close("all")

            # plt.figure(figsize=(16, 10))
            # plt.plot(actionsListEpisode[0], color='b')
            # plt.savefig(logPath + "end_episode_actionAccel_" + str(i) + ".png")
            # plt.close("all")

            # plt.figure(figsize=(16, 10))
            # plt.plot(actionsListEpisode[1], color='b')
            # plt.savefig(logPath + "end_episode_actionAngle_" + str(i) + ".png")
            # plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(distAgent[0], color='b')
            plt.plot(distAgent[1], color='r')
            plt.savefig(logPath + "end_episode_distAgent_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(info[0][3], color='y')
            plt.savefig(logPath + "end_episode_R_Left_Agent_0_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(info[1][3], color='y')
            plt.savefig(logPath + "end_episode_R_Left_Agent_1_" + str(i) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(info[0][0], color='b')
            plt.plot(info[0][1], color='r')
            plt.savefig(logPath + "end_episode_R_Forward_FromLine_Agent_0_100_" + str(i) + "_" + str(stepCounter) + ".png")
            plt.close("all")

            plt.figure(figsize=(16, 10))
            plt.plot(info[1][0], color='b')
            plt.plot(info[1][1], color='r')
            plt.savefig(logPath + "end_episode_R_Forward_FromLine_Agent_1_100_" + str(i) + "_" + str(stepCounter) + ".png")
            plt.close("all")

            with open(logPath + "end_episode_actionList" + str(i) + ".txt", 'w') as f:
                for aa in actionsListEpisode:
                    f.write("%s\n" % aa)



        


