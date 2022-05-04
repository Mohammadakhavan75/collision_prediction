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
import gc
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

tf.config.set_visible_devices([], 'GPU')

def MatplotlibClearMemory():
    allfignums = plt.get_fignums()
    for i in allfignums:
        fig = plt.figure(i)
        fig.clear()
        fig.clf()
        plt.close( fig )
        del fig
        plt.clf()
        gc.collect()

def loggerMid(i, stepCounter, agentList, px, py, pxt, pyt, dtLogger, maxDistfromPath, maxDistfromPathPerEpisode):
    print(f"stepCounter is: {stepCounter} , episode is : {i}")
    print("Age attribute: ", agentList[0].getAttr())
    print("Tar attribute: ", agentList[1].getAttr())
    print(f"score: {score}")
    print(f"maxDistfromPath: {maxDistfromPath}, maxDistfromPathPerEpisode: {maxDistfromPathPerEpisode}")
    logPath = f"./Log/{dtLogger}/episode_{i}/"
    pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)

    sns.lineplot(px, py, label="Agent " + str(agentList[0].id)).set(title='Path of Agnets', xlabel="X", ylabel="Y")
    sns.lineplot(pxt, pyt, label="Agent " + str(agentList[1].id))
    plt.savefig(logPath + "pathCombine" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=distAgent[0]).set(title='Distance Between Agnets', xlabel="X", ylabel="distance")
    plt.savefig(logPath + "distAgent_100_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=info[agentList[0].id][0], label="toward goal").set(title='Rewards Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="distance")
    sns.lineplot(data=info[agentList[0].id][1], label="from line")
    sns.lineplot(data=info[agentList[0].id][2], label="collision")
    sns.lineplot(data=info[agentList[0].id][3], label="going left")
    plt.savefig(logPath + "Reward_Agent_0_100_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=info[agentList[1].id][0], label="toward goal").set(title='Rewards Agnet ' + str(agentList[1].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="distance")
    sns.lineplot(data=info[agentList[1].id][1], label="from line")
    sns.lineplot(data=info[agentList[1].id][2], label="collision")
    sns.lineplot(data=info[agentList[1].id][3], label="going left")
    plt.savefig(logPath + "Reward_Agent_1_100_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=agentList[0].actorLoss, label="loss").set(title='Actor Loss of Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    plt.savefig(logPath + "ActorLoss_Agent0_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=agentList[1].actorLoss, label="loss").set(title='Actor Loss of Agnet ' + str(agentList[1].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    plt.savefig(logPath + "ActorLoss_Agent1_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

def loggerEnd(i, stepCounter, agentList, px, py, pxt, pyt, dtLogger, maxDistfromPath, maxDistfromPathPerEpisode):
    logPath = f"./Log/{dtLogger}/episode_{i}/"
    pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)

    print("score is: ", score, "avg_score:", avg_score)
    print(f"maxDistfromPath: {maxDistfromPath}, maxDistfromPathPerEpisode: {maxDistfromPathPerEpisode}")
    print(f"max Colision {np.sum(info[0][4])}")

    sns.lineplot(data=total_avg_score).set(title='Total Average Score', xlabel="step", ylabel="value")
    plt.savefig(logPath + "end_episode_total_avg_score_" + str(i) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=totalScore).set(title='Total Score', xlabel="step", ylabel="value")
    plt.savefig(logPath + "end_episode_totalScore_" + str(i) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=episodeScore).set(title='Episode' + str(i) + ' Score', xlabel="step", ylabel="value")
    plt.savefig(logPath + "end_episode_episodeScore_" + str(i) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(px, py, label="Agent " + str(agentList[0].id)).set(title='Path of Agnets', xlabel="X", ylabel="Y")
    sns.lineplot(pxt, pyt, label="Agent " + str(agentList[1].id))
    plt.savefig(logPath + "end_episode_pathCombine_" + str(i) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=distAgent[0]).set(title='Distance Between Agnets', xlabel="X", ylabel="distance")
    plt.savefig(logPath + "end_episode_distAgent_" + str(i) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=info[agentList[0].id][0], label="toward goal").set(title='Rewards Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="distance")
    sns.lineplot(data=info[agentList[0].id][1], label="from line")
    sns.lineplot(data=info[agentList[0].id][2], label="collision")
    sns.lineplot(data=info[agentList[0].id][3], label="going left")
    plt.savefig(logPath + "end_episode_R_Agent_" + str(agentList[0].id)+ "_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=info[agentList[1].id][0], label="toward goal").set(title='Rewards Agnet ' + str(agentList[1].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="distance")
    sns.lineplot(data=info[agentList[1].id][1], label="from line")
    sns.lineplot(data=info[agentList[1].id][2], label="collision")
    sns.lineplot(data=info[agentList[1].id][3], label="going left")
    plt.savefig(logPath + "end_episode_R_Agent_" + str(agentList[1].id)+ "_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=agentList[0].actorLoss, label="loss").set(title='Actor Loss of Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    plt.savefig(logPath + "ActorLoss_Agent0_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    sns.lineplot(data=agentList[1].actorLoss, label="loss").set(title='Actor Loss of Agnet ' + str(agentList[1].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    plt.savefig(logPath + "ActorLoss_Agent1_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    with open(logPath + "end_episode_actionList" + str(i) + ".txt", 'w') as f:
        for aa in actionsListEpisode:
            f.write("%s\n" % aa)

if __name__ == '__main__':
    x = 58000 * 1.62 # 1.3
    y = 58000 * 1.62
    deltaT = 0.5
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
    LoggerMidOn = False
    ismanouver = False
    breakEpisode = False
    maxDistfromPath = 0
    maxDistfromPathPerEpisode = 0
    ContinousLearning = False
    totalTime = 0
    N = 20
    learn_iters = 0
    dtLogger = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    world = Env(x, y)
    agentList = world.initAgent(n_actions=5, random=False) # 18
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
        TTCValue = False
        # for ag in agentList:
        #     print(ag.getAttr())

        while not all(done) and not breakEpisode:
            counter += 1
            stepCounter += 1
            if stepCounter % 100 == 0 and LoggerMidOn:
                loggerMid(i, stepCounter, agentList, px, py, pxt, pyt, dtLogger, maxDistfromPath, maxDistfromPathPerEpisode)

            totalTime += deltaT
            for j, agent in enumerate(agentList):
                # if agent.id == 0:
                    # print(f"\n ID: {agent.id}, Angle: {agent.angle}", end=' ')
                if not agent.checkArrival():
                    if agent.id == agentList[0].id:
                        target = agentList[1]
                    else:
                        target = agentList[0]
                    # print(agent.sensor(agentList, ismanouver))
                    if not TTCValue and all(agent.sensor(agentList)):
                        TTCValue = True
                    if TTCValue and agent.id == 0:
                        action, prob, val = agent.choose_action(observation)
                        # if agent.id == 0:
                        #     print(f"action: {action}, action_value: {world.angleBoundryCat[action]}, angle+action: {agent.angle + world.angleBoundryCat[action]}", end=' ')
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList, totalTime)
                        # if agent.id == 0:
                        #     print(f"new angle: {agent.angle}")
                        observation_ = [ob/x for ob in observation_]
                        agent.store_transition(observation, action, prob, val, reward, done[0])
                        if stepCounter % N == 0:
                            # print(f"**********************\n********************\nstart learning in step: {stepCounter}\n**********************\n********************")
                            agent.learn()
                            learn_iters += 1
                            
                        observation = observation_
                        score[j] += reward
                        totalScore.append(score[j])
                        episodeScore.append(score[j])
                        # actionsListEpisode[0].append(action['accel'].numpy())
                        # actionsListEpisode[1].append(action['angle'].numpy())
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
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList, totalTime)
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
                    # action, prob, val = agent.choose_action(observation)
                    print(f"agent attribute is: {agent.getAttr()}")
                    observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList, totalTime)
                    reward = -1000
                    rewardsList[agent.id][1].append(reward/100)
                    observation_ = [ob/x for ob in observation_]
                    agent.store_transition(observation, action, prob, val, reward, done[0])
                    agent.learn(outofbound_loss=reward)
                    observation = observation_

                    print("agent goes out of bound!")

                if stepCounter > 3000:
                    breakEpisode = True
                    print("break of timeout!")
                # if score[j] < -100000:
                #     print("agent score goes lower than -100000")
                #     breakEpisode = True
            
        print("episode %f finished!", i)

        score_history.append(score[j])
        avg_score = np.mean(score_history[-100:])
        total_avg_score.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models()

        if LoggerOn:
            loggerEnd(i, stepCounter, agentList, px, py, pxt, pyt, dtLogger, maxDistfromPath, maxDistfromPathPerEpisode)


        


