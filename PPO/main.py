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
import pickle
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

    sns.lineplot(data=agentList[0].trainLogs[0], label="loss").set(title='Actor Loss of Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    plt.savefig(logPath + "ActorLoss_Agent_0_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    plt.close("all")

    # sns.lineplot(data=agentList[0].trainLogs[1], label="loss").set(title='value of state Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    # plt.savefig(logPath + "Critic_value_Agent_0_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    # plt.close("all")

    # sns.lineplot(data=agentList[1].actorLoss, label="loss").set(title='Actor Loss of Agnet ' + str(agentList[1].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    # plt.savefig(logPath + "ActorLoss_Agent1_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    # plt.close("all")

def loggerEnd(i, stepCounter, agentList, agentsPos, dtLogger, maxDistfromPath, maxDistfromPathPerEpisode):
    logPath = f"./Log/{dtLogger}/episode_{i}/"
    pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)
    logPickel = logPath + 'pickel/'
    pathlib.Path(logPickel).mkdir(parents=True, exist_ok=True)

    print("score is: ", score_history[-1], "avg_score:", avg_score)
    print(f"maxDistfromPath: {maxDistfromPath}, maxDistfromPathPerEpisode: {maxDistfromPathPerEpisode}")
    print(f"max Colision {np.sum(info[0][4])}")

    sns.lineplot(data=total_avg_score).set(title='Total Average Score', xlabel="step", ylabel="value")
    plt.savefig(logPath + "total_avg_score_" + str(i) + ".png", dpi=500)
    plt.close("all")

    with open(logPickel + 'total_avg_score' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(total_avg_score, f)

    sns.lineplot(data=totalScore).set(title='Total Score', xlabel="step", ylabel="value")
    plt.savefig(logPath + "totalScore_" + str(i) + ".png", dpi=500)
    plt.close("all")

    with open(logPickel + 'totalScore' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(totalScore, f)

    sns.lineplot(data=episodeScore).set(title='Episode' + str(i) + ' Score', xlabel="step", ylabel="value")
    plt.savefig(logPath + "episodeScore_" + str(i) + ".png", dpi=500)
    plt.close("all")

    with open(logPickel + 'episodeScore' + str(i) + '.pkl', 'wb') as f:
        pickle.dump(episodeScore, f)

    for agent in agentList:
        sns.lineplot(agentsPos[agent.id]['x'], agentsPos[agent.id]['y'], label="Agent " + str(agent.id)).set(title='Path of Agnets', xlabel="X", ylabel="Y")
        
    plt.savefig(logPath + "path_Combine_" + str(i) + ".png", dpi=500)
    plt.close("all")

    with open(logPickel + 'path_agent_' + str(agentList[0].id) + '_' + str(i) + '.pkl', 'wb') as f:
        pickle.dump([[agentsPos[agent.id]['x'], agentsPos[agent.id]['y']] for agent in agentList], f)
    

    sns.lineplot(data=distAgent[0]).set(title='Distance Between Agnets', xlabel="X", ylabel="distance")
    plt.savefig(logPath + "distAgent_" + str(i) + ".png", dpi=500)
    plt.close("all")

    # sns.lineplot(data=info[agentList[0].id][0], label="toward goal").set(title='Rewards Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="distance")
    # sns.lineplot(data=info[agentList[0].id][1], label="from line")
    # sns.lineplot(data=info[agentList[0].id][2], label="collision")
    # sns.lineplot(data=info[agentList[0].id][3], label="going left")
    # plt.savefig(logPath + "R_Agent_" + str(agentList[0].id)+ "_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    # plt.close("all")

    # with open(logPath + 'R_Agent_' + str(agentList[0].id)+ "_"  + str(i) + '.pkl', 'wb') as f:
    #     pickle.dump(info[agentList[0].id], f)

    for agent in agentList:
        sns.lineplot(data=info[agent.id][0], label="toward goal").set(title='Rewards Agnet ' + str(agent.id) + ' step ' + str(stepCounter), xlabel="X", ylabel="distance")
        sns.lineplot(data=info[agent.id][1], label="from line")
        sns.lineplot(data=info[agent.id][2], label="collision")
        sns.lineplot(data=info[agent.id][3], label="going left")
        plt.savefig(logPath + "R_Agent_" + str(agent.id)+ "_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
        plt.close("all")

        with open(logPickel + 'R_Agent_' + str(agent.id)+ "_"  + str(i) + '.pkl', 'wb') as f:
            pickle.dump(info[agent.id], f)

    # sns.lineplot(data=agentList[0].trainLogs[0], label="loss").set(title='Actor Loss of Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    # plt.savefig(logPath + "ActorLoss_Agent_" + str(agent.id) + "_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    # plt.close("all")

    # with open(logPath + 'R_Agent_' + str(agentList[0].id)+ "_"  + str(i) + '.pkl', 'wb') as f:
    #     pickle.dump(agentList[0].trainLogs[0], f)

    # sns.lineplot(data=agentList[0].trainLogs[1], label="loss").set(title='value of state Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    # plt.savefig(logPath + "Critic_value_Agent_0_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    # plt.close("all")

    # sns.lineplot(data=agentList[0].trainLogs[2], label="loss").set(title='advantage of state Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    # plt.savefig(logPath + "Advantage_value_Agent_0_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    # plt.close("all")

    for agent in agentList:
        sns.lineplot(data=agent.trainLogs[0], label="loss").set(title='Actor Loss of Agnet ' + str(agent.id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
        plt.savefig(logPath + "ActorLoss_Agent_" + str(agent.id) + "_"  + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
        plt.close("all")

        with open(logPickel + 'Actor_Loss_Agent_' + str(agent.id)+ "_"  + str(i) + '.pkl', 'wb') as f:
            pickle.dump(agent.trainLogs[0], f)

    # sns.lineplot(data=agentList[1].trainLogs[1], label="loss").set(title='value of state Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    # plt.savefig(logPath + "Critic_value_Agent_1_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    # plt.close("all")

    # sns.lineplot(data=agentList[1].trainLogs[2], label="loss").set(title='advantage of state Agnet ' + str(agentList[0].id) + ' step ' + str(stepCounter), xlabel="X", ylabel="value")
    # plt.savefig(logPath + "Advantage_value_Agent_1_" + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
    # plt.close("all")

    for agent in agentList:
        sns.lineplot(data=action_history[agent.id][0], label="angle").set(title='Action Angle of Agnet ' + str(agent.id) + ' step ' + str(stepCounter), xlabel="Steps", ylabel="Actions")
        plt.savefig(logPath + "Actions_Angle_Agent_" + str(agent.id) + '_' + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
        plt.close("all")

        sns.lineplot(data=action_history[agent.id][1], label="acceleration").set(title='Action Acceleration of Agnet ' + str(agent.id) + ' step ' + str(stepCounter), xlabel="Steps", ylabel="Actions")
        plt.savefig(logPath + "Actions_Accel_Agent_" + str(agent.id) + '_' + str(i) + "_" + str(stepCounter) + ".png", dpi=500)
        plt.close("all")

    # with open(logPath + 'R_Agent_' + str(agentList[0].id)+ "_"  + str(i) + '.pkl', 'wb') as f:
    #     pickle.dump(agentList[0].trainLogs[0], f)

    # for agent in agentList:
    #     modelPath = f"./Log/{dtLogger}/episode_{i}/agent_{agent.id}/"
    #     # pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True)
    #     # if world.senario != 'Overtaking':
    #     agent.save_models(modelPath)

    # with open(logPath + "actionList" + str(i) + ".txt", 'w') as f:
    #     for aa in actionsListEpisode:
    #         f.write("%s\n" % aa)

if __name__ == '__main__':
    sen = 'None' # None, Crossing, Overtaking
    agnetNumber = 4

    x = 58000 * 1.62 # 1.3
    y = 58000 * 1.62
    deltaT = 0.5
    arrival = 0
    
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
    world = Env(x, y, senario=sen) # Crossing, Overtaking
    agentList = world.initAgent(n_actions=10, random=False, agnetNum=agnetNumber) # 18
    action_history=[[] for _ in agentList]
    agentDist = [0 for _ in agentList]
    episodes=10000
    logPath = f"./Log/{dtLogger}/"
    pathlib.Path(logPath).mkdir(parents=True, exist_ok=True)
    for ag in agentList:
        print(ag.getAttr())
    for i in range(episodes):
        agentsPositions = [{'x': [], 'y': []} for _ in agentList]
        px, pxt, pxd = [], [], []
        py, pyt, pyd = [], [], []
        print("\nepisode %f started!", i)
        done=[False for _ in agentList]
        observation_agent = [{'xPos': 0, 'yPos': 0, 'xSpeed': 0, 'ySpeed': 0} for g in agentList]
        # observation, agentList = world.reset(agentList)
        o__, agentList = world.reset(agentList)
        for agg in agentList:
            observation_agent[agg.id]['xPos'] = agg.xPos / x
            observation_agent[agg.id]['xPos'] = agg.yPos / x
            observation_agent[agg.id]['xSpeed'] = agg.speed['vx'] / 500
            observation_agent[agg.id]['ySpeed'] = agg.speed['vy'] / 500
        # observation = [ob/x for ob in observation]
        # ob = observation
        # observation = [ob[0]/x, ob[1]/x, ob[2]/500, ob[3]/500, ob[4]/x, ob[5]/x, ob[6]/500, ob[7]/500]
        episodeScore = []
        actionsListEpisode = [[] for _ in agentList]
        outputofModel = []
        distAgent = [[] for _ in agentList]
        rewardsList = [[[],[],[],[],[]] for _ in agentList]
        action_history=[[[],[]] for _ in agentList]
        agentDist = []
        agentDistID = []
        score = [0 for _ in agentList]
        stepCounter = 0
        stepManuover = [0 for _ in agentList]
        maxDistfromPathPerEpisode = 0
        breakEpisode = False
        TTCValue = [False for _ in agentList]
        statusALL = [['None' for k_ in agentList] for k_ in agentList]
        ontimeselection = False
        # for ag in agentList:
        #     print(ag.getAttr())
        if agnetNumber == 2:
            if sen == 'None':
                statusALL[0][1] = 'HeadOn'
                statusALL[1][0] = 'HeadOn'
            if sen == 'Crossing':
                statusALL[0][1] = 'Crossing_giveway'
                statusALL[1][0] = 'Crossing_standOn'
            if sen == 'Overtaking':
                statusALL[0][1] = 'Overtaking'
                statusALL[1][0] = 'standBy'
        
        if agnetNumber == 4:
            statusALL[0][0] = 'None'
            statusALL[0][1] = 'HeadOn'
            statusALL[0][2] = 'Crossing_giveway'
            statusALL[0][3] = 'Crossing_standOn'

            statusALL[1][0] = 'HeadOn'
            statusALL[1][1] = 'None'
            statusALL[1][2] = 'Crossing_standOn'
            statusALL[1][3] = 'Crossing_giveway'

            statusALL[2][0] = 'Crossing_standOn'
            statusALL[2][1] = 'Crossing_giveway'
            statusALL[2][2] = 'None'
            statusALL[2][3] = 'HeadOn'

            statusALL[3][0] = 'Crossing_giveway'
            statusALL[3][1] = 'Crossing_standOn'
            statusALL[3][2] = 'HeadOn'
            statusALL[3][3] = 'None'
        while not all(done) and not breakEpisode:
            counter += 1
            stepCounter += 1
            if stepCounter % 100 == 0 and LoggerMidOn:
                loggerMid(i, stepCounter, agentList, px, py, pxt, pyt, dtLogger, maxDistfromPath, maxDistfromPathPerEpisode)

            totalTime += deltaT
            for j, agent in enumerate(agentList):
                agentDist , agentDistID = [], []
                for tg in agentList:
                    if agent.id == tg.id:
                        continue
                    else:
                        agentDist.append(agent.distfromAgent(tg))
                        agentDistID.append(tg.id)

                target = agentList[agentDistID[agentDist.index(np.min(agentDist))]]

                for agg in agentList:
                    observation_agent[agg.id]['xPos'] = agg.xPos / x
                    observation_agent[agg.id]['yPos'] = agg.yPos / x
                    observation_agent[agg.id]['xSpeed'] = agg.speed['vx'] / 500
                    observation_agent[agg.id]['ySpeed'] = agg.speed['vy'] / 500
                # if agent.id == 0:
                    # print(f"\n ID: {agent.id}, Angle: {agent.angle}", end=' ')
                # if not agent.checkArrival():
                if not agent.checkArrival():
                    # if agent.id == agentList[0].id:
                    #     target = agentList[1]
                    # else:
                    #     target = agentList[0]

                    if not TTCValue[agent.id] and all(agent.sensor(agentList)):
                        TTCValue[agent.id]  = True
                        # for tg in agentList:
                        #     if agent.id == tg.id:
                        #         continue
                        #     else:
                        #         statusALL[agent.id][tg.id] = world.selectStatus(agent, tg)

                        # print(statusALL[agent.id])
                        
                    # print(f"agent ID: {agent.id}, target ID: {target.id}, status: {statusALL[agent.id][target.id]}")
                    if TTCValue[agent.id] and statusALL[agent.id][target.id] != 'Crossing_standOn' and statusALL[agent.id][target.id] != 'standBy':
                        observation = [observation_agent[agent.id]['xPos'], observation_agent[agent.id]['yPos'], observation_agent[agent.id]['xSpeed'], observation_agent[agent.id]['ySpeed'],\
                            observation_agent[target.id]['xPos'], observation_agent[target.id]['yPos'], observation_agent[target.id]['xSpeed'], observation_agent[target.id]['ySpeed']]
                        action, prob, val = agent.choose_action(observation)
                        # while agent.maxAngle < agent.angle + world.angleBoundryCat[action] or -agent.maxAngle > agent.angle + world.angleBoundryCat[action]:
                        #     action, prob, val = agent.choose_action(observation)
                        action_history[agent.id][0].append(action['angle'])
                        action_history[agent.id][1].append(action['accel'])
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList, totalTime, statusALL[agent.id][target.id])
                        ob = observation_
                        observation_ = [ob[0]/x, ob[1]/x, ob[2]/500, ob[3]/500, ob[4]/x, ob[5]/x, ob[6]/500, ob[7]/500]
                        agent.store_transition(observation, action, prob, val, reward, done[0])
                        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = agent.memory.generate_batches()
                        # print(f"ID: {agent.id}, stepManuover: {stepManuover[agent.id]}, len(memory): {len(old_prob_arr)}")
                        if stepManuover[agent.id] % N == N-1:
                            # print(f"SSSSSSSSSSStart TTTTTTTTTTrain")
                            agent.learn()
                            learn_iters += 1
                            
                        if agent.id == 0:
                            score[agent.id] += reward
                            totalScore.append(score[agent.id])
                            episodeScore.append(score[agent.id])
                        # actionsListEpisode[0].append(action['accel'].numpy())
                        # actionsListEpisode[1].append(action['angle'].numpy())
                        distAgent[agent.id].append(agent.distfromAgent(target))

                        agentsPositions[agent.id]['x'].append(agent.xPos)
                        agentsPositions[agent.id]['y'].append(agent.yPos)

                        if maxDistfromPathPerEpisode < agent.distfromPathLine():
                            maxDistfromPathPerEpisode = agent.distfromPathLine()

                        if maxDistfromPath < agent.distfromPathLine():
                            maxDistfromPath = agent.distfromPathLine()

                        stepManuover[agent.id] += 1    
                    else:
                        ismanouver = False
                        action = None
                        observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList, totalTime, statusALL[agent.id][target.id])
                        # observation_ = [ob/x for ob in observation_]
                        ob = observation_
                        observation_ = [ob[0]/x, ob[1]/x, ob[2]/500, ob[3]/500, ob[4]/x, ob[5]/x, ob[6]/500, ob[7]/500]
                        score[agent.id] += reward
                        # observation = observation_

                        distAgent[agent.id].append(agent.distfromAgent(target))
                        agentsPositions[agent.id]['x'].append(agent.xPos)
                        agentsPositions[agent.id]['y'].append(agent.yPos)

                elif not agent.arrival:
                    done[agent.id] = True
                    # breakEpisode = True
                    # observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList, totalTime)
                    # rewardsList[agent.id][1].append(reward/100)
                    # reward=50000
                    # observation_ = [ob/x for ob in observation_]
                    # agent.store_transition(observation, action, prob, val, reward, done[0])
                    # agent.learn(outofbound_loss=reward)
                    # observation = observation_
                    state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = agent.memory.generate_batches()
                    print(f"len batches : {len(batches)}")
                    if stepManuover[agent.id] % N > 1 and len(batches) != 0:
                        reward = [100 for i in range(stepManuover[agent.id] % N)]
                        agent.learn(outofbound_loss=reward, outofbound=True)
                    elif stepManuover[agent.id] % N == 0 and len(batches) != 0:
                        reward = [100 for i in range(N)]
                        agent.learn(outofbound_loss=reward, outofbound=True)
        
                    agent.arrival=True
                    for ag in agentList:
                        print(f"agent attribute is: {ag.getAttr()}")
                        print(f"Agent {ag.id} Arrived!")
                

                if agent.outofBound():
                    breakEpisode = True
                    # action, prob, val = agent.choose_action(observation)
                    for ag in agentList:
                        print(f"agent attribute is: {ag.getAttr()}")
                
                    print("agent goes out of bound!")
                    print(f"stepManuover: {stepManuover[agent.id]}")
                    # observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList, totalTime)
                    
                    # rewardsList[agent.id][1].append(reward)
                    # observation_ = [ob/x for ob in observation_]
                    # agent.store_transition(observation, action, prob, val, reward, done[0])
                    state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = agent.memory.generate_batches()
                    print(f"len batches : {len(batches)}")
                    if stepManuover[agent.id] % N > 1 and len(batches) != 0:
                        reward = [-100 for i in range(stepManuover[agent.id] % N)]
                        agent.learn(outofbound_loss=reward, outofbound=True)
                    elif stepManuover[agent.id] % N == 0 and len(batches) != 0:
                        reward = [-100 for i in range(N)]
                        agent.learn(outofbound_loss=reward, outofbound=True)
                    # observation = observation_

                    

                # if agent.collision_occured:
                #     breakEpisode = True
                #     # action, prob, val = agent.choose_action(observation)
                #     # print(f"agent attribute is: {agent.getAttr()}")
                #     observation_, reward, ـ, info = world.step(action, agent, target, deltaT, ismanouver, rewardsList, totalTime)
                #     reward = -10000
                #     rewardsList[agent.id][1].append(reward/100)
                #     observation_ = [ob/x for ob in observation_]
                #     agent.store_transition(observation, action, prob, val, reward, done[0])
                #     agent.learn(outofbound_loss=reward)
                #     observation = observation_

                #     print("Collision Occured!")



                if stepCounter > 3000:
                    breakEpisode = True
                    print("break of timeout!")
                # if score[j] < -100000:
                #     print("agent score goes lower than -100000")
                #     breakEpisode = True
            
            # print(f"\nstep: {stepCounter},\nobservation_agent: {observation_agent}")
        # print("episode %f finished!", i)

        score_history.append(score[agentList[0].id]/stepManuover[agentList[0].id])
        avg_score = np.mean(score_history[-100:])
        total_avg_score.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models()

        if LoggerOn:
            loggerEnd(i, stepCounter, agentList, agentsPositions, dtLogger, maxDistfromPath, maxDistfromPathPerEpisode)

        


        


