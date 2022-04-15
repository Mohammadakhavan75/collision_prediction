from asyncio.log import logger
from env.evn import Env
import numpy as np
import time
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    x = 58000
    y = 58000
    deltaT = 0.05
    arrival = 0
    agnetNumber = 2
    counter = 0
    best_score = 0
    score_history = []
    totalScore = []
    total_avg_score = []
    LoggerOn = True

    world = Env(x, y)
    # agentList = world.initAgent(agnetNumber)
    agentList = world.initAgent(random=False)
    
    # world.initRender()    
    episodes=10
    for i in range(episodes):
        px = []
        py = []
        print("episode %f started!", i)
        done=[False for _ in agentList]
        observation = world.reset(agentList)
        episodeScore = []
        score = 0
        stepCounter = 0
        for ag in agentList:
            print(ag.getAttr())
        valval=False
        while not all(done):
            counter += 1
            stepCounter += 1
            if stepCounter % 5000 == 0:
                print("counter is: ", stepCounter)
                print("Agent attribute: ", agentList[0].getAttr(), agentList[1].getAttr())
            for agent in agentList:
                if not agent.checkArrival() and not agent.outofBound():
                    if agent.id == agentList[1].id:
                        action = None
                        observation_, reward, ـ, info = world.step(action, agent, agentList, deltaT)
                        observation = [agentList[0].xPos, agentList[0].yPos, agentList[0].speed['vx'], agentList[0].speed['vy'], agentList[0].accel['ax'], agentList[0].accel['ay'], agentList[1].xPos, agentList[1].yPos, agentList[1].speed['vx'], agentList[1].speed['vy'], agentList[1].accel['ax'], agentList[1].accel['ay']]
                        continue
                    if all(agent.sensor(agentList)):
                        action = agent.choose_action(observation)
                        observation_, reward, ـ, info = world.step(action, agent, agentList, deltaT)
                        score += reward
                        totalScore.append(score)
                        episodeScore.append(score)
                        agent.learn(observation, reward, observation_, False)
                        observation = observation_
                        px.append(agent.xPos)
                        py.append(agent.yPos)
                    else:
                        action = None
                        observation_, reward, ـ, info = world.step(action, agent, agentList, deltaT)
                        observation = observation_
                        px.append(agent.xPos)
                        py.append(agent.yPos)
                else:
                    done[agent.id] = True
            
            if str(observation_[0]) == 'nan':
                print(observation_[0])
                quit()
        print("episode %f finished!", i)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        total_avg_score.append(avg_score)
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        if LoggerOn:
            logPath = f"./Log/episode_{i}/"
            if not os.path.exists(logPath):
                os.mkdir(logPath)

            print("score is: ", score, "avg_score:", avg_score)

            plt.figure(figsize=(16, 10))
            plt.plot(score_history)
            plt.savefig(logPath + "score_history" + str(i) + "_" + str() + ".png")
            plt.close()

            plt.figure(figsize=(16, 10))
            plt.plot(total_avg_score)
            plt.savefig(logPath + "avg_score" + str(i) + "_" + str() + ".png")
            plt.close()

            plt.figure(figsize=(16, 10))
            plt.plot(totalScore)
            plt.savefig(logPath + "totalScore" + str(i) + "_" + str() + ".png")
            plt.close()

            plt.figure(figsize=(16, 10))
            plt.plot(episodeScore)
            plt.savefig(logPath + "episodeScore" + str(i) + "_" + str() + ".png")
            plt.close()

            plt.figure(figsize=(16, 10))
            plt.plot(px, py)
            plt.savefig(logPath + "pathAgent" + str(i) + "_" + str() + ".png")
            plt.close()



        


