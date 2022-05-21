from tarfile import XGLTYPE
from .agent import Agent
import numpy as np
import time

class Env():
    def __init__(self, xWidth, yWidth, gridCellSize=0):
        self.xWidth = xWidth
        self.yWidth = yWidth
        self.gridCellSize = gridCellSize
        self.accelerationBoundry = [-0.4, 0.4]
        self.angleBoundry = [np.deg2rad(-3) , np.deg2rad(3)]
        self.actionSpace = {'changedAccel': self.accelerationBoundry, 'changedAngle': self.angleBoundry}
        # self.accelerationBoundryCat = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
        # self.angleBoundryCat = [-0.05235987755982989 + i * 0.013089969389957472 for i in range(9)]
        self.accelerationBoundryCat = [-0.2, -0.1, 0, 0.1, 0.2]
        self.angleBoundryCat = [-0.05235987755982989 + i * 0.026179938779914945 for i in range(5)]
        self.actionSpaceCat = {'changedAccel': self.accelerationBoundryCat, 'changedAngle': self.angleBoundryCat}

    
    def initAgent(self, n_actions, alpha=0.0003, batch_size=64, n_epochs=10, agnetNum=2, random=True, senario=None):
        agentList = []
        if senario == 'Crossing':
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=0, y=0, xD=self.xWidth, yD=self.yWidth, id=0)
            agentList.append(ag)
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=self.xWidth, y=0, xD=0, yD=self.yWidth, id=1)
            agentList.append(ag)

            return agentList

        if senario == 'Overtaking':
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=0, y=0, xD=self.xWidth, yD=self.yWidth, id=0)
            agentList.append(ag)
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=11000, y=11000, xD=self.xWidth, yD=self.yWidth, id=1)
            agentList.append(ag)

            return agentList

        if random:
            for i in range(agnetNum):
                print(agentList)
                agn = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
                agn.initRandomPosition(xWidth=self.xWidth, yWidth=self.yWidth, agents=agentList, id=i)
                agentList.append(agn)
            return agentList
        elif agnetNum == 2:
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=0, y=0, xD=self.xWidth, yD=self.yWidth, id=0)
            agentList.append(ag)
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=self.xWidth, y=self.yWidth, xD=0, yD=0, id=1)
            agentList.append(ag)
            return agentList
        elif agnetNum == 4:
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=0, y=0, xD=self.xWidth, yD=self.yWidth, id=0)
            agentList.append(ag)
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=self.xWidth, y=self.yWidth, xD=0, yD=0, id=1)
            agentList.append(ag)
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=self.xWidth, y=0, xD=0, yD=self.yWidth, id=2)
            agentList.append(ag)
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=0, y=self.yWidth, xD=self.xWidth, yD=0, id=3)
            agentList.append(ag)
            return agentList

    def step(self, action, agent, target, deltaT, ismanouver, rewardsList, totalTime):
        if action == None:
            lastDist = agent.distfromAgent(target)
            
            agent.directMove(deltaT)
            changedAngle = 0
            # print(f"lastDist < agent.distfromAgent(target) {lastDist < agent.distfromAgent(target)}, lastDist: {lastDist},  agent.distfromAgent(target), {agent.distfromAgent(target)}")
            # return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], agent.accel['ax'], agent.accel['ay'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy'], target.accel['ax'], target.accel['ay']], self.stepReward(agent, target, rewardsList, totalTime, action), None, rewardsList
            return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy']], self.stepReward(agent, target, rewardsList, totalTime, action), None, rewardsList
        else:
            # print(f"action['accel'].numpy(): {action['accel'].numpy()}, action['angle'].numpy(): {action['angle'].numpy()}")
            lastDist = agent.distfromAgent(target)
            changedAccel = self.accelerationBoundryCat[action['accel']]
            changedAngle = self.angleBoundryCat[action['angle']]
            changedAccel = 0
            # changedAngle = self.angleBoundryCat[action]
            # print(f"changedAccel: {changedAccel}, changedAngle: {changedAngle}")
            agent.maneuverMove(agent.angle, agent.nonVectoralSpeed, changedAngle, changedAccel, deltaT)
            # agent.updateAcceleration(agent.angle, agent.nonVectoralSpeed, changedAngle, changedAccel, deltaT)
            # agent.updateSpeed(agent.angle, agent.nonVectoralSpeed)
            # agent.directMove(deltaT)
            # print(f"lastDist < agent.distfromAgent(target) {lastDist < agent.distfromAgent(target)}, lastDist: {lastDist},  agent.distfromAgent(target), {agent.distfromAgent(target)}")
            # return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], agent.accel['ax'], agent.accel['ay'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy'], target.accel['ax'], target.accel['ay']], self.stepReward(agent, target, rewardsList, totalTime, action), None, rewardsList
            return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy']], self.stepReward(agent, target, rewardsList, totalTime, action), None, rewardsList
        
    def reset(self, agents):
        for ag in agents:
            ag.resetAttr()
        # return [agents[0].xPos, agents[0].yPos, agents[0].speed['vx'], agents[0].speed['vy'], agents[0].accel['ax'], agents[0].accel['ay'], agents[1].xPos, agents[1].yPos, agents[1].speed['vx'], agents[1].speed['vy'], agents[1].accel['ax'], agents[1].accel['ay']], agents
        return [agents[0].xPos, agents[0].yPos, agents[0].speed['vx'], agents[0].speed['vy'], agents[1].xPos, agents[1].yPos, agents[1].speed['vx'], agents[1].speed['vy']], agents
            
    def selectStatus(self, agentObserver, agentTarget): # TODO: Are you sure the |v| sign in doc is not ||v|| and not mean length of vector? !!!!!!!! andaze speedd
        SpeedMultiply = agentObserver.speed['vx'] * agentTarget.speed['vx'] + agentObserver.speed['vy'] * agentTarget.speed['vy']
        absSpeedMultiply = np.sqrt(agentObserver.speed['vx'] ** 2 + agentObserver.speed['vy'] ** 2) * np.sqrt(agentTarget.speed['vx'] ** 2 + agentObserver.speed['vy'] ** 2)
        cos70 = np.cos(np.deg2rad(70))
        cos145 = np.cos(np.deg2rad(145))
        if SpeedMultiply > cos70 * absSpeedMultiply:
            status = "Overtaking"
        if SpeedMultiply < cos145 * absSpeedMultiply:
            status = "HeadOn"
        if SpeedMultiply <= cos70 * absSpeedMultiply or SpeedMultiply >= cos145 * absSpeedMultiply:
            status = "Crossing"

        return status

    def randomAction(w):
        changedAccel = np.random.uniform(w.actionSpace['changedAccel'][0], w.actionSpace['changedAccel'][1])
        changedAngle = np.random.uniform(w.actionSpace['changedAngle'][0], w.actionSpace['changedAngle'][1])
        return changedAccel, changedAngle

    def stepReward(self, agent, target, rewardsList, totalTime, action):
        returnReward = 0
        rewardFinal = 140000 # 1000
        rewardTowardGoalConst = 0.0001
        rewardCollision = -4 # -1000
        rewardLeft = -2 # -10
        deltaUp = [agent.firstSpeed['vx'] - agent.speed['vx'], agent.firstSpeed['vy'] - agent.speed['vy']]
        deltaUp = agent.nonVectoralSpeed - agent.nonVectoralSpeed
        R_c = 1.5
        k_r = 0.01  # 0.01
        k_c = 0.01  # 0.01
        k_v = 0.001 # 0.001
        k_d = 0.001 # 0.001
        discounter = 1 # 1
        
        Vt = [agent.xPos - agent.firstPosX, agent.yPos - agent.firstPosY]
        Va = [agent.firstSpeed['vx'] * totalTime, agent.firstSpeed['vy'] * totalTime]
        # Sia = np.arccos(np.dot(Va, Vt)/(np.linalg.norm(Va) * np.linalg.norm(Vt)))
        Vpx = [agent.firstSpeed['vx'], 0]
        Vp = [agent.firstSpeed['vx'], agent.firstSpeed['vy']]
        da = np.linalg.norm(Va) - (np.dot(Va, Vt)/np.linalg.norm(Va))
        Vox = [agent.speed['vx'], 0]
        SId = np.arccos(np.dot(Vp, Vpx)/(np.linalg.norm(Vp) * np.linalg.norm(Vpx)))
        deltaSId = np.arccos(np.dot([agent.speed['vx'], agent.speed['vy']], [agent.firstSpeed['vx'], agent.firstSpeed['vy']])/(np.linalg.norm([agent.speed['vx'], agent.speed['vy']]) * np.linalg.norm([agent.firstSpeed['vx'], agent.firstSpeed['vy']])))
        
        """
            A) Path following Reward function:
                1- Goal reward
        """

        if agent.checkArrival():
            print("reward arrival")
            # returnReward += rewardFinal
            # returnReward -= rewardFinal
        # elif agent.distfromAgent(target) < agent.acceptableDist:
        # else:
        #     # print(f"lastDist < agent.distfromAgent(target) {lastDist < agent.distfromAgent(target)}, lastDist: {lastDist},  agent.distfromAgent(target), {agent.distfromAgent(target)}")
        #     rr = np.sqrt((agent.xbPos - agent.xDest) ** 2 + (agent.ybPos - agent.yDest) ** 2)  - np.sqrt((agent.xPos - agent.xDest) ** 2 + (agent.yPos - agent.yDest) ** 2)
        #     rere = 1/(1 + np.sqrt((agent.xbPos - agent.xDest) ** 2 + (agent.ybPos - agent.yDest) ** 2)) * rr
        #     returnReward += np.exp(rere) - 1
        #     print(f"rere: {rere}, exp(rere): {np.exp(rere)}")
        #     # returnReward -= rere
        #     rewardsList[agent.id][0].append(np.exp(rere) - 1)
        # else:
        #     rewardsList[agent.id][0].append(0)

        """
            2- Heading error and Cross Error reward
        """

        # if agent.distfromAgent(target) < agent.acceptableDist:
        # if not agent.checkArrival():
        #     # print("reward Heading") 
            
        #     if np.abs(deltaSId) < np.abs(SId):
        #         R_Heading = np.exp(-k_c * np.abs(deltaSId))
        #         # print(f"R_Heading: {R_Heading}, deltaSId: {deltaSId}")
        #     else:
        #         R_Heading = -1
        #         # print(f"else R_Heading: {R_Heading}")
            
        #     if np.abs(da) > np.abs(agent.distfromAgent(target)):
        #         R_Cross = np.exp(-k_d * np.exp(agent.distfromPathLine()))
        #         # print(f"R_Cross: {R_Cross}, deltaSId: {agent.distfromPathLine()}")
        #     else:
        #         R_Cross = -1
        #         # print(f"else R_Cross: {R_Cross}")

        #     R_Speed = np.exp(-k_v * np.abs(deltaUp))
        #     # print(f"R_Speed: {R_Speed}, deltaUp: {deltaUp}")
        #     # print(f"else R_Speed: {R_Speed}")

        #     # rHeadingCross1 = np.exp(-k_c * np.abs(agent.distfromPathLine())) * np.cos(agent.angleFromOriginalLine()) + k_r * (np.exp(-k_c * np.abs(agent.distfromPathLine())) + np.cos(agent.angleFromOriginalLine())) + np.exp(-k_v * np.abs(deltaUp)) - R_c
        #     # rHeadingCross2 = discounter * np.exp(-k_d * np.abs(agent.distfromPathLine())) + discounter * np.exp(-k_c * np.abs(agent.angleFromOriginalLine())) + discounter * np.exp(-k_v * np.abs(deltaUp)) - R_c
        #     rHeadingCross3 = R_Cross + R_Heading + R_Speed
        #     # print(f"R_Cross: {R_Cross}, R_Heading {R_Heading}, R_Speed: {R_Speed}")
        #     # print(f"rHeadingCross3: {rHeadingCross3}")
        #     # if agent.distfromPathLine() > 1:
        #     #     rHeadingCross2 -= np.log(agent.distfromPathLine())/np.log(10)
        #     returnReward += rHeadingCross3 - 2
        #     # returnReward -= rHeadingCross3
        #     rewardsList[agent.id][1].append(rHeadingCross3 - 1)
            # print(f"agnet ID: {agent.id}, rHeadingCross3: {rHeadingCross3}")

        """
            B) Collision Avoidance Reward function
        """

        # if agent.distfromAgent(target) < agent.acceptableDist and ismanouver:
        

        
        
        
        if agent.distfromAgent(target) < agent.acceptableDist:
            # print("reward dist from agent is low", agent.distfromAgent(target))
            # returnReward += rewardCollision * 1/agent.distfromAgent(target)
            returnReward += rewardCollision
            # returnReward -= rewardCollision * 1/agent.distfromAgent(target)
            # rewardsList[agent.id][2].append(rewardCollision * 1/agent.distfromAgent(target))
            rewardsList[agent.id][2].append(rewardCollision)
            rewardsList[agent.id][4].append(1)
            agent.collision_occured = True
        else:
            rewardsList[agent.id][2].append(0)
            rewardsList[agent.id][4].append(0)

        if agent.angleFromPathLine():
            returnReward += rewardLeft
            # returnReward -= rewardLeft
            rewardsList[agent.id][3].append(rewardLeft)
        else:
            rewardsList[agent.id][3].append(0)
        
        
        


        # if action != None:
        #     if action != 2:
        #         # print(f"action was: {action}")
        #         returnReward = -0.1

        # if agent.backward():
        # if agent.id ==0:
        #     print(f"ID: {agent.id}, backward: {agent.backward()}")

        
        # returnReward += (rewardFinal - np.sqrt((agent.xPos - agent.xDest) ** 2 + (agent.yPos - agent.yDest) ** 2))/rewardFinal
        # if (agent.distfromPathLine() > 1 or agent.distfromPathLine() < -1) and agent.id == 0:
        #     returnReward -= 10
        if np.log(agent.distfromPathLine()) > 5.8:
            returnReward -= np.log(agent.distfromPathLine())/np.log(100)
            # if agent.id == 0:
            #     print(f"np.log(agent.distfromPathLine()): {np.log(agent.distfromPathLine())}, Reward: {np.log(agent.distfromPathLine())/np.log(1000)}")
        # rewardsList[agent.id][0].append(returnReward)
        # print(f"reward: {rewardFinal/np.sqrt((agent.xPos - agent.xDest) ** 2 + (agent.yPos - agent.yDest) ** 2)}, dist: {np.sqrt((agent.xPos - agent.xDest) ** 2 + (agent.yPos - agent.yDest) ** 2)}")
        # if agent.id == 0:
            # print(f"returnReward: {returnReward}")

        return returnReward
        