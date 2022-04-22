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
        self.accelerationBoundryCat = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
        self.angleBoundryCat = [-0.05235987755982989 + i * 0.013089969389957472 for i in range(9)]
        self.actionSpaceCat = {'changedAccel': self.accelerationBoundryCat, 'changedAngle': self.angleBoundryCat}
        self.rewardsList = [[],[],[],[]]

    
    def initAgent(self, agnetNum=2, random=True):
        agentList = []
        if random:
            for i in range(agnetNum):
                print(agentList)
                agn = Agent()
                agn.initRandomPosition(xWidth=self.xWidth, yWidth=self.yWidth, agents=agentList, id=i)
                agentList.append(agn)
            return agentList
        else:
            ag = Agent()
            ag.initPredefinePosition(x=0, y=0, xD=self.xWidth, yD=self.yWidth, id=0)
            agentList.append(ag)
            ag = Agent()
            ag.initPredefinePosition(x=self.xWidth, y=self.yWidth, xD=0, yD=0, id=1)
            agentList.append(ag)
            ag = Agent()
            ag.initPredefinePosition(x=0, y=0, xD=self.xWidth, yD=self.yWidth, id=2)
            agentList.append(ag)
            return agentList

    def step(self, action, agent, target, deltaT, ismanouver, rewardsList):
        if action == None:
            lastDist = agent.distfromAgent(target)
            
            agent.directMove(deltaT)
            changedAngle = 0
            # print(f"lastDist < agent.distfromAgent(target) {lastDist < agent.distfromAgent(target)}, lastDist: {lastDist},  agent.distfromAgent(target), {agent.distfromAgent(target)}")
            return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], agent.accel['ax'], agent.accel['ay'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy'], target.accel['ax'], target.accel['ay']], self.stepReward(agent, target, lastDist, changedAngle, self.rewardsList), None, self.rewardsList
        else:
            # print(f"action['accel'].numpy(): {action['accel'].numpy()}, action['angle'].numpy(): {action['angle'].numpy()}")
            lastDist = agent.distfromAgent(target)
            changedAccel = self.accelerationBoundryCat[action['accel'].numpy()]
            changedAngle = self.angleBoundryCat[action['angle'].numpy()]
            # print(f"changedAccel: {changedAccel}, changedAngle: {changedAngle}")
            agent.maneuverMove(agent.angle, agent.nonVectoralSpeed, changedAngle, changedAccel, deltaT)
            # print(f"lastDist < agent.distfromAgent(target) {lastDist < agent.distfromAgent(target)}, lastDist: {lastDist},  agent.distfromAgent(target), {agent.distfromAgent(target)}")
            # return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], agent.accel['ax'], agent.accel['ay'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy'], target.accel['ax'], target.accel['ay']], self.stepReward(agent, agentList, ismanouver), None, None
            return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], agent.accel['ax'], agent.accel['ay'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy'], target.accel['ax'], target.accel['ay']], self.stepReward(agent, target, lastDist, changedAngle, self.rewardsList), None, self.rewardsList
        
    def reset(self, agents):
        for ag in agents:
            ag.resetAttr()
        return [agents[0].xPos, agents[0].yPos, agents[0].speed['vx'], agents[0].speed['vy'], agents[0].accel['ax'], agents[0].accel['ay'], agents[1].xPos, agents[1].yPos, agents[1].speed['vx'], agents[1].speed['vy'], agents[1].accel['ax'], agents[1].accel['ay']], agents
            
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

    def stepReward(self, agent, target, lastDist, changedAngle, rewardsList):
        returnReward = 0
        # if agent.id != agentList[0].id:
        #     target = agentList[0]
        # else:
        #     target = agentList[1]
        rewardFinal = 1000 # 1000
        rewardTowardGoalConst = 0.0001
        rewardCollision = -100000 # -1000
        rewardLeft = -100 # -10
        deltaUp = [agent.firstSpeed['vx'] - agent.speed['vx'], agent.firstSpeed['vy'] - agent.speed['vy']]
        deltaUp = agent.nonVectoralSpeed - agent.nonVectoralSpeed
        R_c = 3
        k_r = 0.01
        k_c = 0.01
        k_v = 0.001
        k_d = 0.001

        # A) Path following Reward function:
            # 1- Goal reward
        if agent.checkArrival():
            print("reward arrival")
            agent.reward += rewardFinal
            returnReward += rewardFinal
            return agent.reward
        elif lastDist <= agent.distfromAgent(target):
            # print(f"lastDist < agent.distfromAgent(target) {lastDist < agent.distfromAgent(target)}, lastDist: {lastDist},  agent.distfromAgent(target), {agent.distfromAgent(target)}")
            rr = np.sqrt((agent.xbPos - agent.xDest) ** 2 + (agent.ybPos - agent.yDest) ** 2)  - np.sqrt((agent.xPos - agent.xDest) ** 2 + (agent.yPos - agent.yDest) ** 2)
            rere = 1/(1 + np.sqrt((agent.xbPos - agent.xDest) ** 2 + (agent.ybPos - agent.yDest) ** 2)) * rr
            # print(f"rr: {rere}")
            agent.reward += rere
            returnReward += rere
            rewardsList[0].append(rere)

            # 2- Heading error and Cross Error reward
        # if not agent.checkArrival() and not ismanouver:
        if not agent.checkArrival() and lastDist <= agent.distfromAgent(target):
            # print("reward Heading") 

            rHeadingCross1 = np.exp(-k_c * np.abs(agent.distfromPathLine())) * np.cos(agent.angleFromPathLine()) + k_r * (np.exp(-k_c * np.abs(agent.distfromPathLine())) + np.cos(agent.angleFromPathLine())) + np.exp(-k_v * np.abs(deltaUp)) - R_c
            rHeadingCross2 = np.exp(-k_d * np.abs(agent.distfromPathLine())) + np.exp(-k_c * np.abs(agent.angleFromPathLine())) + np.exp(-k_v * np.abs(deltaUp)) - R_c
            agent.reward += rHeadingCross2
            returnReward += rHeadingCross2
            rewardsList[1].append(rHeadingCross2)
            # print(f"rHeadingCross2: {rHeadingCross2}, agnet ID: {agent.id}")
        
        # B) Collision Avoidance Reward function
        # if agent.distfromAgent(target) < agent.acceptableDist and ismanouver:
        if agent.distfromAgent(target) < agent.acceptableDist:
            # print("reward dist from agent is low", agent.distfromAgent(target))
            agent.reward += rewardCollision * 1/agent.distfromAgent(target)
            returnReward += rewardCollision * 1/agent.distfromAgent(target)
            rewardsList[2].append(rewardCollision * 1/agent.distfromAgent(target))
        # if agent.checkLeftofLine() > 1e-06:
        # if agent.checkLeftofLine() > 1e-06:
            # print("############## reward going left of line ##################")
            # agent.reward += rewardLeft
        # if agent.checkAngleAction(target, lastDist, changedAngle):
        #     # print(f"reward going left of line: changedAngle: {changedAngle}, lastDist {lastDist}, agent.distfromAgent(target): {agent.distfromAgent(target)}")
        #     agent.reward += rewardLeft
        #     returnReward += rewardLeft
        #     rewardsList[3].append(rewardLeft)
        if agent.angleFromPathLine():
            agent.reward += rewardLeft
            returnReward += rewardLeft
            rewardsList[3].append(rewardLeft)


        return returnReward
        