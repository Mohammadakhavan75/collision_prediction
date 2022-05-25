from tarfile import XGLTYPE
from .agent import Agent
import numpy as np
import time
from .networks import ActorNetwork, CriticNetwork
from tensorflow.keras.optimizers import Adam
from .memory import PPOMemory
import tensorflow.keras as keras

class Env():
    def __init__(self, xWidth, yWidth, gridCellSize=0, senario=None):
        self.xWidth = xWidth
        self.yWidth = yWidth
        self.xMines = -40000
        self.yMines = -40000
        self.gridCellSize = gridCellSize
        self.accelerationBoundry = [-0.4, 0.4]
        self.angleBoundry = [np.deg2rad(-3) , np.deg2rad(3)]
        self.actionSpace = {'changedAccel': self.accelerationBoundry, 'changedAngle': self.angleBoundry}
        # self.accelerationBoundryCat = [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
        # self.angleBoundryCat = [-0.05235987755982989 + i * 0.013089969389957472 for i in range(9)]
        self.accelerationBoundryCat = [-0.2, -0.1, 0, 0.1, 0.2]
        self.angleBoundryCat = [-0.05235987755982989 + i * 0.026179938779914945 for i in range(5)]
        self.actionSpaceCat = {'changedAccel': self.accelerationBoundryCat, 'changedAngle': self.angleBoundryCat}
        self.senario = senario
        
        self.actor = ActorNetwork(10)
        self.actor.compile(optimizer=Adam(learning_rate=0.0003))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=0.001))
        self.memory = PPOMemory(64)
        

    def store_transition(self, state, action, probs, vals, reward, done):
        # print(f"saving memroy ID: {self.id}")
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, path):
        print('... saving models ...')
        self.chkpt_dir = path
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic')

    def model(self):
        pass
    def initAgent(self, n_actions, alpha=0.0003, batch_size=64, n_epochs=10, agnetNum=2, random=True):
        agentList = []
        if self.senario == 'Crossing':
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=0, y=0, xD=self.xWidth, yD=self.yWidth, id=0)
            agentList.append(ag)
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)
            ag.initPredefinePosition(x=self.xWidth, y=0, xD=0, yD=self.yWidth, id=1)
            agentList.append(ag)

            return agentList

        if self.senario == 'Overtaking':
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, nonVectoralSpeed=240)
            ag.initPredefinePosition(x=0, y=0, xD=self.xWidth, yD=self.yWidth, id=0)
            agentList.append(ag)
            ag = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, nonVectoralSpeed=205)
            ag.initPredefinePosition(x=10000, y=10000, xD=self.xWidth, yD=self.yWidth, id=1)
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

    def step(self, action, agent, target, deltaT, ismanouver, rewardsList, totalTime, status=None):
        if action == None:
            lastDist = agent.distfromAgent(target)
            
            agent.directMove(deltaT)
            changedAngle = 0
            # print(f"lastDist < agent.distfromAgent(target) {lastDist < agent.distfromAgent(target)}, lastDist: {lastDist},  agent.distfromAgent(target), {agent.distfromAgent(target)}")
            # return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], agent.accel['ax'], agent.accel['ay'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy'], target.accel['ax'], target.accel['ay']], self.stepReward(agent, target, rewardsList, totalTime, action), None, rewardsList
            return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy']], self.stepReward(agent, target, rewardsList, totalTime, status=None), None, rewardsList
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
            return [agent.xPos, agent.yPos, agent.speed['vx'], agent.speed['vy'], target.xPos, target.yPos, target.speed['vx'], target.speed['vy']], self.stepReward(agent, target, rewardsList, totalTime, status=None), None, rewardsList
        
    def reset(self, agents):
        for ag in agents:
            ag.resetAttr()
        # return [agents[0].xPos, agents[0].yPos, agents[0].speed['vx'], agents[0].speed['vy'], agents[0].accel['ax'], agents[0].accel['ay'], agents[1].xPos, agents[1].yPos, agents[1].speed['vx'], agents[1].speed['vy'], agents[1].accel['ax'], agents[1].accel['ay']], agents
        return [agents[0].xPos, agents[0].yPos, agents[0].speed['vx'], agents[0].speed['vy'], agents[1].xPos, agents[1].yPos, agents[1].speed['vx'], agents[1].speed['vy']], agents
            
    def selectStatus(self, agentObserver, agentTarget): # TODO: Are you sure the |v| sign in doc is not ||v|| and not mean length of vector? !!!!!!!! andaze speedd
        # SpeedMultiply = agentObserver.speed['vx'] * agentTarget.speed['vx'] + agentObserver.speed['vy'] * agentTarget.speed['vy']
        SpeedMultiply = np.dot([agentObserver.speed['vx'], agentObserver.speed['vy']], [agentTarget.speed['vx'], agentTarget.speed['vy']])
        # absSpeedMultiply = np.sqrt(agentObserver.speed['vx'] ** 2 + agentObserver.speed['vy'] ** 2) * np.sqrt(agentTarget.speed['vx'] ** 2 + agentTarget.speed['vy'] ** 2)
        absSpeedMultiply = np.linalg.norm([agentObserver.speed['vx'], agentObserver.speed['vy']]) * np.linalg.norm([agentTarget.speed['vx'], agentTarget.speed['vy']])
        cos70 = np.cos(np.deg2rad(70))
        cos145 = np.cos(np.deg2rad(145))
        # if SpeedMultiply > cos70 * absSpeedMultiply:
        # print(f"arccos: {np.arccos(SpeedMultiply/absSpeedMultiply)}")
        if np.arccos(SpeedMultiply/absSpeedMultiply) <=  ((np.pi / 180) * 70) and np.arccos(SpeedMultiply/absSpeedMultiply) >= 0:
            if agentObserver.nonVectoralSpeed > agentTarget.nonVectoralSpeed:
                status = "Overtaking"
            else:
                status = "standBy"
        # if SpeedMultiply < cos145 * absSpeedMultiply:
        elif np.arccos(SpeedMultiply/absSpeedMultiply) <= np.pi and np.arccos(SpeedMultiply/absSpeedMultiply) >= ((np.pi / 180) * 165):
            status = "HeadOn"
        # if SpeedMultiply <= cos70 * absSpeedMultiply or SpeedMultiply >= cos145 * absSpeedMultiply:
        # if np.arccos(SpeedMultiply/absSpeedMultiply) < np.cos(180 + 15) or np.arccos(SpeedMultiply/absSpeedMultiply) > np.cos(180 - 15):
        else:
            # print(f"SpeedMultiply: {SpeedMultiply}, absSpeedMultiply: {absSpeedMultiply}, cos70: {cos70 * absSpeedMultiply}, cos145: {cos145 * absSpeedMultiply}")
            # print(f"arccos: {np.arccos(SpeedMultiply/absSpeedMultiply)}")
            if agentObserver.xPos < agentTarget.xPos:
                # agentObserver.trainable = True
                status = "Crossing_giveway"
            else:
                # agentObserver.trainable = False
                status = "Crossing_standOn"

        return status

    def randomAction(w):
        changedAccel = np.random.uniform(w.actionSpace['changedAccel'][0], w.actionSpace['changedAccel'][1])
        changedAngle = np.random.uniform(w.actionSpace['changedAngle'][0], w.actionSpace['changedAngle'][1])
        return changedAccel, changedAngle

    def stepReward(self, agent, target, rewardsList, totalTime, status):
        returnReward = 0
        rewardFinal = 140000 # 1000
        rewardTowardGoalConst = 0.0001
        rewardCollision = -4 # -1000
        rewardLeft = -0.5 # -10
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
        Vo = [agent.speed['vx'], agent.speed['vy']]
        SId = np.arccos(np.dot(Vo, Vox)/(np.linalg.norm(Vo) * np.linalg.norm(Vox)))
        deltaSId = np.arccos(float("{:.5f}".format(np.dot(Vo, Vp)/(np.linalg.norm(Vo) * np.linalg.norm(Vp)))))
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
        if not agent.checkArrival():
            # print("reward Heading") 
            
            if np.abs(deltaSId) < np.abs(SId):
                R_Heading = 0.1 * np.exp(-k_c * np.abs(deltaSId))
                
            else:
                R_Heading = -0.1
                # print(f"else R_Heading: {R_Heading}")
            
            if np.abs(da) > np.abs(agent.distfromPathLine()):
                R_Cross = 0.1 * np.exp(-k_d * np.exp(agent.distfromPathLine()))
                
            else:
                R_Cross = -0.1
                # print(f"else R_Cross: {R_Cross}")

            R_Speed = 0.1 * np.exp(-k_v * np.abs(deltaUp))
            # print(f"R_Speed: {R_Speed}, deltaUp: {deltaUp}")
            # print(f"else R_Speed: {R_Speed}")
            # rHeadingCross1 = np.exp(-k_c * np.abs(agent.distfromPathLine())) * np.cos(agent.angleFromOriginalLine()) + k_r * (np.exp(-k_c * np.abs(agent.distfromPathLine())) + np.cos(agent.angleFromOriginalLine())) + np.exp(-k_v * np.abs(deltaUp)) - R_c
            # rHeadingCross2 = discounter * np.exp(-k_d * np.abs(agent.distfromPathLine())) + discounter * np.exp(-k_c * np.abs(agent.angleFromOriginalLine())) + discounter * np.exp(-k_v * np.abs(deltaUp)) - R_c
            rHeadingCross3 = R_Cross + R_Heading + R_Speed
            # print(f"R_Cross: {R_Cross}, R_Heading {R_Heading}, R_Speed: {R_Speed}")
            # print(f"rHeadingCross3: {rHeadingCross3}")
            # if agent.distfromPathLine() > 1:
            #     rHeadingCross2 -= np.log(agent.distfromPathLine())/np.log(10)
            returnReward += rHeadingCross3
            # returnReward -= rHeadingCross3
            rewardsList[agent.id][1].append(rHeadingCross3)
            # print(f"agnet ID: {agent.id}, rHeadingCross3: {rHeadingCross3}")

        """
            B) Collision Avoidance Reward function
        """

        # if agent.distfromAgent(target) < agent.acceptableDist and ismanouver:
        

        
        
        
        if agent.distfromAgent(target) < agent.acceptableDist  and status != 'Crossing_standOn':
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

        if agent.angleFromPathLine() and status != 'Crossing_standOn':
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
        # if np.log(agent.distfromPathLine()) > 5.8:
        #     returnReward -= np.log(agent.distfromPathLine())/np.log(100)
            # if agent.id == 0:
            #     print(f"np.log(agent.distfromPathLine()): {np.log(agent.distfromPathLine())}, Reward: {np.log(agent.distfromPathLine())/np.log(1000)}")
        # rewardsList[agent.id][0].append(returnReward)
        # print(f"reward: {rewardFinal/np.sqrt((agent.xPos - agent.xDest) ** 2 + (agent.yPos - agent.yDest) ** 2)}, dist: {np.sqrt((agent.xPos - agent.xDest) ** 2 + (agent.yPos - agent.yDest) ** 2)}")
        # if agent.id == 0:
            # print(f"returnReward: {returnReward}")

        return returnReward
        