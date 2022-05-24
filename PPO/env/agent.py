from math import radians
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
import copy
import random

class Agent():
    def __init__(self, n_actions, gamma=0.99, alpha=0.0001 ,alpha1=0.0002, alpha2=0.003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=50, chkpt_dir='models/', acceptableDist=9260, nonVectoralSpeed=205, trainable=True):
        self.xPos = 0
        self.yPos = 0
        self.xbPos = 0
        self.ybPos = 0
        self.firstPosX = 0
        self.firstPosY = 0
        self.xDest = 0
        self.yDest = 0
        self.speed = {'vx': 0, 'vy': 0}
        self.accel = {'ax': 0, 'ay': 0}
        self.firstSpeed = copy.deepcopy(self.speed)
        self.id = 0
        self.acceptableDist = acceptableDist
        self.reward = 0
        self.nonVectoralSpeedStart = nonVectoralSpeed
        self.nonVectoralSpeed = nonVectoralSpeed
        self.angle = 0
        self.firstAngle = 0
        self.maxAngle = 0
        self.nonVectoralAccel = 0
        self.timetoManouver = 300 # 300 # 190
        self.logProbs = 0
        self.lastDistance = 0
        self.trainLogs = [[],[],[]]
        self.temp = 0
        self.actionAngle_ = [_ for _ in range(5)]
        self.actionAccel_ = [_ for _ in range(5)]
        self.collision_occured = False
        self.arrival = False
        self.trainable = trainable
        self.trainOccured = False

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

    def choose_action(self, Actor, Critic, observation):
        state = tf.convert_to_tensor([observation])
        probs = Actor(state)
        # dist = tfp.distributions.Categorical(probs=probs)
        distAngle = tfp.distributions.Categorical(probs=probs[0][:5])
        distAccel = tfp.distributions.Categorical(probs=probs[0][5:])
        # print(f"probs: {probs}, probs.numpy(): {probs.numpy()}")
        # print(f"probs: {probs[0]}, sum:{np.sum(probs[0])}")
        # action = np.random.choice(self.actionAble,p=probs[0])
        # action = random.choices(self.actionAngle, probs[0])
        actionAngle = random.choices(self.actionAngle_, probs[0][:5])
        actionAccel = random.choices(self.actionAccel_, probs[0][5:])
        # action = dist.sample()
        # log_prob = dist.log_prob(action)
        log_prob_angle = distAngle.log_prob(actionAngle)
        log_prob_accel = distAccel.log_prob(actionAccel)
        value = Critic(state)
        # action = action.numpy()[0]
        # action = action[0]
        actionAngle = actionAngle[0]
        actionAccel = actionAccel[0]
        value = value.numpy()[0]
        # log_prob = log_prob.numpy()[0]
        log_prob_angle = log_prob_angle.numpy()
        log_prob_accel = log_prob_accel.numpy()
        # print(f"value: {value[0]}, gamaValue: {self.gamma*value[0]}")
        # ala = self.gamma*value[0] - self.temp
        # self.temp = value[0]
        # print(f'ala: {ala}')
        # if self.id==0 :
        #     print(f'\naction: {action}, value: {value[0]},probs: {probs}')
        # self.trainLogs[1].append(value[0])
        # self.trainLogs[2].append(ala)
        action = {'angle': actionAngle, 'accel': actionAccel}
        log_prob = [*log_prob_angle, *log_prob_accel]
        return action, log_prob, value

    def learn(self, Actor, Critic, Memory, outofbound_loss=0, outofbound=False):
        loggg = False
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = Memory.generate_batches()
        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        # print("\n")
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                # a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])
                a_t += discount*reward_arr[k]
                discount *= self.gamma*self.gae_lambda
                # print(f"discount: {discount}, reward_arr: {reward_arr[k]}, gv: {self.gamma*values[k+1]}, vk+1: {values[k+1]}, vk: {values[k]}, at_first: {discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])}, a_t: {a_t}")
                # with open("./Logs.txt", 'a') as f:
                #         f.write(f"\ndiscount: {discount}, reward_arr: {reward_arr[k]}, gv: {self.gamma*values[k+1]}, vk+1: {values[k+1]}, vk: {values[k]}, at_first: {discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])}, a_t: {a_t}")

            advantage[t] = a_t
            
        # if self.id==0:
        #     print(f"advantage: {np.mean(advantage)}")
        # if self.id==0:
        
        if outofbound:
            advantage = np.asarray(outofbound_loss)
            loggg = True
            # print(f"advantage: {advantage}")
            # print(f"new advantage: {np.mean(advantage)}, {len(advantage)}")
        for _ in range(self.n_epochs):
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    # print(f"batch {batch}, action_arr: {action_arr[batch]}")
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    # old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    # actions = tf.convert_to_tensor(action_arr[batch])
                    actions_accel = tf.convert_to_tensor([ac['accel'] for ac in action_arr[batch]])
                    actions_angle = tf.convert_to_tensor([ac['angle'] for ac in action_arr[batch]])
                    probs = Actor(states)
                    # dist = tfp.distributions.Categorical(probs=probs)
                    distAngle = tfp.distributions.Categorical(probs=probs[0][:5])
                    distAccel = tfp.distributions.Categorical(probs=probs[0][5:])
                    # new_probs = dist.log_prob(actions)
                    # if self.id==0 :
                    # #     # print(f'\n\nstates: {states}')
                    # #     print(f'\nprobs: {probs}')
                    #     print(f'actions: {actions}')
                    #     print(f'old_probs: {old_probs},\n new_probs: {new_probs},\n ratio: {tf.math.exp(new_probs - old_probs)}')
                        # print(f'')
                    #     if np.isnan(probs[0][0]):
                    #         print("QQQQQQQQQ")
                    #         quit()
                    # print(f"actions: {actions},\nnew_probs: {new_probs}\n\n")
                    old_probs_accel, old_probs_angle= [], []
                    for prob in old_probs:
                        old_probs_angle.append(prob[0])
                        old_probs_accel.append(prob[1])
                    old_probs = [*old_probs_angle, *old_probs_accel]
                    new_probs_accel = distAccel.log_prob(actions_accel)
                    new_probs_angle = distAngle.log_prob(actions_angle)
                    new_probs = [*new_probs_angle, *new_probs_accel]
                    old_probs = tf.convert_to_tensor(old_probs)
                    new_probs = tf.convert_to_tensor(new_probs)
                    # print(f"\nnew_probs: {len(new_probs)}, old_probs: {len(old_probs)}")
                    critic_value = Critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    adv_dupl = []
                    # print(f"batch: {batch}")
                    # if loggg:
                    #     print(f"advantage: {advantage}")
                    #     print(f"batch: {batch}")
                    advantage = np.array(advantage)[batch.astype(int)]
                    # print(f"advantage[batch]: {advantage[batch]}")
                    for adv in list(advantage[batch]):
                        # print(f"adv: {adv}")
                        adv_dupl.append(adv)
                        adv_dupl.append(adv)

                    advantage = adv_dupl
                    # advantage = tf.convert_to_tensor(advantage)
                    # prob_ratio = tf.math.exp(new_probs - old_probs)
                    # prob_ratio = tf.math.divide(new_probs, old_probs)
                    # prob_ratio = tf.math.divide(tf.math.exp(new_probs), tf.math.exp(old_probs))
                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    # prob_ratio = prob_ratio.numpy()
                    # for indx, p_r in enumerate(prob_ratio):
                    #     if np.isnan(p_r):
                    #         prob_ratio[indx] = 1

                    # prob_ratio = tf.convert_to_tensor(prob_ratio)
                    # if outofbound:
                        # print(f"len(advantage): {advantage},  batch: {batch}, batches: {batches}")
                    # weighted_probs = advantage[batch] * prob_ratio
                    weighted_probs = advantage * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    # if self.id == 0:
                    #     print(f"\naction advantage values: {[actions[batch[0]].numpy(), advantage[batch[0]]]}")
                    #     print(f"prob_ratio: {prob_ratio}\nadvantage[batch]: {advantage[batch]}")
                    # weighted_clipped_probs = clipped_probs * advantage[batch]
                    weighted_clipped_probs = clipped_probs * advantage
                    # print(f"\n\nweighted_probs: {weighted_probs}\nweighted_clipped_probs: {weighted_clipped_probs}")
                    # print(f"\ntf.math.minimum(weighted_probs, weighted_clipped_probs): {list(set(list(weighted_clipped_probs.numpy())) - set(list(weighted_clipped_probs.numpy())))}")
                    # actor_loss = tf.math.maximum(weighted_probs, weighted_clipped_probs)
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)       # -
                    # if self.id == 0:
                    #     print(f'actor_loss: {actor_loss}')
                        # print(f"action: {actions}")
                    actor_loss = tf.math.reduce_mean(actor_loss)
                    # if outofbound:
                    #     print(f"actor_loss out of bound: {actor_loss}")
                    #     print(f"len(advantage): {len(advantage)}, batches: {batches}, batch: {batch}")
                    # if self.id == 0:
                    #     print(f'actor_loss_reduce_mean: {actor_loss}')
                    # if self.id == 0:
                    #     print(f'actor_loss: {actor_loss.numpy()}, diff_in_probs: {list(set(list(weighted_probs.numpy())) - set(list(weighted_clipped_probs.numpy())))}')
                    # if self.outofBound():
                    #     actor_loss += outofbound_loss
                    # returns = advantage[batch] + values[batch]
                    # returns = advantage + values[batch]
                    # critic_loss = keras.losses.MSE(critic_value, returns)
                    # print(f"actor_loss: {actor_loss}, critic_loss: {critic_loss}")

                # actor_loss = -actor_loss
                # critic_loss = [-cl for cl in critic_loss]
                
                actor_params = Actor.trainable_variables
                # if self.id==0 :
                #     print(f"actor_params: {actor_params[-1].numpy()}") #, probs: {probs}, new_probs: {new_probs}")
                    # print(f"Actor.trainable_variables: {Actor.trainable_variables}")
                    # quit()
                # if self.id==0:
                #     print(f"\n\nactor_params: {actor_params[-1].numpy()}")
                actor_grads = tape.gradient(actor_loss, actor_params)
                # critic_params = Critic.trainable_variables
                # critic_grads = tape.gradient(critic_loss, critic_params)
                Actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                # Critic.optimizer.apply_gradients(zip(critic_grads, critic_params))
        # print(f"actor params: {actor_params[-1].numpy()}")
        # print(f"actor_loss: {actor_loss}, critic_loss: {critic_loss}\n")
        # if self.id == 0:
        #     print(f"actor_loss: {actor_loss}\n\n")
        self.trainLogs[0].append(actor_loss.numpy())
        # self.trainLogs[1].append(critic_loss.numpy())
        Memory.clear_memory()
        self.trainOccured = True
        
    def initRandomPosition(self, xWidth, yWidth, agents, id):
        loactionEmpty = []
        x = np.random.uniform(0, xWidth)
        y = np.random.uniform(0, yWidth)
        xD = np.random.uniform(0, xWidth)
        yD = np.random.uniform(0, yWidth)
        if bool(agents):
            for i, agent in enumerate(agents):
                if i == id: # Handeling self compromise
                    continue
                else:
                    startDist = np.sqrt((x - agent.getAttr()['xPos']) ** 2 + (y - agent.getAttr()['yPos']) ** 2)
                    endDist = np.sqrt((x - agent.getAttr()['xDest']) ** 2 + (y - agent.getAttr()['yDest']) ** 2)
                    if self.acceptableDist > startDist or self.acceptableDist > endDist:
                        loactionEmpty.append(False)
                    else:
                        loactionEmpty.append(True)
        else:       
            self.xPos = x
            self.yPos = y
            self.xbPos = x
            self.ybPos = y
            self.firstPosX = self.xPos
            self.firstPosY = self.yPos
            self.xDest = xD
            self.yDest = yD
            self.id = id
            self.initSpeed(self.nonVectoralSpeedStart)
            self.initLine()
            self.firstSpeed = copy.deepcopy(self.speed)
        
        if all(loactionEmpty):
            self.xPos = x
            self.yPos = y
            self.xbPos = x
            self.ybPos = y
            self.firstPosX = self.xPos
            self.firstPosY = self.yPos
            self.xDest = xD
            self.yDest = yD
            self.id = id
            self.initSpeed(self.nonVectoralSpeedStart)
            self.initLine()
            self.firstSpeed = copy.deepcopy(self.speed)
        else:
            print(loactionEmpty)
            self.initRandomPosition(xWidth, yWidth, agents, id)

    def initPredefinePosition(self, x, y, xD, yD, id):
        self.xPos = x
        self.yPos = y
        self.xbPos = x
        self.ybPos = y
        self.firstPosX = self.xPos
        self.firstPosY = self.yPos
        self.xDest = xD
        self.yDest = yD
        self.id = id
        self.initSpeed(self.nonVectoralSpeedStart)
        self.initLine()
        self.firstSpeed = copy.deepcopy(self.speed)

    def initSpeed(self, u):
        Si = np.arctan((self.yDest - self.yPos) / (self.xDest - self.xPos)) 
        # if self.yDest - self.yPos < 0: # Check if speed vector should be negetive
        #     Si += np.pi
        if self.yDest - self.yPos < 0 and self.xDest - self.xPos < 0: # Check if speed vector should be negetive
            Si += np.pi
        if self.yDest - self.yPos > 0 and self.xDest - self.xPos < 0:
            Si += np.pi
        if self.yDest - self.yPos < 0 and self.xDest - self.xPos > 0:
            Si += 2*(np.pi)
        """
        if  -1e-10 < np.cos(Si) < 1e-10:
            cc = 0
        if  -1e-10 < np.sin(Si) < 1e-10:
            ss = 0
        """
        self.speed['vx'] = np.cos(Si) * u
        self.speed['vy'] = np.sin(Si) * u
        self.angle = Si 
        self.firstAngle = Si
        self.maxAngle = Si + np.deg2rad(30)
        print(f"ID: {self.id}, angle: {self.angle}, si: {Si}")

    def initLine(self):
        self.slope = (self.yDest - self.yPos) / (self.xDest - self.xPos)
        self.widthofOrigin = self.yPos - self.slope * self.xPos
        
    def resetAttr(self):
        self.xPos = self.firstPosX
        self.yPos = self.firstPosY
        self.speed = {'vx': 0, 'vy': 0}
        self.accel = {'ax': 0, 'ay': 0}
        self.speed = copy.deepcopy(self.firstSpeed)
        self.reward = 0
        self.nonVectoralSpeed = self.nonVectoralSpeedStart
        self.angle = self.firstAngle
        self.nonVectoralAccel = 0
        self.trainLogs = [[],[],[]]

    def getAttr(self):
        return {'firstPosX': self.firstPosX, 'firstPosY': self.firstPosY, 'xPos': self.xPos, 'yPos': self.yPos, 'xDest': self.xDest, 'yDest': self.yDest,
            'speed': self.speed, 'accel': self.accel, 'Slope': self.slope, 'widthofOrigin': self.widthofOrigin}

    def checkArrival(self):
        if self.remainDist() < self.nonVectoralSpeed:
            return True
        else:
            return False

    def outofBound(self):
        if self.firstPosX < self.xDest and self.firstPosY < self.yDest:
            if self.xPos > self.xDest or self.yPos > self.yDest or self.xPos < self.firstPosX or self.yPos < self.firstPosY:
                return True
        elif self.firstPosX < self.xDest and self.firstPosY > self.yDest:
            if self.xPos > self.xDest or self.yPos < self.yDest or self.xPos < self.firstPosX or self.yPos > self.firstPosY:
                return True
        elif self.firstPosX > self.xDest and self.firstPosY < self.yDest:
            if self.xPos < self.xDest or self.yPos > self.yDest or self.xPos > self.firstPosX or self.yPos < self.firstPosY:
                return True
        elif self.firstPosX > self.xDest and self.firstPosY > self.yDest:
            if self.xPos < self.xDest or self.yPos < self.yDest or self.xPos > self.firstPosX or self.yPos > self.firstPosY:
                return True
        else:
            return False

    def distfromAgent(self, target):
        Dist = np.sqrt((self.getAttr()['xPos'] - target.getAttr()['xPos']) ** 2 + (self.getAttr()['yPos'] - target.getAttr()['yPos']) ** 2)
        return Dist

    def distfromPathLine(self): # Distance from the line
        dist = np.abs(self.slope * self.xPos - self.yPos +  self.widthofOrigin)/np.sqrt(self.slope ** 2 + 1)
        return dist
    
    def remainDist(self):
        Dist = np.sqrt((self.xPos - self.xDest) ** 2 + (self.yPos - self.yDest) ** 2)
        return Dist

    def TTCD(self, target): # Calculating time to collision for direct move.
        ttc = Symbol('ttc')
        ttcValue = solve(((self.xPos + self.speed['vx'] * ttc) - \
                        (target.xPos + target.speed['vx'] * ttc)) ** 2 + \
                        ((self.yPos + self.speed['vy'] * ttc) - \
                        (target.yPos + target.speed['vy'] * ttc)) ** 2 - 85747600, ttc) # 46300, 85747600
        return ttcValue

    def TTCM(self, target): # Calculating time to collision for maneuver move.
        ttc = Symbol('ttc')
        ttcValue = solve(((self.xPos + self.speed['vx'] * ttc + 0.5 * self.accel['ax'] * (ttc ** 2)) - \
                        (target.xPos + target.speed['vx'] * ttc + 0.5 * target.accel['ax'] * (ttc ** 2))) ** 2 + \
                        ((self.yPos + self.speed['vy'] * ttc + 0.5 * self.accel['ay'] * (ttc ** 2)) - \
                        (target.yPos + target.speed['vy'] * ttc + 0.5 * target.accel['ay'] * (ttc ** 2))) ** 2 - 85747600, ttc ) # 46300
        # print("ttcValue for maneuver move: ", ttcValue) 

    def TTCDv2(self):
        pass

    def TTCMv2(self):
        pass

    def sensor(self, agents):
        sensorAlarm=[]
        for target in agents:
            if self.id == target.id:
                continue
            else:
                Dist = self.distfromAgent(target)
                ttc = self.TTCD(target)
                # print(f"ttc {ttc}, Dist, {Dist}")
                if not bool(ttc):
                    sensorAlarm.append(False)
                elif ttc[0].is_real:
                    if self.acceptableDist > Dist or ttc[0] < self.timetoManouver:
                        sensorAlarm.append(True)
                    else:
                        sensorAlarm.append(False)
                else:
                    sensorAlarm.append(False)
        return sensorAlarm

    def directMove(self, deltaT):
        self.xPos = self.xPos + self.speed['vx'] * deltaT
        self.yPos = self.yPos + self.speed['vy'] * deltaT

    def updateAcceleration(self, Si, u, omega, g, deltaT):
        u = u + g * deltaT
        Si = Si + omega * deltaT
        # if Si > 2 * np.pi:
        #     Si = Si - 2 * np.pi
        # if Si < 0:
        #     Si = Si + 2 * np.pi
        # if self.id == 0:
        #     print(f"g: {g}, omega: {omega}, u: {u}, Si: {Si}")
        self.nonVectoralSpeed = u
        self.angle = Si

    def updateSpeed(self, Si, u):
        return np.cos(Si) * u, np.sin(Si) * u

    def maneuverMove(self, angle, nonVectoralSpeed, changedAngle, changedAccel, deltaT):
        # print(f"ID: {self.id}, angle: {angle}")
        self.updateAcceleration(angle, nonVectoralSpeed , changedAngle, changedAccel, deltaT)
        newXspeed, newYspeed = self.updateSpeed(self.angle, self.nonVectoralSpeed)
        self.xPos = self.xPos + 0.5 * (self.speed['vx'] + newXspeed) * deltaT
        self.yPos = self.yPos + 0.5 * (self.speed['vy'] + newYspeed) * deltaT
        self.speed['vx'], self.speed['vy'] = newXspeed, newYspeed 

    def angleFromOriginalLine(self):
        distance = [self.xPos - self.firstPosX, self.yPos - self.firstPosX]
        norm = np.sqrt(distance[0] ** 2 + distance[1] ** 2)
        direction = [distance[0] / norm, distance[1] / norm]
        bulletVectorAgentPoisionLine = [direction[0] * np.sqrt(2), direction[1] * np.sqrt(2)]

        distance = [self.xDest - self.firstPosX, self.yDest - self.firstPosY]
        norm = np.sqrt(distance[0] ** 2 + distance[1] ** 2)
        direction = [distance[0] / norm, distance[1] / norm]
        bulletVectorAgentDirectLine = [direction[0] * np.sqrt(2), direction[1] * np.sqrt(2)]

        v1_u = bulletVectorAgentPoisionLine / np.linalg.norm(bulletVectorAgentPoisionLine)
        v2_u = bulletVectorAgentDirectLine / np.linalg.norm(bulletVectorAgentDirectLine)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        # print(f"ID: {self.id}, angle: {angle}")
        return angle


    def angleFromPathLine(self):
        v1 = [self.xPos - self.firstPosX, self.yPos - self.firstPosY]
        # distance = [self.xPos - self.firstPosX, self.yPos - self.firstPosX]
        # norm = np.sqrt(distance[0] ** 2 + distance[1] ** 2)
        # direction = [distance[0] / norm, distance[1] / norm]
        # bulletVectorAgentPoisionLine = [direction[0] * np.sqrt(2), direction[1] * np.sqrt(2)]

        v2 = [self.xDest - self.firstPosX, self.yDest - self.firstPosY]
        # distance = [self.xDest - self.firstPosX, self.yDest - self.firstPosY]
        # norm = np.sqrt(distance[0] ** 2 + distance[1] ** 2)
        # direction = [distance[0] / norm, distance[1] / norm]
        # bulletVectorAgentDirectLine = [direction[0] * np.sqrt(2), direction[1] * np.sqrt(2)]

        # v1_u = bulletVectorAgentPoisionLine / np.linalg.norm(bulletVectorAgentPoisionLine)
        # v2_u = bulletVectorAgentDirectLine / np.linalg.norm(bulletVectorAgentDirectLine)
        # angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        al = np.cross(v2, v1)
        # al = np.dot(v2, v1)
        # print(f"agent.id: {self.id}, v1 {v1}, v2: {v2}, al: {al}")
        return al > 0

    def backward(self):
        v1 = [self.xPos - self.firstPosX, self.yPos - self.firstPosY]
        v2 = [self.xDest - self.firstPosX, self.yDest - self.firstPosY]

        al = np.dot(v2, v1)

        return al

    def checkLeftofLine(self):
        __angle = np.dot(list(self.speed.values()), list(self.firstSpeed.values())) / (np.linalg.norm(list(self.speed.values())) * np.linalg.norm(list(self.firstSpeed.values())))
        return np.arccos(__angle)

    def checkAngleAction(self, target, lastDistance, angle):
        if angle > 0 and lastDistance > self.distfromAgent(target):
            return True
        else:
            return False