from math import radians
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
import copy

import tensorflow.keras as keras
from .memory import PPOMemory
from .networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, n_actions, gamma=0.99, alpha=0.0001 ,alpha1=0.0002, alpha2=0.003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=20, chkpt_dir='models/', acceptableDist=9260):
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
        self.nonVectoralSpeedStart = 205
        self.nonVectoralSpeed = 205
        self.angle = 0
        self.firstAngle = 0
        self.nonVectoralAccel = 0
        self.timetoManouver = 300 # 300 # 190
        self.logProbs = 0
        self.lastDistance = 0
        self.actorLoss = []

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha1))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha2))
        self.memory = PPOMemory(batch_size)
           
    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic')

    # def choose_action(self, observation):
    #     state = tf.convert_to_tensor([observation])

    #     probs = self.actor(state)
    #     dist = tfp.distributions.Categorical(probs)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)
    #     value = self.critic(state)

    #     action = action.numpy()[0]
    #     value = value.numpy()[0]
    #     log_prob = log_prob.numpy()[0]

    #     return action, log_prob, value


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs[0][:9])
        # dist_angle = tfp.distributions.Categorical(probs[0][:9])
        # dist_accel = tfp.distributions.Categorical(probs[0][9:])
        action = dist.sample()
        # action_accel = dist_accel.sample()
        # action_angle = dist_angle.sample()
        log_prob = dist.log_prob(action)
        # log_prob_accel = dist_accel.log_prob(action_accel)
        # log_prob_angle = dist_angle.log_prob(action_angle)

        value = self.critic(state)

        value = value.numpy()[0]
        # action = {'accel': action_accel, 'angle': action_angle}
        # log_prob = {'accel': log_prob_accel, 'angle': log_prob_angle}
        # log_prob = (log_prob_accel + log_prob_angle)/2
        
        return action, log_prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()
            # if self.id==0:
            #     print(f"\nagnet ID: {self.id}, reward_arr: {reward_arr}")
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            
            # if self.id==0:
            #     print(f"\nadvantage: {advantage}\n\n")

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    # actions_accel = tf.convert_to_tensor([ac['accel'] for ac in action_arr[batch]])
                    # actions_angle = tf.convert_to_tensor([ac['angle'] for ac in action_arr[batch]])
                    probs = self.actor(states)
                    # if self.id==0:
                    #     print(f'\n\nbatch spliter\nprobs: {probs}')
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)
                    # print(f"actions: {actions},\nnew_probs: {new_probs}\n\n")
                    # new_probs_accel = dist.log_prob(actions_accel)
                    # new_probs_angle = dist.log_prob(actions_angle)
                    # new_probs = (new_probs_accel + new_probs_angle)/2

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    # print(f"prob_ratio: {prob_ratio}, self.policy_clip: {self.policy_clip}, clipped_probs: {list(set(list(clipped_probs.numpy())) - set(list(prob_ratio.numpy())))}")
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)      
                    actor_loss = tf.math.reduce_mean(actor_loss)
                    # print(f'actor_loss: {actor_loss.numpy()}, diff_in_probs: {list(set(list(weighted_probs.numpy())) - set(list(weighted_clipped_probs.numpy())))}')
                    returns = advantage[batch] + values[batch]
                    critic_loss = keras.losses.MSE(critic_value, returns)

                # actor_loss = -actor_loss
                # critic_loss = [-cl for cl in critic_loss]
                
                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))
        # print(f"actor params: {actor_params[-1].numpy()}")
        # print(f"actor_loss: {actor_loss}, critic_loss: {critic_loss}\n")
        self.actorLoss.append(actor_loss.numpy())
        self.memory.clear_memory()

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
        self.actorLoss = []

    def getAttr(self):
        return {'firstPosX': {self.firstPosX}, 'firstPosY': {self.firstPosY}, 'xPos': self.xPos, 'yPos': self.yPos, 'xDest': self.xDest, 'yDest': self.yDest,
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
        Dist = np.sqrt((self.getAttr()['xPos'] - self.getAttr()['xDest']) ** 2 + (self.getAttr()['yPos'] - self.getAttr()['yDest']) ** 2)
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

    def sensor(self, agents, ismanouver):
        sensorAlarm=[]
        for target in agents:
            if self.id == target.id:
                continue
            else:
                # if ismanouver:
                #     Dist = self.distfromAgent(target)
                #     ttc = self.TTCM(target)
                # else:
                if True:
                    Dist = self.distfromAgent(target)
                    ttc = self.TTCD(target)
                    # print(f"ttc {ttc}, Dist, {Dist}")
                if not bool(ttc):
                    sensorAlarm.append(False)
                    break
                elif ttc[0].is_real:
                    if self.acceptableDist > Dist or ttc[0] < self.timetoManouver:
                        sensorAlarm.append(True)
                        break
                    else:
                        sensorAlarm.append(False)
                        break
                else:
                    sensorAlarm.append(False)
                    break

        return sensorAlarm

    def directMove(self, deltaT):
        self.xPos = self.xPos + self.speed['vx'] * deltaT
        self.yPos = self.yPos + self.speed['vy'] * deltaT

    def updateAcceleration(self, Si, u, omega, g, deltaT):
        u = u + g * deltaT
        Si = Si + omega * deltaT
        if Si > 2 * np.pi:
            Si = Si - 2 * np.pi
        if Si < 0:
            Si = Si + 2 * np.pi
        self.accel['ax'] = g * np.cos(Si) + u * omega * np.sin(Si)
        self.accel['ay'] = g * np.sin(Si) - u * omega * np.cos(Si)
        # if self.id == 0:
        #     print(f"g: {g}, omega: {omega}, u: {u}, Si: {Si}, self.accel: {self.accel}")
        self.nonVectoralSpeed = u
        self.angle = Si

    def updateSpeed(self, Si, u):
        self.speed['vx'] = np.cos(Si) * u
        self.speed['vy'] = np.sin(Si) * u

    def maneuverMove(self, angle, nonVectoralSpeed, changedAngle, changedAccel, deltaT):
        # print(f"ID: {self.id}, angle: {angle}")
        self.updateAcceleration(angle, nonVectoralSpeed , changedAngle, changedAccel, deltaT)
        self.xPos = self.xPos + self.speed['vx'] * deltaT + 0.5 * self.accel['ax'] * (deltaT ** 2)
        self.yPos = self.yPos + self.speed['vy'] * deltaT + 0.5 * self.accel['ay'] * (deltaT ** 2)
        self.updateSpeed(angle, nonVectoralSpeed)

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

        al  = np.cross(v2, v1)
        # print(f"agent.id: {self.id}, v1 {v1}, v2: {v2}, al: {al}")
        return al > 0

    def checkLeftofLine(self):
        __angle = np.dot(list(self.speed.values()), list(self.firstSpeed.values())) / (np.linalg.norm(list(self.speed.values())) * np.linalg.norm(list(self.firstSpeed.values())))
        return np.arccos(__angle)

    def checkAngleAction(self, target, lastDistance, angle):
        if angle > 0 and lastDistance > self.distfromAgent(target):
            return True
        else:
            return False