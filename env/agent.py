from math import radians
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from .network import ActorCriticNetwork
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
import copy
class Agent():
    def __init__(self, acceptableDist=9260):
        self.xPos = 0
        self.yPos = 0
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
        self.nonVectoralSpeedStart = 250
        self.nonVectoralSpeed = 250
        self.angle = 0
        self.firstAngle = 0
        self.nonVectoralAccel = 0
        self.timetoManouver = 160 # 300
        self.logProbs = 0
           
    def initModelCategorical(self, n_actions=18, learning_rate=0.001, gamma=0.99):
        tf.config.set_visible_devices([], 'GPU')
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=learning_rate))

    def choose_action_categorical(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilitiesSpeed = tfp.distributions.Categorical(probs[0][:9])
        action_probabilitiesAngle = tfp.distributions.Categorical(probs[0][9:])
        actionSpeed, actionAngle = action_probabilitiesSpeed.sample(), action_probabilitiesAngle.sample()
        self.action = {'accel': actionSpeed, 'angle': actionAngle}

        return self.action

    def learnCategorical(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            # print(f"probs.numpy()[0]: {probs.numpy()[0]}")
            action_probabilitiesAccel = tfp.distributions.Categorical(probs.numpy()[0][:9])
            action_probabilitiesAngle = tfp.distributions.Categorical(probs.numpy()[0][9:])
            log_prob_accel = action_probabilitiesAccel.log_prob(self.action['accel'])
            log_prob_angle = action_probabilitiesAngle.log_prob(self.action['angle'])

            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob_accel * delta - log_prob_angle * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss
            # print(f"reward: {reward}, state_value_: {state_value_}, state_value: {state_value}, log_prob_accel: {log_prob_accel}, log_prob_angle: {log_prob_angle}")
            # print(f"delta: {delta}, actor_loss: {actor_loss}, total_loss: {total_loss}\n")

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients([
            (grad, var) 
            for (grad, var) in zip(
            gradient, self.actor_critic.trainable_variables) if grad is not None])


    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def initModel(self, n_actions=4, learning_rate=0.001, gamma=0.99):
        tf.config.set_visible_devices([], 'GPU')
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=learning_rate))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)
        action_probabilitiesSpeed = tfp.distributions.Normal(loc=probs.numpy()[0][0], scale=probs.numpy()[0][1])
        action_probabilitiesAngle = tfp.distributions.Normal(loc=probs.numpy()[0][2], scale=probs.numpy()[0][3])
        actionSpeed, actionAngle = action_probabilitiesSpeed.sample(), action_probabilitiesAngle.sample()
        self.action = {'accel': actionSpeed, 'angle': actionAngle}
        # print(probs.numpy())
        return self.action

    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probabilitiesAccel = tfp.distributions.Normal(loc=probs.numpy()[0][0], scale=float(np.power(probs.numpy()[0][1], 2)))
            action_probabilitiesAngle = tfp.distributions.Normal(loc=probs.numpy()[0][2], scale=float(np.power(probs.numpy()[0][3], 2)))
            log_prob_accel = action_probabilitiesAccel.log_prob(self.action['accel'])
            log_prob_angle = action_probabilitiesAngle.log_prob(self.action['angle'])

            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob_accel * delta - log_prob_angle * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss
            # print(f"reward: {reward}, state_value_: {state_value_}, state_value: {state_value}, log_prob_accel: {log_prob_accel}, log_prob_angle: {log_prob_angle}")
            # print(f"delta: {delta}, actor_loss: {actor_loss}, total_loss: {total_loss}\n")

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients([
            (grad, var) 
            for (grad, var) in zip(
            gradient, self.actor_critic.trainable_variables) if grad is not None])

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
            self.firstPosX = self.xPos
            self.firstPosY = self.yPos
            self.xDest = xD
            self.yDest = yD
            self.id = id
            self.initSpeed(self.nonVectoralSpeedStart)
            self.initLine()
            # self.initModel()
            self.initModelCategorical()
            self.firstSpeed = copy.deepcopy(self.speed)
        
        if all(loactionEmpty):
            self.xPos = x
            self.yPos = y
            self.firstPosX = self.xPos
            self.firstPosY = self.yPos
            self.xDest = xD
            self.yDest = yD
            self.id = id
            self.initSpeed(self.nonVectoralSpeedStart)
            self.initLine()
            # self.initModel()
            self.initModelCategorical()
            self.firstSpeed = copy.deepcopy(self.speed)
        else:
            print(loactionEmpty)
            self.initRandomPosition(xWidth, yWidth, agents, id)

    def initPredefinePosition(self, x, y, xD, yD, id):
        self.xPos = x
        self.yPos = y
        self.firstPosX = self.xPos
        self.firstPosY = self.yPos
        self.xDest = xD
        self.yDest = yD
        self.id = id
        self.initSpeed(self.nonVectoralSpeedStart)
        self.initLine()
        # self.initModel()
        self.initModelCategorical()
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

    def getAttr(self):
        return {'xPos': self.xPos, 'yPos': self.yPos, 'xDest': self.xDest, 'yDest': self.yDest,
            'speed': self.speed, 'accel': self.accel, 'Slope': self.slope, 'widthofOrigin': self.widthofOrigin}

    def checkArrival(self):
        if self.remainDist() < self.nonVectoralSpeed:
            return True
        else:
            return False

    def outofBound(self):
        if self.firstPosX < self.xDest and self.firstPosY < self.yDest:
            if self.xPos > self.xDest or self.yPos > self.yDest:
                return True
        elif self.firstPosY < self.yDest and self.firstPosY > self.yDest:
            if self.yPos > self.yDest or self.yPos < self.yDest:
                return True
        elif self.firstPosX > self.xDest and self.firstPosY < self.yDest:
            if self.xPos < self.xDest or self.yPos > self.yDest:
                return True
        elif self.firstPosY > self.yDest and self.firstPosY > self.yDest:
            if self.yPos < self.yDest or self.yPos < self.yDest:
                return True
        else:
            return False

    def distfromAgent(self, agent):
        Dist = np.sqrt((self.getAttr()['xPos'] - agent.getAttr()['xPos']) ** 2 + (self.getAttr()['yPos'] - agent.getAttr()['yPos']) ** 2)
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
        # if Si > 2 * np.pi:
        #     Si = Si - 2 * np.pi
        # if Si < 0:
        #     Si = Si + 2 * np.pi
        self.accel['ax'] = g * np.cos(Si) + u * omega * np.sin(Si)
        self.accel['ay'] = g * np.sin(Si) - u * omega * np.cos(Si)
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
        

    def angleFromPathLine(self):
        distance = [self.xDest - self.xPos, self.yDest - self.yPos]
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

        return angle

    def checkLeftofLine(self):
        __angle = np.dot(list(self.speed.values()), list(self.firstSpeed.values())) / (np.linalg.norm(list(self.speed.values())) * np.linalg.norm(list(self.firstSpeed.values())))
        return np.arccos(__angle)