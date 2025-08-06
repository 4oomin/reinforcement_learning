"""
[2] DQN

기존 Q-learning에서는 Q-table에 Q-value를 저장 및 업데이트

DQN은 q-network라는 state를 입력으로 주고 각 action에 대한 Q-value를 출력함

1) Experince replay

현재 상태에서 action 실행 시 생성되는 (현상태,행동,보상,다음상태) 페어를 저장해두고 여기서 random 샘플링하여 Q-update함

->기존에는 행동을 radom 선택하여 같은 페어가 중복적으로 연산됨

->모든 페어를 탐색하지 않는 경우가 발생한 불안정성 해소

2) target network

loss = |Q_answer - Q_expect|

Q_answer는 target network이 계산, Q_expect는 main_network이 계산함

loss가 작아지는 방향 (Q가 수렴)으로 main_network의 파라미터 업데이트

target network는 주기적으로 main_network 복붙

"""

import os
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

# 현재 state를 입력 받아 action 별 Q-value 반환 모델
class DQN(tf.keras.Model): #tensorflow keras model 상속
  def __init__(self,action_size): # 인공 신경망 생성
    super(DQN, self).__init__()
    self.fc1 = Dense(24,activation='relu')  # hidden 1 : 16개 노드 ,활성함수는 relu
    self.fc2 = Dense(24,activation='relu')  # hidden 2 : 16개 노드 , 활성함수는 relu
    self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3)) # output : action 별 q-value 반환으로 노드개수는 action size

  def call(self,x): #인공 신경망에 state를 input으로 주고 q값 return
    x = self.fc1(x)  # hidden1 층에 x를 인풋으로 줌, 인풋 layer size는 x가 결정
    x = self.fc2(x)  # hidden1의 결과를 hidden2 층에 인풋으로 줌
    q = self.fc_out(x) # hidden 2의 결과를 output 층의 인풋으로 줌
    return q

# 시뮬레이션 환경 행동주체 생성(카트)
class DQNAgent:
  def __init__(self, state_size, action_size):
    self.render = False

    #상태와 행동 크기
    self.state_size = state_size
    self.action_size = action_size

    #Q-learning 파라미터
    self.discount_factor = 0.99
    self.learning_rate = 0.001
    self.epsilon = 1.0
    self.epsilon_decay = 0.999
    self.epsilon_min =0.01
    self.batch_size = 64
    self.train_start = 1000

    #replay 메모리 최대 2000개 페어(상태,행동,보상,다음상태) 저장
    self.memory = deque(maxlen=2000)

    #main 모델과(예측값) target모델(정답값) 생성
    self.model = DQN(action_size)
    self.target_model = DQN(action_size)
    self.optimizer = Adam(learning_rate=self.learning_rate)

    #target 모델 초기화
    self.update_target_model()

  # target 모델에 main 모델 복붙 (학습 진행 시 주기적으로)
  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  # 확률적으로 action 선택(탐험)
  def get_action(self,state):
    if np.random.rand() <= self.epsilon:  # 랜덤 선택
      return random.randrange(self.action_size)
    else: # 지금까지 학습 기반으로 보상이 가장 컸던 방향으로 선택
      q_value = self.model(state)
      return np.argmax(q_value[0]) # ?

  #리플레이 메모리에 경험(상태,행동,보상,다음상태,종료?) 저장
  def append_sample(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  #모델 학습
  def train_model(self):
    # 학습 진행 할 수록 탐험 확률 감소
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

    #리플레이 메모리에서 무작위로 학습데이터 추출
    mini_batch = random.sample(self.memory, self.batch_size)
    # 각 라벨별로 배열에 저장
    states = np.array([sample[0][0] for sample in mini_batch])
    actions = np.array([sample[1] for sample in mini_batch])
    rewards = np.array([sample[2] for sample in mini_batch])
    next_states = np.array([sample[3][0] for sample in mini_batch])
    dones = np.array([sample[4] for sample in mini_batch])


    #학습
    #모델에 input값 넣을 때 샘플 리스트 통째로 넣어도 됨 결과도 샘플 인덱스에 맞게 여러개 나옴
    #input[[1,1,1,1] [2,2,2,2]] 이면 모델 실제 input size는 4, 샘플 개수는 2
    # output [[10,2] [20,3]] 인풋 샘플 개수에 맞춰 나옴 output_size는 모델에서 미리 정의함
    with tf.GradientTape() as tape:               #자동 미분 API 딥러닝 학습은 loss를 최소로 하는 파라미터를 미분으로 찾음
      # main 모델이 예측하는 q 값
      predicts = self.model(states)                #리플레이 메모리에서 추출한 상태정보들을 모델에 넣음
      one_hot_action = tf.one_hot(actions, self.action_size) #실제로 한 행동에 1로 설정 ex) 왼쪽움직임[1,0] 오른쪽움직임[0,1]
      predicts = tf.reduce_sum(one_hot_action * predicts, axis=1) #선택한 행동의 q value만 추출 , model은 output이 2라 둘다 나옴

      # target 모델이 내놓은 정답 q값
      target_predicts = self.target_model(next_states)
      target_predicts = tf.stop_gradient(target_predicts) #타겟 모델 학습 방지

      #벨만 포드로 loss 값 계산
      max_q = np.amax(target_predicts, axis=-1) # 다음 상태에서 제일 큰 q값
      targets = rewards + (1 - dones) * self.discount_factor * max_q  # 현재 상태 최대 q값 = 현재 보상에 할인율을 곱한 다음상태 최대 q값
      loss = tf.reduce_mean(tf.square(targets - predicts)) # 현재상태에 대해 target 모델의 정답 q값과 main모델의 예측 q값 차이 계산

    # loss를 줄이는 방향(예측이 정답으로 수렴하는 방향)으로 모델 파라미터 업데이트
    model_params = self.model.trainable_weights #모델이 학습 할 파라미터(노드별 가중치와 편향 값) 리스트로 저장
    grads = tape.gradient(loss, model_params) #미분이용(극소값에서 미분하면 0이라 loss의 미분값을 0으로 하는 파라미터 찾기)
    self.optimizer.apply_gradients(zip(grads, model_params))
    
if __name__ == "__main__":
  # cart pole 시뮬레이션 환경 가져오기
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]  # (카트 위치, 카트 속도, 막대 각도, 막대 각속도)
  action_size = env.action_space.n # 왼쪽이동(0) 오른쪽이동(1)

  #카트 가져오기
  agent = DQNAgent(state_size, action_size)

  scores, episodes = [], [] #episode는 카트끌기 시작부터 막대가 쓰러질 때까지의 과정단위(안 쓰러질 수 도 있어서 최대 스텝 제한 있음)
  score_avg = 0
  num_episode = 300

  for e in range(num_episode):
    done = False  # 막대가 떨어지면 True
    score = 0     # 매 에피소드마다 점수

    state = env.reset() # 카트와 막대 초기화 하고 상태 정보 저장
    state = np.reshape(state,[1,state_size]) #[[카트위치,카트속도,막대각도,막대각속도]] state[0,0]=카트위치

    #막대기 떨어질 때까지 반복
    while not done :
      if agent.render:
        env.render()

      #현재 상태에서 행동 선택(탐험)
      action = agent.get_action(state)
      #행동 실행 후 다음 상태 ([카트위치,카트속도,막대각도,막대각속도],행동에 대한 보상, 종료여부, - )
      next_state , reward , done, info = env.step(action)
      next_state = np.reshape(next_state, [1, state_size])

      #현재 에피소드 점수 업데이트 (reqard는 안 떨어지면 0.1 떨어지면 -1)
      score += reward
      reward = 0.1 if not done or score == 500 else -1

      #경험을 리플레이 메모리에 저장
      agent.append_sample(state,action,reward,next_state,done)

      #리플레이 메모리가 기준치 이상 차면 매 step마다 학습
      if len(agent.memory) >= agent.train_start:
                agent.train_model()


      #다음 상태로 넘어가서 경험
      state = next_state

      #막대가 떨어졌으면 (에피소드 종료 시점)
      if done:
        # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
        agent.update_target_model()

        # 에피소드마다 학습 결과 출력 (점수가 높을 수록 막대기를 오랫동안 유지)
        score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
        print("episode: {:3d} | score avg: {:3.2f} | memory length: {:4d} | epsilon: {:.4f}".format(
               e, score_avg, len(agent.memory), agent.epsilon))

        # 에피소드마다 학습 결과 그래프로 저장
        scores.append(score_avg)
        episodes.append(e)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("episode")
        pylab.ylabel("average score")
        pylab.savefig("./save_graph/graph.png")

        # 카트가 평균 200이상 획득 하게되면 학습종료
        if score_avg > 200:
          agent.model.save("./save_model/model.h5")
          sys.exit()