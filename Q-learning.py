"""
[1] Q-learning

Q값 : 현재 상태에서 행동 실행 시 최종적으로 받는 보상 값 ( = 현재 행동 보상 + 미래 기대 보상)

Q 테이블 : 매 state 마다 각 행동에 대한 Q값 저장(초기값은 모두 0 학습을 통해 업데이트)

업데이트를 마친 Q 테이블을 기반으로 각 state에서 최종적으로 가장 많은 보상을 받을 것으로 기대되는 행동을 선택
"""

import numpy as np
import random
#상황
#가로3 세로 1 격자를 움직이는 로보트를 0에서 시작해서 2로 보내기
#매 상태마다 왼쪽과 오른쪽으로 움직일 수 있음

n_states = 3 # |0(시작)|1|2(종료)|
n_actions = 2 # 왼쪽:0(<-), 오른쪽:1(->)
goal_state = 2

max_steps = 50 # 한 에피소드(시작~종료까지)당 최대 step(무한루프 방지)
n_episodes = 1000

learning_rate = 0.1  # 학습률 a (업데이트 가중치)
gamma = 0.9          # 할인율 r (미래보상 가중치)
epsilon = 0.1        # 탐험율 e (action 선택 시)

Q = np.zeros((n_states, n_actions)) # Q-table[3][2] 초기화

def get_current_reward_and_next_state(state,action):

  if action == 0 : #왼쪽 이동
    nxt = state-1
  else :           # 오른쪽 이동
    nxt = state+1

  if nxt<0 :       # 벽에 부딪히면 -1
    rwd = -2
    nxt = 0
  elif nxt<2:      # 안 부딪히면 1
    rwd = 0
    nxt = nxt
  else:            # 종료하면 2
    rwd = 3
    nxt = 2

  return rwd,nxt

#학습(Q-table 업데이트)
for episode in range(n_episodes):
  state = 0  # 매 에피소드마다 시작 상태 초기화

  for step in range(max_steps):
    # 다음 행동 확률적 선택
    if random.uniform(0,1) < epsilon :# 랜덤함수 리턴값이 epsilon 보다 작으면
      action = random.randint(0, n_actions - 1) # 왼쪽 오른쪽 랜덤 선택
    else:
      action = np.argmax(Q[state]) # 지금까지 학습을 기반으로 누적보상이 컸던 행동 선택

    #현재 상태에서 행동 후 보상 값과 다음 상태 확인
    reward , next_state = get_current_reward_and_next_state(state,action)

    #Q-table 업데이트 (벨만 방정식)
    Q[state][action] = (1-learning_rate)*Q[state][action] + (learning_rate)*(reward + gamma*np.max(Q[next_state]))

    #다음 상태 이동
    state = next_state
    if state == goal_state:
      break


#학습 후 Q-table
print("Final Q-table:")
print(Q)

#Q-table에 기반해 각 상태에서 최적의 행동 선택
for state in range(n_states):
    action = np.argmax(Q[state])
    print(f"State {state}: Action {action}")