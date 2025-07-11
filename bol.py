import os
import pandas as pd
import numpy as np
from collections import deque
import random
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
EPISODES = 10
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
WINDOW_SIZE = 10  # Number of days for state

# Data directory
DATA_DIR = './주식csv'


# Helper functions for technical indicators
def calculate_indicators(df):
    df = df.copy()
    df['MA20'] = df['종가'].rolling(window=20).mean()
    df['STD20'] = df['종가'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + 2 * df['STD20']
    df['Lower'] = df['MA20'] - 2 * df['STD20']
    df['RSI'] = compute_rsi(df['종가'])
    df = df.dropna()
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, action_size)
        self.last_loss = None  # 마지막 loss 저장용

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)

        states = torch.FloatTensor([m[0] for m in minibatch]).to(device)
        actions = torch.LongTensor([m[1] for m in minibatch]).to(device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(device)

        with torch.no_grad():
            target_next = self.model(next_states).max(1)[0]

        reward_norm = rewards / 1000.0
        targets = reward_norm + GAMMA * target_next

        outputs = self.model(states)
        target_vals = outputs.clone()
        target_vals[range(BATCH_SIZE), actions] = targets

        loss = self.criterion(outputs, target_vals)
        self.last_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY


# Environment interaction
def get_state(df, t):
    window = df.iloc[t - WINDOW_SIZE:t]
    state = window[['종가', '시가', '고가', '저가', '거래량', 'Upper', 'Lower', 'RSI']].values
    return state.flatten()


def get_reward(df, t, action):
    price_now = df.iloc[t]['종가']
    price_next = df.iloc[t + 1]['종가']

    # 지표 값 가져오기
    rsi = df.iloc[t]['RSI']
    upper = df.iloc[t]['Upper']
    lower = df.iloc[t]['Lower']
    close = price_now
    volume = df.iloc[t]['거래량']

    reward = 0

    # 매수 (action=1) - RSI 30 이하 과매도 구간, 볼린저 밴드 하단 터치 조건일 때 상승 시 보상
    if action == 1:
        if rsi < 30 and close <= lower:  # 과매도 + 밴드 하단 터치 판단
            if price_next > price_now:  # 다음날 상승
                reward = (price_next - price_now) * 10  # 보상 가중치
            else:
                reward = (price_next - price_now)  # 손실은 그대로 반영

    # 매도 (action=2) - RSI 70 이상 과매수 구간, 볼린저 밴드 상단 터치 조건일 때 하락 시 보상
    elif action == 2:
        if rsi > 70 and close >= upper:  # 과매수 + 밴드 상단 터치 판단
            if price_next < price_now:  # 다음날 하락
                reward = (price_now - price_next) * 10  # 보상 가중치
            else:
                reward = (price_now - price_next)  # 손실은 그대로 반영

    # 관망(0)일 때는 보상 0으로 유지
    else:
        reward = 0

    # 거래량도 반영하고 싶으면 추가 가능 (예: 거래량 일정 이상일 때만 보상)
    if volume < df['거래량'].rolling(window=5).mean().iloc[t]:  
        reward *= 0.5  # 거래량이 적으면 보상 절반으로 감소

    return reward



# train_all_stocks 함수 일부 수정
def train_all_stocks():
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    example_df = pd.read_csv(os.path.join(DATA_DIR, all_files[0]), encoding='cp949')
    example_df = calculate_indicators(example_df)
    state_size = WINDOW_SIZE * 8  # 8 features
    action_size = 3  # hold, buy, sell
    agent = DQNAgent(state_size, action_size)

    for episode in range(EPISODES):
        print(f"\n[에피소드 {episode+1}/{EPISODES}]")
        for file_name in all_files:
            print(f"현재 학습 중인 종목: {file_name}")  # **현재 분석 중인 종목 출력**
            file_path = os.path.join(DATA_DIR, file_name)
            try:
                df = pd.read_csv(file_path, encoding='cp949')
                df = calculate_indicators(df)
            except Exception as e:
                print(f"파일 읽기 실패: {file_name}, 이유: {e}")
                continue
            if len(df) < WINDOW_SIZE + 2:
                continue

            for t in range(WINDOW_SIZE, len(df) - 1):
                state = get_state(df, t)
                action = agent.act(state)
                reward = get_reward(df, t, action)
                next_state = get_state(df, t + 1)
                agent.remember(state, action, reward, next_state)
                agent.replay()

    # Save model
    torch.save(agent.model.state_dict(), "stock_rl_model.pth")
    print("\n[학습 완료 및 모델 저장됨]")


train_all_stocks()

