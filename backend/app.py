from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import pickle
from collections import deque

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
input_dim = 4  # Balance, Transaction Amount, Transaction Type, User Feedback
output_dim = 3  # Actions: Decrease, Keep, Increase savings percentage
ACTIONS = [-0.01, 0, 0.01]
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_size = 10000

# Experience replay buffer
memory = deque(maxlen=memory_size)

# Initialize database
def init_db():
    conn = sqlite3.connect("savings.db")
    c = conn.cursor()

    # User balance table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    balance REAL DEFAULT 5000,
                    savings REAL DEFAULT 0
                )''')

    # User transaction history
    c.execute('''CREATE TABLE IF NOT EXISTS savings_trends (
                    user_id INTEGER,
                    transaction_number INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    category TEXT,
                    balance REAL,
                    savings_amount REAL,
                    savings_percentage REAL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )''')

    # Store Q-learning models for users
    c.execute('''CREATE TABLE IF NOT EXISTS user_dqn_models (
                    user_id INTEGER PRIMARY KEY,
                    model BLOB,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )''')
    
    conn.commit()
    conn.close()

init_db()

# Save and load Q-learning models per user
def save_dqn_model(user_id, model):
    conn = sqlite3.connect("savings.db")
    c = conn.cursor()
    model_data = pickle.dumps(model)
    c.execute("INSERT OR REPLACE INTO user_dqn_models (user_id, model) VALUES (?, ?)", (user_id, model_data))
    conn.commit()
    conn.close()

def load_dqn_model(user_id):
    conn = sqlite3.connect("savings.db")
    c = conn.cursor()
    c.execute("SELECT model FROM user_dqn_models WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return pickle.loads(result[0])  # Deserialize model
    else:
        return DQN(input_dim, output_dim)  # Return a new model if user is new

# Optimize the Q-learning model
def optimize_model(model, target_model, optimizer):
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)

    q_values = model(state_batch).gather(1, action_batch).squeeze()
    
    with torch.no_grad():
        max_next_q_values = target_model(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# API to set user balance
@app.route("/set_balance", methods=["POST"])
def set_balance():
    data = request.json
    user_id = data.get("user_id", 1)
    balance = float(data["balance"])

    conn = sqlite3.connect("savings.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users (user_id, balance) VALUES (?, ?)", (user_id, balance))
    conn.commit()
    conn.close()

    return jsonify({"message": "Balance updated", "balance": balance})

# API to process transactions
@app.route("/transaction", methods=["POST"])
def api_process_transaction():
    data = request.json
    user_id = data.get("user_id", 1)
    amount = float(data["amount"])
    category = data["type"]
    user_feedback = data["feedback"]

    conn = sqlite3.connect("savings.db")
    c = conn.cursor()
    c.execute("SELECT balance, savings FROM users WHERE user_id = ?", (user_id,))
    user_data = c.fetchone()
    conn.close()

    if not user_data:
        balance, savings = 5000, 0
    else:
        balance, savings = user_data

    model = load_dqn_model(user_id)
    target_model = DQN(input_dim, output_dim)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    normalized_balance = balance / 10000.0
    normalized_amount = amount / 1000.0
    transaction_type = 1 if category == "luxurious" else 0
    feedback_state = 1 if user_feedback else 0

    state = np.array([normalized_balance, normalized_amount, transaction_type, feedback_state])

    action_index = random.choice(range(len(ACTIONS)))
    savings_adjustment = ACTIONS[action_index]

    new_savings_percentage = max(0.01, min(0.20, savings_adjustment + 0.05))
    savings_amount = round(amount * new_savings_percentage, 2)
    new_total = amount + savings_amount
    new_balance = balance - new_total
    new_savings = savings + savings_amount

    memory.append((state, action_index, user_feedback, state, 0))
    optimize_model(model, target_model, optimizer)
    save_dqn_model(user_id, model)

    conn = sqlite3.connect("savings.db")
    c = conn.cursor()
    c.execute("UPDATE users SET balance = ?, savings = ? WHERE user_id = ?", (new_balance, new_savings, user_id))
    c.execute("INSERT INTO savings_trends (user_id, category, balance, savings_amount, savings_percentage) VALUES (?, ?, ?, ?, ?)",
              (user_id, category, new_balance, savings_amount, new_savings_percentage))
    conn.commit()
    conn.close()

    return jsonify({
        "savings_suggestion": savings_amount,
        "new_total": new_total,
        "new_balance": new_balance,
        "new_savings": new_savings
    })

# API to retrieve savings trends
@app.route("/savings_trend", methods=["GET"])
def get_savings_trend():
    user_id = request.args.get("user_id", 1)
    conn = sqlite3.connect("savings.db")
    df = pd.read_sql_query("SELECT transaction_number, balance, savings_amount, savings_percentage FROM savings_trends WHERE user_id = ? ORDER BY transaction_number", conn, params=(user_id,))
    conn.close()
    return df.to_json(orient="records")

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
