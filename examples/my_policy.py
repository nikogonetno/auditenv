import json
import sys

def send(action):
    print(json.dumps(action), flush=True)     # flush обязателен: без него раннер ждёт до лимита времени

first = json.loads(sys.stdin.readline())       # после reset: {observation, info}
episode_id = first["observation"]["manifest"]["episode_id"]
send({"action": "list_records", "table": "ledger", "limit": 2})
reply = json.loads(sys.stdin.readline())       # после шага: {observation, reward, terminated, truncated, info}
send({"action": "submit", "findings": {"episode_id": episode_id, "findings": []}})
final = json.loads(sys.stdin.readline())       # ответ на submit, эпизод окончен
