"""
Locust load testing file for ft-bench vLLM servers.

Usage:
  locust -f serving/vllm/load_test.py --host http://localhost:8002 \
         --users 10 --spawn-rate 2 --run-time 60s --headless
"""

import random
import json
from locust import HttpUser, task, between


SAMPLE_PROMPTS = [
    "wake me up at seven am tomorrow",
    "what is the weather in london",
    "play some jazz music",
    "turn off the bedroom lights",
    "set a timer for five minutes",
    "tell me a joke",
    "what time is it in tokyo",
    "add milk to my shopping list",
    "call mom",
    "what movies are playing nearby",
]


class VLLMUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task
    def generate(self):
        prompt = random.choice(SAMPLE_PROMPTS)
        self.client.post(
            "/generate",
            json={"prompt": prompt, "max_tokens": 128, "temperature": 0.0},
            name="/generate",
        )

    @task(weight=1)
    def health_check(self):
        self.client.get("/health", name="/health")
