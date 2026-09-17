"""Locust concurrent load testing suite for FT-Bench gateway."""
import random
from locust import HttpUser, task, between

SAMPLE_UTTERANCES = [
    "I need to cancel my order #ORD-12345 placed yesterday.",
    "Where is my refund? I returned the product 12 days ago.",
    "I cannot log in - my password reset email never arrived.",
    "Please change my delivery address to 99 Maple Lane, Boston.",
    "You charged me twice for order #98271! Fix this immediately!",
    "When will my package arrive? Tracking says in transit for 5 days.",
    "How do I delete my account and all my data?",
    "I love the new dashboard design - much faster than before!",
]


class LLMGatewayUser(HttpUser):
    wait_time = between(0.05, 0.3)

    @task(4)
    def generate_awq(self):
        payload = {
            "utterance": random.choice(SAMPLE_UTTERANCES),
            "system": "awq",
            "temperature": 0.1,
            "max_new_tokens": 256,
        }
        with self.client.post("/generate", json=payload, catch_response=True) as resp:
            if resp.status_code == 200 and resp.json().get("is_json_valid"):
                resp.success()
            else:
                resp.failure(f"status={resp.status_code}")

    @task(1)
    def generate_base(self):
        payload = {
            "utterance": random.choice(SAMPLE_UTTERANCES),
            "system": "base",
        }
        self.client.post("/generate", json=payload)
