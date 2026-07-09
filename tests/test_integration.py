import unittest
import json
from flask import Flask
from api.routes import api_bp
from database.crud import DatabaseManager


class DummyHealthEvaluator:
    def __init__(self):
        self.tracked_persons = {"1": "DummyPerson"}


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(api_bp)

        # Mock dependencies in config
        self.app.config["HEALTH_EVALUATOR"] = DummyHealthEvaluator()
        self.app.config["DB_CONN"] = DatabaseManager()

        self.client = self.app.test_client()

    def test_system_health_endpoint(self):
        response = self.client.get("/api/system/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertEqual(data["status"], "healthy")
        self.assertIn("platform", data)
        self.assertEqual(data["tracked_targets"], 1)

    def test_history_endpoint_empty(self):
        response = self.client.get("/api/history/UnknownTestUser999")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, [])  # Should be empty list for unknown user


if __name__ == "__main__":
    unittest.main()
