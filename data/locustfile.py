from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests

    @task
    def index(self):
        self.client.get("/")