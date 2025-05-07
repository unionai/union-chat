import time
import httpx


for endpoint in [
    "https://jolly-frog-f6dd5.apps.demo.hosted.unionai.cloud",
    "https://spring-fog-b69bd.apps.demo.hosted.unionai.cloud",
    "https://winter-smoke-eea4c.apps.demo.hosted.unionai.cloud",
]:
    start = time.time()
    response = httpx.get(f"{endpoint}/health", headers={"Authorization": "Bearer ABC"}, timeout=600)
    end = time.time()
    print(f"{endpoint} took {end - start} seconds")
    print(response.status_code)
