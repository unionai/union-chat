
import httpx
import asyncio
import time


endpoints = [
    "https://jolly-frog-f6dd5.apps.demo.hosted.unionai.cloud",
    "https://spring-fog-b69bd.apps.demo.hosted.unionai.cloud",
    "https://winter-smoke-eea4c.apps.demo.hosted.unionai.cloud",
]

async def make_async_request(url):
    print(f"Request started on {url}")
    async with httpx.AsyncClient() as client:
        # This sends the request without waiting for the response
        start = time.time()
        response = await client.get(
            url,
            timeout=600,
            headers={"Authorization": "Bearer ABC"},
        )
        print(f"Request finished on {url} in {time.time() - start} seconds")
        return response


async def other_work():
    print("Other work started")
    await asyncio.sleep(10)
    print("Other work finished")


# Create a task that will run in the background
async def main():
    # This schedules the request without waiting for it to complete
    tasks = []
    for endpoint in endpoints:
        task = asyncio.create_task(make_async_request(endpoint))
    
    # Do other work here while the request is in progress
    print(f"Request started on {endpoint}, continuing with other work...")
    await other_work()
    
    # If you eventually need the result:
    response = await task
    print(response.text)

# Run the async function
asyncio.run(main())
