# import asyncio
# import aiohttp
# import time


# async def test_concurrent_requests(n=10):
#     async with aiohttp.ClientSession() as session:
#         tasks = [
#             session.get(f"http://localhost:8000/ask?query=test{i}&k=5")
#             for i in range(n)
#         ]
#         start = time.time()
#         responses = await asyncio.gather(*tasks)
#         elapsed = time.time() - start
#         print(f"{n} requests in {elapsed:.2f}s")
#         return responses

# asyncio.run(test_concurrent_requests(10))

import asyncio
import httpx
import time

async def test_concurrent():
    async with httpx.AsyncClient() as client:
        queries = [f"test{i}" for i in range(10)]
        
        start = time.time()
        tasks = [
            client.get(f"http://localhost:8000/ask?query={q}&k=3", timeout=30.0)
            for q in queries
        ]
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        print(f"\n✓ {len(responses)} requests in {elapsed:.2f}s\n")
        
        for i, resp in enumerate(responses):
            print(f"--- Response {i} ---")
            # Streaming responses come as newline-delimited JSON
            for line in resp.text.strip().split('\n'):
                if line:
                    print(line)
            print()

if __name__ == "__main__":
    asyncio.run(test_concurrent())
