import time
import httpx
import asyncio
import logging


logger = logging.getLogger(__name__)


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

        logger.info(f"\n{len(responses)} requests in {elapsed:.2f}s\n")

        for i, resp in enumerate(responses):
            logger.info(f"--- Response {i} ---")
            for line in resp.text.strip().split("\n"):
                if line:
                    logger.info(line)


if __name__ == "__main__":
    asyncio.run(test_concurrent())
