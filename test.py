import httpx
import asyncio
import json

async def main():
    print("Testing API endpoint at http://localhost:8000/api/v1/analyze...")
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "http://localhost:8000/api/v1/analyze", 
                json={"user_prompt": "We are a startup building a collaborative workspace tailored for writing and task management, similar to Notion or Obsidian."}, 
                timeout=120.0
            )
            print("Status Code:", res.status_code)
            print("Response JSON:")
            print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"Test Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
