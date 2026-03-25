import httpx
import asyncio
import json

async def main():
    print("========================================")
    print("MarketLens Intelligence - Interactive CLI")
    print("========================================")
    
    prompt = input("Describe your startup idea or product:\n> ")
    
    if not prompt.strip():
        print("Empty prompt. Exiting.")
        return

    print("\n[+] Triggering asynchronous agent pipeline... (please wait)")
    
    try:
        # Send payload to the local FastAPI server
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "http://localhost:8000/api/v1/analyze", 
                json={"user_prompt": prompt.strip()}, 
                timeout=60.0
            )
            
            if res.status_code == 200:
                print("\n[+] Pipeline Success! Extracted TOON Data:\n")
                # Prettify the output dict
                print(json.dumps(res.json(), indent=2))
            else:
                print(f"\n[-] API Error ({res.status_code}): {res.text}")
                
    except Exception as e:
        print(f"\n[-] Network or Execution Error: {e}")

if __name__ == "__main__":
    # Ensure local server is running using `python main.py` in another terminal first
    asyncio.run(main())
