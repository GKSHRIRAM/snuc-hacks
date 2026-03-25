import httpx
import asyncio
import json
import os
from datetime import datetime

async def main():
    print("=" * 50)
    print("  MarketLens BI Engine v4.0 — Interactive CLI")
    print("=" * 50)

    prompt = input("\nDescribe your startup idea or product:\n> ")

    if not prompt.strip():
        print("Empty prompt. Exiting.")
        return

    print("\n[+] Submitting to pipeline...")

    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Submit job
            res = await client.post(
                "http://localhost:8000/api/v1/analyze",
                json={"user_prompt": prompt.strip()},
                timeout=30.0
            )

            if res.status_code != 200:
                print(f"\n[-] Failed to start job: {res.text}")
                return

            job_id = res.json()["job_id"]
            print(f"[+] Job started: {job_id}")
            print("[+] Polling for results (this takes 2-5 minutes)...\n")

            # Step 2: Poll for completion
            max_wait = 300  # 5 minutes
            interval = 5
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(interval)
                elapsed += interval

                status_res = await client.get(
                    f"http://localhost:8000/api/v1/status/{job_id}",
                    timeout=10.0
                )

                if status_res.status_code != 200:
                    print(f"  Poll error: {status_res.status_code}")
                    continue

                data = status_res.json()
                status = data["status"]
                progress = data.get("progress", 0)

                print(f"  -> {status} ({progress}%)")

                if status == "COMPLETED":
                    result = data["result"]
                    print(f"\n{'='*50}")
                    print("  PIPELINE COMPLETE!")
                    print(f"{'='*50}\n")
                    print(json.dumps(result, indent=2))

                    # Auto-export via CLI
                    os.makedirs("data_exports", exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"data_exports/cli_export_{ts}.json"
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)
                    print(f"\n[+] Exported to: {path}")
                    return

                elif status == "FAILED":
                    print(f"\n[-] Pipeline failed: {data.get('error', 'Unknown error')}")
                    return

            print(f"\n[-] Timeout after {max_wait}s. Job may still be running.")
            print(f"    Check manually: GET http://localhost:8000/api/v1/status/{job_id}")

    except httpx.ConnectError:
        print("\n[-] Cannot connect to http://localhost:8000")
        print("    Make sure the server is running: python main.py")
    except Exception as e:
        print(f"\n[-] Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
