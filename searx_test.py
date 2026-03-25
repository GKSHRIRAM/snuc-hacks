import httpx

def test_searxng():
    urls = [
        "https://searx.be/search",
        "https://search.mdosch.de/search",
        "https://search.ononoki.org/search",
        "https://searx.tiekoetter.com/search",
        "https://searxng.site/search"
    ]
    for u in urls:
        try:
            r = httpx.get(f"{u}?q=notion+competitors&format=json", timeout=5.0)
            if r.status_code == 200:
                print(f"SUCCESS: {u}")
                break
            else:
                print(f"FAILED (Status {r.status_code}): {u}")
        except Exception as e:
            print(f"ERROR: {u} - {str(e)}")

if __name__ == "__main__":
    test_searxng()
