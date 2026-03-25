import os
import json
import queue
import threading
import asyncio
from flask import Flask, send_from_directory, jsonify, request, Response

# Import the collectors we built earlier
import collectors.wayback
import collectors.news
import collectors.reviews
import collectors.webcrawl

app = Flask(__name__, static_folder='frontend')
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_queue = queue.Queue()

def log(msg, status="running"):
    """Helper to push log messages to queue and stdout."""
    print(msg)
    log_queue.put(json.dumps({"status": status, "message": msg}))

@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/run', methods=['POST'])
def run_collection():
    data = request.json
    domain = data.get('domain', '')
    companies = data.get('companies', [])
    
    # Validate input
    if not domain or not isinstance(domain, str):
        return jsonify({'error': 'Domain is required and must be a string'}), 400
    if not companies or not isinstance(companies, list) or len(companies) > 10 or len(companies) < 1:
        return jsonify({'error': 'Companies must be a list of 1 to 10 items'}), 400
        
    # Start pipeline in background
    threading.Thread(target=run_pipeline, args=(domain, companies)).start()
    
    return jsonify({'status': 'started', 'message': f'Collection started for {domain}'})

def run_pipeline(domain, companies):
    """Background thread to run collectors sequentially."""
    
    # Clear queue
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break
            
    async def pipeline():
        log("Starting reviews collection...")
        try:
            r_res = await collectors.reviews.collect_reviews(domain)
            with open(os.path.join(OUTPUT_DIR, 'reviews.json'), 'w') as f:
                json.dump(r_res, f)
        except Exception as e:
            log(f"Reviews collection error: {e}")

        log("Starting webcrawl...")
        try:
            c_res = await collectors.webcrawl.collect_webcrawl(domain)
            with open(os.path.join(OUTPUT_DIR, 'webcrawl.json'), 'w') as f:
                json.dump(c_res, f)
        except Exception as e:
            log(f"Webcrawl error: {e}")

        log("Starting Wayback snapshots...")
        try:
            w_res = await collectors.wayback.collect_wayback(domain)
            with open(os.path.join(OUTPUT_DIR, 'snapshots.json'), 'w') as f:
                json.dump(w_res, f)
        except Exception as e:
            log(f"Wayback error: {e}")

        log("Starting news collection...")
        try:
            n_res = await collectors.news.collect_news(domain)
            with open(os.path.join(OUTPUT_DIR, 'news.json'), 'w') as f:
                json.dump(n_res, f)
        except Exception as e:
            log(f"News error: {e}")
            
        log("All done. Files saved to outputs/", status="complete")
        
    asyncio.run(pipeline())

@app.route('/api/stream')
def stream():
    def event_stream():
        while True:
            try:
                msg_json = log_queue.get(timeout=10)
                yield f"data: {msg_json}\n\n"
                
                # Check if this is the final message to close stream
                data = json.loads(msg_json)
                if data.get("status") == "complete":
                    break
            except queue.Empty:
                # Keep connection alive
                yield ": keepalive\n\n"
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/api/download/<filename>')
def download_file(filename):
    valid_files = ['reviews.json', 'snapshots.json', 'news.json', 'webcrawl.json']
    if filename not in valid_files:
        return jsonify({'error': 'Invalid file'}), 400
        
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    return jsonify({'error': f'{filename} not found. Has collection finished?'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
