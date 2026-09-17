import os
import sys
import re
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time
import threading

BASE_URL = "https://www.gdacs.org/flooddetection/DATA/ALL/"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "data", "flooddet")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

os.makedirs(TARGET_DIR, exist_ok=True)

def fetch_directory(url):
    req = urllib.request.Request(url, headers=HEADERS)
    subdirs = []
    files = []
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            raw = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html, re.I)
            for href in raw:
                if '[To Parent Directory]' in href or href == '/' or '?' in href:
                    continue
                if href.rstrip('/') in ['/flooddetection/DATA', '/flooddetection']:
                    continue
                
                full_url = urllib.parse.urljoin(url, href)
                parsed = urllib.parse.urlparse(full_url)
                path = parsed.path
                if not path.startswith('/flooddetection/DATA/ALL/'):
                    continue
                    
                rel = path[len('/flooddetection/DATA/ALL/'):].strip('/')
                if full_url.endswith('/'):
                    if full_url.startswith(BASE_URL) and full_url != url:
                        subdirs.append(full_url)
                else:
                    if rel:
                        dest = os.path.join(TARGET_DIR, os.path.normpath(rel))
                        files.append((full_url, dest))
    except Exception as e:
        print(f"[Fetch Error] {url}: {e}", flush=True)
        
    return subdirs, files

def download_worker(file_url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return ("SKIPPED", dest_path)
        
    temp_path = dest_path + ".part"
    req = urllib.request.Request(file_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=40) as resp, open(temp_path, "wb") as out_f:
            chunk_size = 64 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        return ("DOWNLOADED", dest_path)
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return ("FAILED", f"{file_url}: {e}")

def main():
    print(f"=== GDACS FloodDET High-Throughput Downloader ===", flush=True)
    print(f"Source: {BASE_URL}", flush=True)
    print(f"Destination: {TARGET_DIR}", flush=True)
    start_time = time.time()
    
    dir_queue = queue.Queue()
    dir_queue.put(BASE_URL)
    visited_dirs = {BASE_URL}
    visited_lock = threading.Lock()
    
    file_queue = queue.Queue()
    seen_files = set()
    seen_lock = threading.Lock()
    
    scan_active = True
    active_scanners = 0
    scanner_lock = threading.Lock()
    
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "dirs_scanned": 0}
    stats_lock = threading.Lock()
    
    def scanner():
        nonlocal scan_active, active_scanners
        while True:
            try:
                current_dir = dir_queue.get(timeout=2)
            except queue.Empty:
                with scanner_lock:
                    if active_scanners == 0:
                        break
                time.sleep(0.1)
                continue
                
            with scanner_lock:
                active_scanners += 1
                
            subdirs, files = fetch_directory(current_dir)
            with stats_lock:
                stats["dirs_scanned"] += 1
                if stats["dirs_scanned"] % 50 == 0:
                    print(f"[*] Scanned {stats['dirs_scanned']} directories... Found {len(seen_files)} files.", flush=True)
            
            with visited_lock:
                for sd in subdirs:
                    if sd not in visited_dirs:
                        visited_dirs.add(sd)
                        dir_queue.put(sd)
                        
            with seen_lock:
                for u, d in files:
                    if d not in seen_files:
                        seen_files.add(d)
                        file_queue.put((u, d))
                        
            with scanner_lock:
                active_scanners -= 1
            dir_queue.task_done()
            
    def downloader():
        while True:
            try:
                item = file_queue.get(timeout=3)
            except queue.Empty:
                if not scan_active and file_queue.empty():
                    break
                time.sleep(0.2)
                continue
                
            status, info = download_worker(item[0], item[1])
            with stats_lock:
                if status == "DOWNLOADED":
                    stats["downloaded"] += 1
                    rel = os.path.relpath(info, TARGET_DIR)
                    print(f"[+] Downloaded: {rel}", flush=True)
                elif status == "SKIPPED":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1
                    print(f"[-] Failed: {info}", flush=True)
            file_queue.task_done()

    print("\nStarting concurrent scanner & downloader threads (24 crawler workers + 24 download workers)...", flush=True)
    
    # Launch scanners
    scanner_threads = [threading.Thread(target=scanner) for _ in range(24)]
    for t in scanner_threads:
        t.daemon = True
        t.start()
        
    # Launch downloaders
    downloader_threads = [threading.Thread(target=downloader) for _ in range(24)]
    for t in downloader_threads:
        t.daemon = True
        t.start()
        
    for t in scanner_threads:
        t.join()
        
    scan_active = False
    print(f"\n[OK] Directory scanning complete. Total directories: {stats['dirs_scanned']}. Total files: {len(seen_files)}", flush=True)
    
    for t in downloader_threads:
        t.join()
        
    duration = time.time() - start_time
    print(f"\n=======================================================", flush=True)
    print(f"GDACS FloodDET Download Process Finished in {duration:.1f}s!", flush=True)
    print(f"Results: {stats['downloaded']} Downloaded | {stats['skipped']} Skipped | {stats['failed']} Failed", flush=True)
    print(f"Saved into: {TARGET_DIR}", flush=True)
    print(f"=======================================================", flush=True)

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()

