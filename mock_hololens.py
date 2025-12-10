import socket
import json
import time

def send_phrase(phrase):
    host = "127.0.0.1"
    port = 5005
    
    print(f"Connecting to {host}:{port}...")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            
            # Send each letter one by one
            for char in phrase:
                # payload = json.dumps({"letter": char}) + "\n"
                # The server expects "letter" in the JSON payload
                # We can also simulate "joints" if we wanted, but "letter" is the fallback path supported by server.py
                msg = {"letter": char}
                payload = json.dumps(msg) + "\n"
                s.sendall(payload.encode("utf-8"))
                print(f"Sent: {char}")
                time.sleep(0.2) # Fast typing
                
            print(f"Sent phrase: {phrase}")
            print("Waiting... (Auto-correct should trigger in 5s on the server)")
            
    except ConnectionRefusedError:
        print("Failed to connect. Is server.py running?")

if __name__ == "__main__":
    # Simulate a bad spelling of "BUS STOP HERE"
    target = "MYY NMEE IIIISS TONNMY I WUD LIKKEEEJJBF TGPIOW ORDEHR ACC COFFEOOMUUE"
    send_phrase(target)
