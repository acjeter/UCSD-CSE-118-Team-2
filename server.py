import socket
import threading
import json
import time
import sys
import os
import torch
import numpy as np

# Add 'ai' folder to path so we can import model and utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))

from model import ASLModel
from openai import OpenAI
from utils import normalize_landmarks

from flask import Flask, Response, render_template_string, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Shared state
current_letter = "?"
letters = []  # list of all letters received so far
history = "" # committed sentences
last_update_time = time.time()
is_dirty = False # True if we have uncorrected letters

# ---------- MODEL LOADING ----------
try:
    print("Loading AI Model...")
    classes = np.load("ai/label_classes.npy")
    model = ASLModel()
    model.load_state_dict(torch.load("ai/asl_model.pth", map_location='cpu'))
    model.eval()
    print("Model Loaded Successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# ---------- TCP SERVER (HoloLens -> Laptop) ----------

def tcp_server(host="0.0.0.0", port=5005):
    """Listens for incoming TCP connections from HoloLens and updates letters."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"[TCP] Listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            print(f"[TCP] Connection from {addr}")
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


def handle_client(conn, addr):
    global current_letter, letters, last_update_time, is_dirty
    buffer = b""
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                print(f"[TCP] Client {addr} disconnected")
                break

            buffer += data
            # Process newline-delimited JSON
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    payload = json.loads(line.decode("utf-8"))
                    
                    # Case 1: Raw Joints from HoloLens (Run Inference)
                    if "joints" in payload and model is not None:
                        raw_data = payload["joints"]
                        # Expecting list of 66 floats (22 joints * 3)
                        if isinstance(raw_data, list) and len(raw_data) == 66:
                            # Preprocess
                            coords = np.array(raw_data, dtype=np.float32).reshape(-1, 3)
                            coords = normalize_landmarks(coords)
                            
                            # Inference
                            with torch.no_grad():
                                x = torch.tensor(coords.flatten(), dtype=torch.float32).unsqueeze(0)
                                output = model(x)
                                pred_id = torch.argmax(output, dim=1).item()
                                letter = str(classes[pred_id])
                                
                                # Update State
                                current_letter = letter
                                letters.append(letter)
                                last_update_time = time.time()
                                is_dirty = True
                                print(f"[AI] Detected: {letter}")
                        else:
                            print(f"[TCP] received joints but wrong shape: {len(raw_data)}")

                    # Case 2: Letter sent directly (Legacy/Fallback)
                    elif "letter" in payload:
                        letter = payload.get("letter")
                        if letter:
                            # Take only the first character, uppercased
                            letter = str(letter)[0].upper()
                            current_letter = letter
                            letters.append(letter)
                            last_update_time = time.time()
                            is_dirty = True
                            print(f"[TCP] New letter: {letter} | Translation: {''.join(letters)}")
                            
                except Exception as e:
                    print(f"[TCP] Error parsing message: {e}")


# ---------- HTTP SERVER (Laptop -> Phone) ----------

HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>ASL Letter Viewer</title>
    <style>
      body {
        font-family: system-ui, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        margin: 0;
        padding: 1rem;
        box-sizing: border-box;
      }
      #letter {
        font-size: 6rem;
        margin-bottom: 0.5rem;
      }
      #translation {
        font-size: 1.5rem;
        margin-top: 0.5rem;
        word-wrap: break-word;
        max-width: 90vw;
        text-align: center;
      }
      #history {
        color: #888;
        font-weight: normal;
      }
      #current {
        font-weight: bold;
        color: #000;
      }
      #status {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-top: 0.25rem;
      }
      .btn-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
        justify-content: center;
      }
      button {
        font-size: 1rem;
        padding: 0.8rem 1.2rem;
      }
      h2 {
        margin: 0.5rem 0 0.25rem;
      }
    </style>
  </head>
  <body>
    <h2>Current letter</h2>
    <div id="letter">?</div>

    <h2>Translation so far</h2>
    <div id="translation">
      <span id="history"></span><span id="current"></span>
    </div>

    <div id="status">Connecting...</div>

    <div class="btn-row">
      <button id="toggleAudio">Audio: ON</button>
      <button id="clearBtn">Clear</button>
      <button id="testAudio">Test Audio</button>
      <button id="speakTranslation">Speak Translation</button>
      <button id="correctBtn">Correct Sentence</button>
    </div>

    <script>
      const letterSpan = document.getElementById("letter");
      const historySpan = document.getElementById("history");
      const currentSpan = document.getElementById("current");
      const statusSpan = document.getElementById("status");
      const toggleBtn = document.getElementById("toggleAudio");
      const clearBtn = document.getElementById("clearBtn");
      const testBtn = document.getElementById("testAudio");
      const speakBtn = document.getElementById("speakTranslation");
      const correctBtn = document.getElementById("correctBtn");

      let audioEnabled = true;
      let synth = window.speechSynthesis || null;

      // Helper: show if TTS is unavailable
      if (!synth) {
        statusSpan.textContent = "Connected (no speechSynthesis support on this browser)";
      }

      function speak(text) {
        if (!synth) return;
        if (!text) return;
        const utter = new SpeechSynthesisUtterance(text);
        // Optional: pick a default voice if needed
        // utter.voice = synth.getVoices()[0] || null;

        // Cancel any ongoing speech and speak the new one
        try {
          synth.cancel();
        } catch (e) {
          console.error("Error cancelling speech:", e);
        }
        try {
          synth.speak(utter);
        } catch (e) {
          console.error("Error speaking:", e);
        }
      }

      toggleBtn.onclick = () => {
        audioEnabled = !audioEnabled;
        toggleBtn.textContent = audioEnabled ? "Audio: ON" : "Audio: OFF";
      };

      clearBtn.onclick = async () => {
        try {
          await fetch("/clear", { method: "POST" });
          // Reset UI immediately
          letterSpan.textContent = "?";
          historySpan.textContent = "";
          currentSpan.textContent = "";
        } catch (e) {
          console.error("Failed to clear:", e);
        }
      };

      // Speak the full translation on demand
      speakBtn.onclick = () => {
        if (!audioEnabled) return;
        const raw = historySpan.textContent + " " + currentSpan.textContent;
        const text = raw.trim().toLowerCase();
        if (text) {
          speak(text);
        }
      };

      // Explicit "user gesture" to unlock audio on iOS
      testBtn.onclick = () => {
        speak("Audio test");
      };

      correctBtn.onclick = async () => {
        statusSpan.textContent = "Correcting...";
        try {
          await fetch("/correct", { method: "POST" });
        } catch (e) {
          console.error("Failed to correct:", e);
          statusSpan.textContent = "Correction failed";
        }
      };

      const es = new EventSource("/stream");

      es.onopen = () => {
        statusSpan.textContent = "Connected";
      };

      es.onerror = () => {
        statusSpan.textContent = "Disconnected, retrying...";
      };

      es.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const letter = payload.letter || "?";
          const translation = payload.translation || "";
          const history = payload.history || "";

          letterSpan.textContent = letter;
          currentSpan.textContent = translation;
          historySpan.textContent = history;

        } catch (e) {
          console.error("Error parsing SSE payload:", e);
        }
      };
    </script>
  </body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/stream")
def stream():
    def event_stream():
        last_letter = None
        last_translation = None

        while True:
            global current_letter, letters, history
            translation = "".join(letters)

            # We trigger update if anything changed
            payload = {
                "letter": current_letter,
                "translation": translation,
                "history": history,
            }
            # IMPORTANT: real newline characters here
            yield "data: " + json.dumps(payload) + "\n\n"

            time.sleep(0.1)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/clear", methods=["POST"])
def clear():
    global current_letter, letters, history
    current_letter = "?"
    letters = []
    history = ""
    print("[HTTP] Translation cleared by client")
    return ("", 204)  # No Content


def perform_correction():
    """Helper function to run the OpenAI correction"""
    global letters, is_dirty, history
    
    if not letters:
        return
    
    # Optional: Don't correct if it's already clean/empty (handled by is_dirty check in auto-loop, but good for safety)
    
    raw_text = "".join(letters)
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("[AI] No OpenAI API Key found.")
        return
    
    try:
        print(f"[AI] Sending to OpenAI (Auto/Manual): {raw_text}")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects sentences generated from ASL sign language recognition model. The input may have improper spacing, missing letters, or wrong characters. Try to infer what the user might be trying to say and correct it. Output ONLY the corrected sentence."},
                {"role": "user", "content": raw_text}
            ]
        )
        cleaned = response.choices[0].message.content.strip()
        print(f"[AI] Corrected '{raw_text}' to '{cleaned}'")
        
        # Update history with the cleaned version and clear buffer
        history += cleaned + " "
        letters.clear() # Clear the buffer
        is_dirty = False # Reset dirty flag
        
    except Exception as e:
        print(f"[AI] OpenAI Error: {e}")

def auto_correct_loop():
    """Background thread to check for inactivity"""
    global last_update_time, is_dirty
    print("[AutoCorrect] Started background monitor (5s timeout)")
    while True:
        time.sleep(1)
        # Check if 5 seconds passed since last update AND we have new data
        if is_dirty and (time.time() - last_update_time > 5):
            print("[AutoCorrect] Inactivity detected. Triggering correction...")
            perform_correction()

@app.route("/correct", methods=["POST"])
def correct():
    perform_correction()
    return ("", 204)


if __name__ == "__main__":
    # Start TCP server in background thread
    threading.Thread(target=tcp_server, daemon=True).start()
    
    # Start Auto-Correct background thread
    threading.Thread(target=auto_correct_loop, daemon=True).start()

    # Run Flask HTTP server
    print("[HTTP] Serving web UI on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)