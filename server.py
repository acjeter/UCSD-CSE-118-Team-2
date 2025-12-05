import socket
import threading
import json
import time

from flask import Flask, Response, render_template_string, request

app = Flask(__name__)

# Shared state
current_letter = "?"
letters = []  # list of all letters received so far


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
    global current_letter, letters
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
                    letter = payload.get("letter")
                    if letter:
                        # Take only the first character, uppercased
                        letter = str(letter)[0].upper()
                        current_letter = letter
                        letters.append(letter)
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
      #status {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-top: 0.25rem;
      }
      .btn-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
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
    <div id="translation"></div>

    <div id="status">Connecting...</div>

    <div class="btn-row">
      <button id="toggleAudio">Audio: ON</button>
      <button id="clearBtn">Clear</button>
    </div>

    <script>
      const letterSpan = document.getElementById("letter");
      const translationSpan = document.getElementById("translation");
      const statusSpan = document.getElementById("status");
      const toggleBtn = document.getElementById("toggleAudio");
      const clearBtn = document.getElementById("clearBtn");

      let audioEnabled = true;

      toggleBtn.onclick = () => {
        audioEnabled = !audioEnabled;
        toggleBtn.textContent = audioEnabled ? "Audio: ON" : "Audio: OFF";
      };

      clearBtn.onclick = async () => {
        try {
          await fetch("/clear", { method: "POST" });
          // Reset UI immediately
          letterSpan.textContent = "?";
          translationSpan.textContent = "";
        } catch (e) {
          console.error("Failed to clear:", e);
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

          letterSpan.textContent = letter;
          translationSpan.textContent = translation;

          if (audioEnabled && letter && letter !== "?") {
            const utterance = new SpeechSynthesisUtterance(letter);
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
          }
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
            global current_letter, letters
            translation = "".join(letters)

            if current_letter != last_letter or translation != last_translation:
                payload = {
                    "letter": current_letter,
                    "translation": translation,
                }
                # IMPORTANT: real newline characters here
                yield "data: " + json.dumps(payload) + "\n\n"
                last_letter = current_letter
                last_translation = translation

            time.sleep(0.1)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/clear", methods=["POST"])
def clear():
    global current_letter, letters
    current_letter = "?"
    letters = []
    print("[HTTP] Translation cleared by client")
    return ("", 204)  # No Content


if __name__ == "__main__":
    # Start TCP server in background thread
    threading.Thread(target=tcp_server, daemon=True).start()

    # Run Flask HTTP server
    print("[HTTP] Serving web UI on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)