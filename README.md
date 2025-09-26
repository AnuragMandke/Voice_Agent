## Voice Agent

A browser-based, low-latency voice assistant. The client records microphone audio, the server transcribes it with Deepgram (prerecorded API), generates a reply with Groq, and streams TTS audio with Cartesia back to the browser. After TTS completes, the client automatically resumes listening for the next turn.

### Features
- Real-time recording in the browser using MediaRecorder
- Binary WebSocket frames for efficient audio upload
- Prerecorded transcription via Deepgram `nova-2`
- LLM response generation via Groq (configurable models)
- Streaming TTS via Cartesia (PCM f32 LE at 24 kHz)
- Client auto-blocks recording during TTS; resumes listening after TTS
- Graceful fallbacks when API keys are missing

### Architecture
- Frontend (`index.html`)
  - UI controls (Start/Stop/Clear), transcript view
  - Captures audio chunks every 100 ms; sends as binary WS frames
  - Plays TTS PCM audio streamed from server; resumes listening on `tts_complete`
- Backend (`server.py`)
  - Async WebSocket server with `websockets`
  - Buffers binary audio chunks while listening; on stop, sends one prerecorded transcription request to Deepgram
  - Uses Groq to generate assistant replies
  - Calls Cartesia TTS; streams audio back in 100 ms frames

### Requirements
- Python 3.10+ (Windows supported)
- A modern browser with MediaRecorder support (Chrome/Edge recommended)

Python dependencies (installed via `requirements.txt`):
- websockets, deepgram-sdk, webrtcvad, groq, requests

### Configuration
Create a `.env` file with:

```
DEEPGRAM_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
CARTESIA_API_KEY=your_cartesia_key
```

Notes:
- The server runs with partial keys: missing Deepgram will skip transcription; missing Groq will fail over as configured; missing Cartesia will skip TTS.

### Setup
Install dependencies

```
pip install -r requirements.txt
```

### Running
1) Start the server

```
python3 server.py
```

2) Serve the frontend (from the project folder)

```
python3 -m http.server 8000
```

3) Open the app

Visit `http://127.0.0.1:8000/` in your browser. The page connects to `ws://127.0.0.1:8765`.

### Usage
- Click Start Listening, speak, then either stop manually or wait for silence auto-stop.
- The server will:
  1) Send `listening_stopped` immediately
  2) Transcribe the recorded audio (prerecorded API)
  3) Generate an assistant response (Groq)
  4) Stream TTS audio (Cartesia) and send `tts_complete`
- After `tts_complete`, the client automatically resumes listening for the next turn.

### Key Behaviors
- Recording is blocked while TTS is playing; any ongoing recording is stopped when TTS begins.
- Audio upload uses binary WS frames to reduce overhead vs base64.
- TTS frames are 100 ms (approx. 9600 bytes at 24 kHz f32). Adjust to trade off start-up latency vs smoothness.

### Troubleshooting
- WebSocket connects but no transcript:
  - Ensure `.env` has `DEEPGRAM_API_KEY` and the key is valid.
  - The client sends WebM/Opus; the server uses Deepgram prerecorded with `mimetype: audio/webm`.
- TTS not audible:
  - Verify `CARTESIA_API_KEY` and network access.
  - Check browser console for audio context errors; user gesture may be required before audio playback.
- High latency:
  - Current design is “push-to-talk then upload”; switching back to Deepgram Live can reduce latency but requires continuous streaming and codec alignment.

### Customization
- Update voice/model in `Config` (`server.py`):
  - `TTS_MODEL`, `CARTESIA_VOICE_ID`
  - Groq model names in `generate_response`
- Silence detection parameters (client-side):
  - `vadSilenceMsToStop` (default 1000 ms)

### Security & Privacy
- API keys are loaded from `.env` and masked in logs.
- Audio is sent over WebSocket locally; use HTTPS + WSS if deploying over the network.

### File Map
- `server.py` — WebSocket server, ASR/LLM/TTS orchestration
- `index.html` — UI + WebAudio/MediaRecorder + WS client
- `requirements.txt` — dependencies



