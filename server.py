import asyncio
import websockets
import json
import base64
import logging
from typing import Optional, Dict, Any
import os
from dataclasses import dataclass
import threading
import queue
import time

# Third-party imports (install with pip)
try:
    from deepgram import DeepgramClient, PrerecordedOptions
    import webrtcvad
    import groq
    import requests
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install deepgram-sdk webrtcvad groq requests")
    exit(1)

# Configuration
@dataclass
class Config:
    # API Keys (loaded from .env file at E:\Voice_Agent\.env)
    from dotenv import load_dotenv
    load_dotenv("E:\\Voice_Agent\\.env")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    CARTESIA_API_KEY: str = os.getenv("CARTESIA_API_KEY", "")
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    VAD_MODE: int = 3  # Aggressive VAD
    FRAME_DURATION: int = 30  # ms
    
    # TTS settings
    CARTESIA_VOICE_ID: str = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Default voice
    TTS_MODEL: str = "sonic-english"
    
    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8765

config = Config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _mask_secret(value: str) -> str:
    """Return a masked representation of a secret for safe logging."""
    if not value:
        return "<empty>"
    visible = value[:4]
    return f"{visible}{'*' * max(0, len(value) - 4)}"

def _is_meaningful_text(text: str) -> bool:
    """Return True if text has at least one alphanumeric character."""
    if not text:
        return False
    return any(ch.isalnum() for ch in text)

# Early debug: confirm whether API keys were loaded from the environment (masked)
logger.info(
    "Loaded API keys (masked): DEEPGRAM=%s, GROQ=%s, CARTESIA=%s",
    _mask_secret(config.DEEPGRAM_API_KEY),
    _mask_secret(config.GROQ_API_KEY),
    _mask_secret(config.CARTESIA_API_KEY),
)

class VoiceActivityDetector:
    """Ricky VAD implementation using WebRTC VAD"""
    
    def __init__(self, sample_rate=16000, frame_duration=30, mode=3):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.vad = webrtcvad.Vad(mode)
        self.buffer = b''
        
    def is_speech(self, audio_data: bytes) -> bool:
        """Check if audio data contains speech"""
        self.buffer += audio_data
        
        # Process complete frames
        while len(self.buffer) >= self.frame_size * 2:  # 2 bytes per sample (16-bit)
            frame = self.buffer[:self.frame_size * 2]
            self.buffer = self.buffer[self.frame_size * 2:]
            
            try:
                return self.vad.is_speech(frame, self.sample_rate)
            except:
                return False
        
        return False

class AudioBuffer:
    """Buffer for smooth TTS audio playback"""
    
    def __init__(self, min_buffer_size=8192):
        self.buffer = queue.Queue()
        self.min_buffer_size = min_buffer_size
        self.current_size = 0
        self.is_playing = False
        
    def add_chunk(self, chunk: bytes):
        """Add audio chunk to buffer"""
        self.buffer.put(chunk)
        self.current_size += len(chunk)
        
    def get_chunk(self) -> Optional[bytes]:
        """Get audio chunk from buffer"""
        try:
            chunk = self.buffer.get_nowait()
            self.current_size -= len(chunk)
            return chunk
        except queue.Empty:
            return None
            
    def is_ready(self) -> bool:
        """Check if buffer has enough data for smooth playback"""
        return self.current_size >= self.min_buffer_size
        
    def clear(self):
        """Clear buffer"""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break
        self.current_size = 0

class VoiceAgent:
    """Main voice agent class"""
    
    def __init__(self):
        self.deepgram_client = DeepgramClient(config.DEEPGRAM_API_KEY) if config.DEEPGRAM_API_KEY else None
        self.groq_client = groq.Groq(api_key=config.GROQ_API_KEY)
        self.vad = VoiceActivityDetector()
        self.audio_buffer = AudioBuffer()
        
        self.conversation_history = []
        self.is_listening = False
        self.current_transcript = ""
        # Collected binary audio chunks (from MediaRecorder) for this session
        self.recording_chunks = []
        
    async def handle_websocket(self, websocket):
        """Handle WebSocket connection"""
        try:
            peer = getattr(websocket, 'remote_address', None)
        except Exception:
            peer = None
        logger.info("WS: connection opened from %s", peer)
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary audio chunk from client; collect during active session
                    if self.is_listening:
                        self.recording_chunks.append(message)
                        logger.debug("ASR: collected chunk len=%d total_chunks=%d", len(message), len(self.recording_chunks))
                    continue
                # Text control messages
                logger.debug("WS: received message (len=%d)", len(message) if isinstance(message, (bytes, str)) else -1)
                data = json.loads(message)
                logger.debug("WS: parsed message keys=%s", list(data.keys()))
                await self.process_message(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WS: connection closed")
        except Exception as e:
            logger.exception("WS: error handling message: %s", e)
            
    async def process_message(self, websocket, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        message_type = data.get("type")
        logger.info("WS: process_message type=%s", message_type)
        
        if message_type == "start_listening":
            await self.start_listening(websocket)
        elif message_type == "stop_listening":
            await self.stop_listening(websocket)
        elif message_type == "audio_data":
            audio_len = len(data.get("audio", "")) if isinstance(data.get("audio"), str) else -1
            logger.debug("WS: audio_data received len(base64)=%d", audio_len)
            await self.process_audio(websocket, data["audio"])
        elif message_type == "get_history":
            await self.send_conversation_history(websocket)
        elif message_type == "clear_history":
            self.conversation_history.clear()
            await websocket.send(json.dumps({
                "type": "history_cleared"
            }))
            logger.info("WS: conversation history cleared")
            
    async def start_listening(self, websocket):
        """Start listening for audio input"""
        self.is_listening = True
        self.current_transcript = ""
        self.recording_chunks = []
        logger.info("ASR: start_listening")
        
        await websocket.send(json.dumps({
            "type": "listening_started"
        }))
        
    async def stop_listening(self, websocket):
        """Stop listening and process final transcript"""
        self.is_listening = False
        logger.info("ASR: stop_listening collected_chunks=%d", len(self.recording_chunks))
        # Notify UI immediately that listening has stopped (processing may follow)
        await websocket.send(json.dumps({
            "type": "listening_stopped"
        }))
        # Combine collected chunks and transcribe via prerecorded API
        try:
            if self.recording_chunks:
                audio_bytes = b"".join(self.recording_chunks)
                logger.info("ASR: collected total bytes=%d for prerecorded transcription", len(audio_bytes))
                await self.transcribe_audio(websocket, audio_bytes)
            else:
                logger.info("ASR: no audio chunks collected")
        except Exception as e:
            logger.exception("ASR: error during prerecorded transcription: %s", e)
        
    async def process_audio(self, websocket, audio_base64: str):
        """Deprecated: previous base64 audio upload path (kept for compatibility)."""
        try:
            audio_data = base64.b64decode(audio_base64)
            logger.info("ASR: received legacy audio bytes=%d", len(audio_data))
            await self.transcribe_audio(websocket, audio_data)
        except Exception as e:
            logger.exception("ASR: legacy audio processing error: %s", e)

    async def process_audio_chunk(self, audio_bytes: bytes):
        """Collect audio chunk during active recording."""
        if self.is_listening:
            self.recording_chunks.append(audio_bytes)
            
    async def transcribe_audio(self, websocket, audio_data: bytes):
        """Transcribe audio using Deepgram prerecorded API."""
        try:
            if not self.deepgram_client:
                logger.warning("ASR: DEEPGRAM_API_KEY missing; skipping transcription and using fallback text")
                self.current_transcript = ""
                await websocket.send(json.dumps({
                    "type": "transcript_final",
                    "text": "",
                    "speaker": "user"
                }))
                await self.generate_response(websocket, "")
                return

            options = PrerecordedOptions(
                model="nova-2",
                language="en-US",
                smart_format=True
            )
            logger.info("ASR: transcribe_audio start bytes=%d", len(audio_data))
            
            # Use prerecorded file transcription for uploaded webm audio
            response = self.deepgram_client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio_data, "mimetype": "audio/webm"},
                options
            )
            logger.debug("ASR: DG response received")
            
            if response.results.channels[0].alternatives:
                transcript = response.results.channels[0].alternatives[0].transcript
                if transcript:
                    self.current_transcript = transcript.strip()
                    logger.info("ASR: interim transcript len=%d", len(self.current_transcript))
                    
                    # Add user message to history once
                    self.conversation_history.append({
                        "role": "user",
                        "content": self.current_transcript
                    })

                    # Send final result to frontend
                    await websocket.send(json.dumps({
                        "type": "transcript_final",
                        "text": self.current_transcript,
                        "speaker": "user"
                    }))
                    
                    # Generate response
                    await self.generate_response(websocket, self.current_transcript)
                else:
                    # No transcript extracted; proceed with fallback response
                    self.current_transcript = ""
                    await websocket.send(json.dumps({
                        "type": "transcript_final",
                        "text": "",
                        "speaker": "user"
                    }))
                    await self.generate_response(websocket, "")
                    
        except Exception as e:
            logger.exception("ASR: transcription error: %s", e)
            # Fallback: still move to response generation
            await websocket.send(json.dumps({
                "type": "transcript_final",
                "text": "",
                "speaker": "user"
            }))
            await self.generate_response(websocket, "")
            
    async def generate_response(self, websocket, user_input: str):
        """Generate AI response using Groq"""
        try:
            MODEL_PRIMARY = "llama-3.3-70b-versatile"
            MODEL_FALLBACK = "llama-3.1-8b-instant"
            # Prepare conversation context
            messages = [
                {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise and conversational."}
            ]
            
            # Add conversation history (last 10 messages)
            messages.extend(self.conversation_history[-10:])
            logger.info("NLP: generating response, history=%d", len(self.conversation_history))
            
            # Generate response (with fallback model)
            try:
                completion = self.groq_client.chat.completions.create(
                    model=MODEL_PRIMARY,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150
                )
            except Exception as primary_err:
                logger.warning("NLP: primary model failed (%s). Trying fallback %s", str(primary_err)[:120], MODEL_FALLBACK)
                completion = self.groq_client.chat.completions.create(
                    model=MODEL_FALLBACK,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=150
                )
            
            response_text = completion.choices[0].message.content or ""
            if not _is_meaningful_text(response_text):
                logger.warning("NLP: empty/meaningless response; using friendly fallback")
                response_text = "Sorry, I didn't quite catch that. Could you rephrase?"
            logger.info("NLP: response generated len=%d", len(response_text) if response_text else 0)
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Send response to frontend
            await websocket.send(json.dumps({
                "type": "ai_response",
                "text": response_text
            }))
            
            # Generate TTS
            await self.generate_tts(websocket, response_text)
            
        except Exception as e:
            logger.exception("NLP: response generation error: %s", e)
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Failed to generate response"
            }))
            
    async def generate_tts(self, websocket, text: str):
        """Generate TTS using Cartesia"""
        try:
            if not _is_meaningful_text(text):
                logger.info("TTS: skipped (empty or punctuation-only text)")
                await websocket.send(json.dumps({
                    "type": "tts_complete"
                }))
                return
            url = "https://api.cartesia.ai/tts/bytes"
            
            payload = {
                "model_id": config.TTS_MODEL,
                "voice": {
                    "mode": "id",
                    "id": config.CARTESIA_VOICE_ID
                },
                "transcript": text,
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": 24000
                }
            }
            
            headers = {
                "X-API-Key": config.CARTESIA_API_KEY,
                "Content-Type": "application/json",
                "Cartesia-Version": "2025-04-16",
                "Accept": "application/octet-stream"
            }
            
            safe_text = text.strip()
            logger.info(
                "TTS: provider=Cartesia model=%s voice_id=%s encoding=pcm_f32le sample_rate=%d text_len=%d",
                payload["model_id"],
                payload["voice"]["id"],
                payload["output_format"]["sample_rate"],
                len(safe_text)
            )
            response = requests.post(url, json=payload, headers=headers, stream=True)

            if response.status_code == 200:
                # Stream audio with proper framing buffers (100ms per frame)
                bytes_per_sample = 4  # pcm_f32le
                sample_rate = 24000
                frame_ms = 100
                frame_bytes = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)  # 9600 bytes

                accumulator = bytearray()
                total_sent = 0

                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    accumulator.extend(chunk)

                    # While we have at least one full frame, send it
                    while len(accumulator) >= frame_bytes:
                        out = accumulator[:frame_bytes]
                        del accumulator[:frame_bytes]
                        total_sent += len(out)
                        await websocket.send(json.dumps({
                            "type": "tts_audio",
                            "audio": base64.b64encode(out).decode()
                        }))
                    # Yield occasionally to avoid blocking the event loop
                    await asyncio.sleep(0)

                # Flush any remaining partial frame
                if len(accumulator) > 0:
                    out = bytes(accumulator)
                    total_sent += len(out)
                    await websocket.send(json.dumps({
                        "type": "tts_audio",
                        "audio": base64.b64encode(out).decode()
                    }))

                # Signal end of TTS
                await websocket.send(json.dumps({
                    "type": "tts_complete"
                }))
                logger.info("TTS: complete, total_bytes_sent=%d", total_sent)
                
            else:
                body_preview = response.text[:200] if hasattr(response, 'text') else ""
                logger.error("TTS: error status=%s body=%s", response.status_code, body_preview)
                # Fallback: handle punctuation-only errors by speaking a generic acknowledgement
                if response.status_code == 400 and isinstance(body_preview, str) and "invalid transcript" in body_preview.lower():
                    logger.info("TTS: retrying with fallback text")
                    payload_retry = dict(payload)
                    payload_retry["transcript"] = "Okay."
                    response_retry = requests.post(url, json=payload_retry, headers=headers, stream=True)
                    if response_retry.status_code == 200:
                        self.audio_buffer.clear()
                        for chunk in response_retry.iter_content(chunk_size=4096):
                            if chunk:
                                self.audio_buffer.add_chunk(chunk)
                                if self.audio_buffer.is_ready():
                                    audio_chunk = self.audio_buffer.get_chunk()
                                    if audio_chunk:
                                        await websocket.send(json.dumps({
                                            "type": "tts_audio",
                                            "audio": base64.b64encode(audio_chunk).decode()
                                        }))
                        while True:
                            chunk = self.audio_buffer.get_chunk()
                            if not chunk:
                                break
                            await websocket.send(json.dumps({
                                "type": "tts_audio",
                                "audio": base64.b64encode(chunk).decode()
                            }))
                        await websocket.send(json.dumps({"type": "tts_complete"}))
                        logger.info("TTS: fallback complete")
                    else:
                        logger.error("TTS: fallback failed status=%s body=%s", response_retry.status_code, response_retry.text[:200] if hasattr(response_retry, 'text') else "")
                
        except Exception as e:
            logger.exception("TTS: generation error: %s", e)
            
    async def send_conversation_history(self, websocket):
        """Send conversation history to frontend"""
        logger.info("WS: sending conversation_history count=%d", len(self.conversation_history))
        await websocket.send(json.dumps({
            "type": "conversation_history",
            "history": self.conversation_history
        }))

async def main():
    """Main entry point"""
    if not all([config.DEEPGRAM_API_KEY, config.GROQ_API_KEY, config.CARTESIA_API_KEY]):
        logger.error("Missing required API keys. Set DEEPGRAM_API_KEY, GROQ_API_KEY, and CARTESIA_API_KEY environment variables.")
        return

    agent = VoiceAgent()

    logger.info(f"Starting voice agent server on {config.HOST}:{config.PORT}")

    async with websockets.serve(
        agent.handle_websocket,
        config.HOST,
        config.PORT
    ):
        # Keep the server running forever
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())