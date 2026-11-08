import os, site, re, time, logging, threading
from contextlib import contextmanager
for p in site.getsitepackages():
    for sub in ("cublas", "cudnn"):
        d = os.path.join(p, "nvidia", sub, "bin")
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
#python -m pip install openwakeword faster-whisper piper-tts sounddevice numpy requests webrtcvad-wheels rapidfuzz
#wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx
#wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json

# ---- konfiguracja ----
API_URL = "http://localhost:11434/v1/chat/completions"
API_KEY = ""
MODEL = "pllum-home:latest"
WAKE_MODEL_PATH = "hey_pooshia.onnx"
WAKE_WORD = "hey_pooshia"
WAKE_THRESHOLD = 0.5
WHISPER_MODEL = "large-v3-turbo"
DEVICE, COMPUTE = "cuda", "float16"
SILENCE_SEC = 0.8
MAX_REC_SEC = 20
SYSTEM_PROMPT = "Odpowiadaj po polsku, krótko i naturalnie, bez markdownu i list - twoja odpowiedź będzie czytana na głos."

# ---- logowanie ----
LOG_LEVEL = "DEBUG"                # DEBUG = szczegóły (score wake word, VAD), INFO = tylko akcje i czasy
LOG_FILE = "asystent.log"          # None = tylko konsola

# ---- słownik ----
CUSTOM_WORDS = ["Pusia"]
REPLACEMENTS = {
    "pusia": "Pusia",
    "pusza": "Pusia",
    "pucia": "Pusia",
}
FUZZY_CUTOFF = 85
MIN_WORD_LEN = 4
# ----------------------

handlers = [logging.StreamHandler()]
if LOG_FILE:
    handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=handlers,
)
log = logging.getLogger("asystent")


@contextmanager
def timed(name, store=None, level=logging.INFO):
    """Mierzy czas bloku, loguje go i opcjonalnie zapisuje do słownika."""
    t0 = time.perf_counter()
    log.debug("→ start: %s", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if store is not None:
            store[name] = dt
        log.log(level, "⏱ %s: %.0f ms", name, dt * 1000)


T_START = time.perf_counter()
log.info("Start programu, ładowanie bibliotek...")
with timed("import bibliotek"):
    import numpy as np, sounddevice as sd, requests, webrtcvad
    from rapidfuzz import process, fuzz
    from openwakeword.model import Model as WakeModel
    from faster_whisper import WhisperModel
    from piper import PiperVoice

SR = 16000
CHUNK = 1280                       # 80 ms, wymagane przez openWakeWord
FRAME = 320                        # 20 ms dla VAD
BEEP_SR = 44100
TAIL_CHUNKS = 3                    # ~240 ms zapasu po ostatniej mowie

with timed("ładowanie wake word"):
    wake = WakeModel(wakeword_models=[WAKE_MODEL_PATH], inference_framework="onnx")
with timed(f"ładowanie Whisper ({WHISPER_MODEL}, {DEVICE}/{COMPUTE})"):
    stt = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE)
vad = webrtcvad.Vad(2)
with timed("ładowanie Piper"):
    tts = PiperVoice.load("pl_PL-gosia-medium.onnx")
history = [{"role": "system", "content": SYSTEM_PROMPT}]

INITIAL_PROMPT = ("Rozmowa po polsku z domowym asystentem głosowym o imieniu Pusia. "
                  "Słownik: " + ", ".join(CUSTOM_WORDS) + ".")
_lower_words = {w.lower(): w for w in CUSTOM_WORDS}


# ---------- dźwięki ----------
def _tone(freq, dur, vol=0.25):
    t = np.linspace(0, dur, int(BEEP_SR * dur), False)
    wave = np.sin(2 * np.pi * freq * t)
    fade = int(BEEP_SR * 0.01)
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return wave * env * vol


def play_pattern(pattern, wait=True):
    parts = []
    for freq, dur in pattern:
        parts.append(_tone(freq, dur))
        parts.append(np.zeros(int(BEEP_SR * 0.03)))
    sd.play((np.concatenate(parts) * 32767).astype(np.int16), BEEP_SR)
    if wait:
        sd.wait()


def beep_listening():
    with timed("pip 'słucham'"):
        play_pattern([(880, 0.08), (1320, 0.08)])


@contextmanager
def waiting_beeps(label, freq=600, interval=0.5):
    """Pipa co `interval` sekund w tle, dopóki trwa blok (rozpoznawanie / czekanie na LLM)."""
    tick = (_tone(freq, 0.05, vol=0.2) * 32767).astype(np.int16)
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            sd.play(tick, BEEP_SR)
            stop.wait(interval)

    th = threading.Thread(target=worker, daemon=True)
    log.info("🔔 pipam co %.1f s: %s", interval, label)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join()
        sd.stop()                          # nie zostawiaj niedogranego pipa przed odpowiedzią


# ---------- korekta słownikowa ----------
def correct(text):
    changes = []

    def fix(m):
        word = m.group(0)
        low = word.lower()
        new = word
        if low in REPLACEMENTS:
            new = REPLACEMENTS[low]
        elif low in _lower_words:
            new = _lower_words[low]
        elif len(low) >= MIN_WORD_LEN:
            match = process.extractOne(low, _lower_words.keys(), scorer=fuzz.ratio,
                                       score_cutoff=FUZZY_CUTOFF)
            if match:
                new = _lower_words[match[0]]
        if new != word:
            changes.append(f"{word}→{new}")
        return new

    out = re.sub(r"\w+", fix, text)
    if changes:
        log.info("✏️ korekta słownikowa: %s", ", ".join(changes))
    else:
        log.debug("korekta słownikowa: bez zmian")
    return out


# ---------- audio ----------
def is_speech(chunk):
    b = chunk.tobytes()
    votes = [vad.is_speech(b[i*FRAME*2:(i+1)*FRAME*2], SR) for i in range(CHUNK // FRAME)]
    return sum(votes) >= 2


def record(stream):
    frames, started, silent, last_speech = [], False, 0, -1
    reason = "limit czasu"
    t0 = time.perf_counter()
    for _ in range(int(MAX_REC_SEC * SR / CHUNK)):
        chunk = stream.read(CHUNK)[0][:, 0]
        frames.append(chunk)
        if is_speech(chunk):
            if not started:
                log.info("🗣️ wykryto początek mowy (po %.0f ms od pipa)",
                         (time.perf_counter() - t0) * 1000)
            started, silent, last_speech = True, 0, len(frames) - 1
        elif started:
            silent += 1
            if silent * CHUNK / SR >= SILENCE_SEC:
                reason = f"cisza {SILENCE_SEC:.1f}s"
                break
        elif len(frames) * CHUNK / SR > 5:
            reason = "nikt nic nie powiedział (5 s)"
            log.warning("Koniec nagrywania: %s", reason)
            return None
    if last_speech < 0:
        log.warning("Koniec nagrywania: brak wykrytej mowy")
        return None
    audio = np.concatenate(frames[:last_speech + 1 + TAIL_CHUNKS])
    log.info("🎙️ koniec nagrywania (%s): nagrano %.2f s, do Whispera %.2f s",
             reason, len(frames) * CHUNK / SR, len(audio) / SR)
    return audio


def transcribe(audio):
    audio_sec = len(audio) / SR
    t0 = time.perf_counter()
    segs, info = stt.transcribe(
        audio.astype(np.float32) / 32768.0,
        language="pl",
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
        without_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=INITIAL_PROMPT,
    )
    raw = " ".join(s.text.strip() for s in segs).strip()   # generator - tu dzieje się właściwa praca
    dt = time.perf_counter() - t0
    log.info("⏱ Whisper: %.0f ms (audio %.2f s, RTF %.2f)", dt * 1000, audio_sec, dt / audio_sec)
    log.debug("Whisper surowy tekst: %r", raw)
    return correct(raw), dt


def warmup():
    list(stt.transcribe(np.zeros(SR, dtype=np.float32), language="pl", beam_size=1)[0])


def ask_llm(text):
    history.append({"role": "user", "content": text})
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    log.info("🤖 wysyłam do LLM (%s), wiadomości w historii: %d", MODEL, len(history))
    t0 = time.perf_counter()
    r = requests.post(API_URL, headers=headers, timeout=120,
                      json={"model": MODEL, "messages": history})
    r.raise_for_status()
    dt = time.perf_counter() - t0
    data = r.json()
    answer = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage") or {}
    extra = ""
    if usage.get("completion_tokens"):
        extra = f", {usage['completion_tokens']} tokenów ({usage['completion_tokens'] / dt:.1f} tok/s)"
    log.info("⏱ LLM: %.0f ms, odpowiedź %d znaków%s", dt * 1000, len(answer), extra)
    history.append({"role": "assistant", "content": answer})
    del history[1:-20]
    return answer, dt


def speak(text, t_ref=None):
    """t_ref - moment końca wypowiedzi użytkownika; liczymy opóźnienie do pierwszego dźwięku."""
    synth_total, play_total, first = 0.0, 0.0, True
    t_wall = time.perf_counter()
    it = iter(tts.synthesize(text))
    n_chunks = 0
    while True:
        t0 = time.perf_counter()
        try:
            chunk = next(it)
        except StopIteration:
            break
        synth_total += time.perf_counter() - t0
        n_chunks += 1
        pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        if first and t_ref is not None:
            log.info("⚡ opóźnienie: %.0f ms od końca Twojej wypowiedzi do pierwszego dźwięku",
                     (time.perf_counter() - t_ref) * 1000)
        first = False
        t1 = time.perf_counter()
        sd.play(pcm, chunk.sample_rate)
        sd.wait()
        play_total += time.perf_counter() - t1
    log.info("⏱ TTS: synteza %.0f ms, odtwarzanie %.0f ms (fragmentów: %d, razem %.0f ms)",
             synth_total * 1000, play_total * 1000, n_chunks, (time.perf_counter() - t_wall) * 1000)
    return synth_total, play_total


def handle_interaction(stream):
    t = {}                                   # czasy etapów do podsumowania
    t_wake = time.perf_counter()

    beep_listening()
    stream.read(stream.read_available)       # wyrzuć pip i resztki słowa wybudzającego

    log.info("🎤 nagrywam...")
    t0 = time.perf_counter()
    audio = record(stream)
    t["nagrywanie"] = time.perf_counter() - t0
    if audio is None:
        return
    t_end_speech = time.perf_counter()

    with waiting_beeps("rozpoznawanie mowy", freq=600):
        text, t["Whisper"] = transcribe(audio)
    log.info("👤 Ty: %s", text)
    if not text:
        log.warning("Pusta transkrypcja, pomijam")
        return

    with waiting_beeps("czekam na LLM", freq=800):
        answer, t["LLM"] = ask_llm(text)
    log.info("💬 LLM: %s", answer)
    t["TTS synteza"], t["TTS odtwarzanie"] = speak(answer, t_ref=t_end_speech)

    # podsumowanie
    total = time.perf_counter() - t_wake
    parts = " | ".join(f"{k} {v*1000:.0f} ms" for k, v in t.items())
    log.info("📊 PODSUMOWANIE: %s | RAZEM %.2f s", parts, total)


def main():
    with timed("rozgrzewka Whisper"):
        warmup()
    log.info("🚀 gotowy po %.1f s od startu", time.perf_counter() - T_START)
    speak("Cześć, jestem Pusia, słucham...")
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=CHUNK) as stream:
        log.info("👂 nasłuchuję słowa '%s' (próg %.2f)", WAKE_WORD, WAKE_THRESHOLD)
        while True:
            chunk = stream.read(CHUNK)[0][:, 0]
            t0 = time.perf_counter()
            score = wake.predict(chunk)[WAKE_WORD]
            if score > 0.1:
                log.debug("wake score=%.2f (inferencja %.1f ms)", score, (time.perf_counter() - t0) * 1000)
            if score < WAKE_THRESHOLD:
                continue
            log.info("✅ wykryto wake word (score %.2f)", score)
            try:
                handle_interaction(stream)
            except requests.RequestException as e:
                log.error("Błąd połączenia z LLM: %s", e)
            except Exception:
                log.exception("Nieoczekiwany błąd w obsłudze interakcji")
            finally:
                wake.reset()
                stream.read(stream.read_available)
                log.info("👂 nasłuchuję...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Zatrzymano (Ctrl+C)")