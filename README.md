# Jarvis — assistente vocale desktop

App desktop (PySide6) che ascolta la tua voce in automatico (VAD, niente bottoni), trascrive con ElevenLabs Speech-to-Text, manda la conversazione a Claude (con tool per apertura siti/app/Spotify) e risponde a voce con Piper TTS (locale, gratuito, offline) oltre che per iscritto.

`python jarvis.py` avvia l'app (è solo un launcher per `jarvis_assistant.py`).

All'apertura: suona una canzone di benvenuto da `song/` per almeno `MIN_LOADING_SONG_S` secondi, poi Jarvis dice la frase di benvenuto configurata e si mette in ascolto.

Il vecchio script che apriva Spotify/Chrome/Cursor al doppio-clap è conservato in `clap_trigger.py` (non più collegato a `jarvis.py`).

## Setup

```bash
python -m pip install -r requirements.txt
```

## Environment variables (`.env`)

| Variable | Purpose |
| -------- | ------- |
| `ANTHROPIC_API_KEY` | Chiave API Claude (console.anthropic.com). |
| `ELEVENLABS_API_KEY` | Chiave API ElevenLabs, usata solo per la trascrizione vocale (STT). |
| `ELEVENLABS_STT_LANGUAGE` | Lingua forzata per la trascrizione (default `it`). |
| `ANTHROPIC_MODEL` | Modello Claude (default `claude-sonnet-4-6`). |
| `PIPER_VOICE_MODEL` | Nome del modello voce Piper in `.piper_voices/` (default `it_IT-paola-medium`). |

La sintesi vocale (TTS) usa [Piper](https://github.com/rhasspy/piper) in locale: nessuna chiave API, nessun costo. I modelli voce (`.onnx` + `.onnx.json`) vanno scaricati una volta in `.piper_voices/`.

## Run

```bash
python jarvis.py
```

Consenti l'accesso al microfono se richiesto da macOS. Chiudi con il bottone × (richiede conferma) o Cmd+Q.

## Troubleshooting

- **Errore TTS / niente voce:** verifica che il file `.piper_voices/<PIPER_VOICE_MODEL>.onnx` esista.
- **Trascrizione in lingua sbagliata:** imposta `ELEVENLABS_STT_LANGUAGE` in `.env`.
- **Claude non risponde / errore API:** verifica `ANTHROPIC_API_KEY` e il credito disponibile su console.anthropic.com.
