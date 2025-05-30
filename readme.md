# EPUB to Audiobook Converter

Converts EPUB books to audiobooks using Gemini Pro TTS API.

## Installation

```sh
pip install -r requirements.txt
```

## Usage

Note: I've tested and about 800 words (equals to about 4-5mins of audio) is close to the maximum output token limit for the Gemini-2.5-Flash-TTS and Gemini-2.5-Pro-TTS (16k tokens).

Note2: The Pro model produces speech that sounds a lot more natural. I chose the 'Leda' voice, but you can choose other voices from the Gemini API here https://ai.google.dev/gemini-api/docs/speech-generation#voices

1. Convert EPUB to JSON

```sh
python epub-json.py <epub_path> <output_path> [options]

Options:
  --log-level {DEBUG,INFO,WARNING,ERROR}
  --log-file PATH              Optional log file path
  --analyze-only               Analyze EPUB structure only
  --chunk-by-tokens           Chunk by token count instead of words
  --max-words MAX             Maximum words per chunk (default: 1700)
```

2. Convert JSON to Audiobook

```sh
python json-audio.py <json_path> <output_dir> [options]

Options:
  --api-key KEY               Gemini API key (or set GEMINI_API_KEY env var)
  --log-level {DEBUG,INFO,WARNING,ERROR}
  --log-file PATH            Optional log file path
  ```

Splits EPUB into chunks, processes them through Gemini TTS, and combines into chapter-based WAV files.
