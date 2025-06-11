# EPUB to Audiobook Converter

Convert EPUB books to audiobooks using Google's Gemini TTS API. High-quality voice synthesis with resumable processing.

Note: I've tested and about 800 words (equals to about 4-5mins of audio) is close to the maximum output token limit for the Gemini-2.5-Flash-TTS and Gemini-2.5-Pro-TTS (16k tokens).

Note2: The Pro model produces speech that sounds a lot more natural. I chose the 'Leda' voice, but you can choose other voices from the Gemini API here https://ai.google.dev/gemini-api/docs/speech-generation#voices

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API key**
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key"
   ```

3. **Convert EPUB to audiobook**
   ```bash
   # Step 1: Extract and chunk EPUB
   python epub-json.py input/book.epub output/book.json

   # Step 2: Generate audio
   python json-audio.py output/book.json output/audiobook/
   ```

## How It Works

The conversion happens in two stages:

1. **EPUB → JSON**: Extracts chapters and splits text into ~1700-word chunks
2. **JSON → Audio**: Converts chunks to WAV files using Gemini TTS, then combines into chapter files

Features:
- Preserves chapter structure and reading order
- Smart text chunking at sentence boundaries
- Resumable processing (can restart if interrupted)

## Command Options

**epub-json.py**
```bash
python epub-json.py <epub_path> <output_path> [options]

Options:
  --max-words MAX           Words per chunk (default: 1700)
  --analyze-only           Just analyze EPUB structure
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

**json-audio.py**
```bash
python json-audio.py <json_path> <output_dir> [options]

Options:
  --api-key KEY            Gemini API key
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

## Output

Generated files:
- `Chapter_N_Title.wav`: Final chapter audio files
- `temp/`: Intermediate files for resuming interrupted processing

## Requirements

- Python 3.7+
- Valid Gemini API key from Google AI Studio
- ~1GB disk space per 100-page book (temporary files)
