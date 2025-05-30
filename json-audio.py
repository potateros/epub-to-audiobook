import json
import os
import re
import struct
import time
import logging
from pathlib import Path
from typing import List, Dict
from google import genai
from google.genai import types
from pydub import AudioSegment


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("AudiobookGenerator")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class AudiobookGenerator:
    """Handles audio generation from chunked JSON file."""

    def __init__(self, api_key: str = None, log_level: str = "INFO"):
        self.logger = setup_logging(log_level)

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set as environment variable")

        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-pro-preview-tts"

        self.stats = {
            'total_chapters': 0,
            'processed_chapters': 0,
            'total_chunks': 0,
            'failed_chunks': 0,
            'start_time': None,
        }

    def load_chunked_json(self, json_path: str) -> Dict:
        """Load the chunked JSON file."""
        self.logger.info(f"Loading chunked JSON from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        chapters = data.get('chapters', {})

        self.logger.info(f"Loaded JSON with {len(chapters)} chapters")
        self.logger.info(f"Total chunks: {metadata.get('total_chunks', 'unknown')}")

        return data

    def combine_wav_files(self, wav_files: List[str], output_path: str):
        """Combine multiple WAV files into one using pydub."""
        if not wav_files:
            self.logger.warning("No WAV files to combine")
            return

        try:
            combined = AudioSegment.empty()
            valid_files = []
            invalid_files = []

            # First validate all input files
            for wav_file in wav_files:
                if os.path.exists(wav_file):
                    if self.validate_wav_file(wav_file):
                        valid_files.append(wav_file)
                    else:
                        invalid_files.append(wav_file)
                        self.logger.error(f"Invalid WAV file will be skipped: {wav_file}")
                else:
                    self.logger.warning(f"WAV file does not exist: {wav_file}")

            if not valid_files:
                self.logger.error("No valid WAV files to combine")
                return

            # Process files in batches to avoid memory issues
            batch_size = 10  # Adjust based on expected file sizes
            for i in range(0, len(valid_files), batch_size):
                batch = valid_files[i:i+batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(valid_files)-1)//batch_size + 1} ({len(batch)} files)")

                batch_segment = AudioSegment.empty()
                for j, wav_file in enumerate(batch):
                    try:
                        # Load the audio file
                        audio = AudioSegment.from_file(wav_file)
                        batch_segment += audio

                        # Add small pause between chunks except for the last one in the whole set
                        if i + j < len(valid_files) - 1:
                            batch_segment += AudioSegment.silent(duration=500)

                        self.logger.debug(f"Added {wav_file} to combined audio")
                    except Exception as e:
                        self.logger.error(f"Error loading WAV file {wav_file}: {e}")

                # Add batch to combined segment
                combined += batch_segment

                # Log progress
                progress = min(100, (i + len(batch)) / len(valid_files) * 100)
                self.logger.info(f"Combination progress: {progress:.1f}%")

            if len(combined) > 0:
                # Export as WAV with explicit parameters
                combined.export(
                    output_path,
                    format="wav",
                    parameters=["-ar", "24000", "-ac", "1"]  # 24kHz, mono
                )
                self.logger.info(f"Successfully combined {len(valid_files)} files into {output_path}")

                # Validate the final file
                if self.validate_wav_file(output_path):
                    self.logger.info(f"Combined file validated successfully")
                else:
                    self.logger.error(f"Combined file validation failed: {output_path}")
            else:
                self.logger.error("No valid audio content to combine")

            # Log information about skipped files
            if invalid_files:
                self.logger.warning(f"Skipped {len(invalid_files)} invalid files during combination")

        except Exception as e:
            self.logger.error(f"Error combining WAV files: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def validate_wav_file(self, file_path: str) -> bool:
        """Validate that a WAV file is properly formatted."""
        try:
            # Try to load with pydub
            audio = AudioSegment.from_wav(file_path)
            duration = len(audio) / 1000.0  # Duration in seconds
            self.logger.debug(f"WAV validation: {file_path} - {duration:.2f}s, {audio.frame_rate}Hz")
            return True
        except Exception as e:
            self.logger.error(f"WAV validation failed for {file_path}: {e}")

            # Try to examine the file structure
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(44)  # WAV header is typically 44 bytes
                    if len(header) >= 12:
                        riff = header[:4]
                        wave = header[8:12]
                        self.logger.debug(f"File signature: RIFF={riff}, WAVE={wave}")

                        if riff != b'RIFF' or wave != b'WAVE':
                            self.logger.error("File is not a valid WAV file (missing RIFF/WAVE signature)")
                    else:
                        self.logger.error("File too short to be a valid WAV file")
            except Exception as exam_e:
                self.logger.error(f"Could not examine file structure: {exam_e}")

            return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Remove problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Replace spaces with underscores
        filename = re.sub(r'\s+', '_', filename)
        # Limit length
        return filename[:50]

    def process_chapter(self, chapter_num: str, chapter_data: Dict, output_dir: str) -> str:
        """Process a single chapter from JSON data into audio."""
        chapter_title = chapter_data['title']
        chunks = chapter_data['chunks']

        # Sanitize title for filename
        safe_title = self.sanitize_filename(chapter_title)

        self.logger.info("="*80)
        self.logger.info(f"Processing Chapter {chapter_num}: '{chapter_title}'")
        self.logger.info(f"Content: {len(chunks)} chunks")

        # Create temporary directory structure
        temp_parent_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_parent_dir, exist_ok=True)

        temp_dir = os.path.join(temp_parent_dir, f"ch{chapter_num}")
        os.makedirs(temp_dir, exist_ok=True)

        # Create status tracking file
        status_file = os.path.join(temp_dir, "status.json")
        chunk_files = []

        # Check if we're resuming an interrupted operation
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    if 'chunk_files' in status_data:
                        existing_files = status_data['chunk_files']
                        # Verify these files actually exist
                        chunk_files = [f for f in existing_files if os.path.exists(f)]
                        self.logger.info(f"Resuming chapter processing with {len(chunk_files)} existing chunks")
            except Exception as e:
                self.logger.warning(f"Could not load status file, starting fresh: {e}")
                chunk_files = []

        # Get the current count of chunks
        current_chunks = len(chunk_files)
        total_needed = len(chunks) + 1  # +1 for intro

        try:
            # Generate chapter introduction if needed
            intro_file = os.path.join(temp_dir, f"{safe_title}-intro.wav")
            if not os.path.exists(intro_file) or intro_file not in chunk_files:
                intro_text = f"Chapter {chapter_num}: {chapter_title}"
                if self.generate_audio_chunk(intro_text, intro_file):
                    chunk_files.append(intro_file)
            elif intro_file not in chunk_files:
                chunk_files.append(intro_file)

            # Generate audio for each chunk if needed
            for i, chunk_text in enumerate(chunks, 1):
                chunk_file = os.path.join(temp_dir, f"{safe_title}-{i:03d}.wav")

                if not os.path.exists(chunk_file) or chunk_file not in chunk_files:
                    if self.generate_audio_chunk(chunk_text, chunk_file):
                        chunk_files.append(chunk_file)
                    else:
                        self.logger.error(f"Failed to generate audio for chunk {i}")
                elif chunk_file not in chunk_files:
                    chunk_files.append(chunk_file)

                # Save progress after each chunk
                with open(status_file, 'w') as f:
                    json.dump({'chunk_files': chunk_files}, f)

            if not chunk_files:
                self.logger.error(f"No audio files generated for Chapter {chapter_num}")
                return None

            # Combine all chunks into final chapter file
            # Use string formatting for chapter number (not numeric formatting)
            final_filename = f"Chapter_{chapter_num}_{safe_title}.wav"
            final_path = os.path.join(output_dir, final_filename)

            self.combine_wav_files(chunk_files, final_path)

            # Note: No deletion of temporary files as per requirements

            if os.path.exists(final_path):
                self.stats['processed_chapters'] += 1
                file_size = os.path.getsize(final_path) / (1024 * 1024)
                self.logger.info(f"✓ Chapter {chapter_num} completed: {final_filename} ({file_size:.1f} MB)")
                return final_path
            else:
                self.logger.error(f"✗ Failed to create final audio file for Chapter {chapter_num}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing chapter {chapter_num}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Save current progress even on error
            with open(status_file, 'w') as f:
                json.dump({'chunk_files': chunk_files}, f)

            return None

    def generate_audio_chunk(self, text: str, output_path: str) -> bool:
        """Generate audio for a single text chunk and save directly."""
        self.logger.info(f"Generating audio: {len(text)} characters -> {os.path.basename(output_path)}")

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],  # Use lowercase "audio"
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Leda"
                    )
                )
            ),
        )

        try:
            # Use non-streaming generate_content instead of generate_content_stream
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            )

            # Check if response has audio data
            if (response.candidates and
                response.candidates[0].content and
                response.candidates[0].content.parts):

                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        inline_data = part.inline_data
                        self.logger.debug(f"Received audio data: {inline_data.mime_type}, {len(inline_data.data)} bytes")

                        # Check MIME type and convert if needed
                        audio_data = inline_data.data
                        if inline_data.mime_type and not inline_data.mime_type.startswith("audio/wav"):
                            self.logger.info(f"Converting from {inline_data.mime_type} to WAV format")
                            audio_data = self.convert_to_wav(audio_data, inline_data.mime_type)

                        # Save the audio data
                        with open(output_path, "wb") as f:
                            f.write(audio_data)

                        # Validate the WAV file
                        if self.validate_wav_file(output_path):
                            self.logger.info(f"Saved and validated audio: {output_path} ({len(audio_data)} bytes)")
                            return True
                        else:
                            self.logger.error(f"Generated file failed validation: {output_path}")
                            return False

            self.logger.error("No audio data received from Gemini TTS")
            return False

        except Exception as e:
            self.logger.error(f"Error generating audio: {e}")
            self.stats['failed_chunks'] += 1
            return False

    def convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Convert audio data to WAV format."""
        parameters = self.parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
            num_channels, sample_rate, byte_rate, block_align,
            bits_per_sample, b"data", data_size
        )
        return header + audio_data

    def parse_audio_mime_type(self, mime_type: str) -> Dict[str, int]:
        """Parse audio MIME type for WAV conversion."""
        bits_per_sample = 16
        rate = 24000

        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate = int(param.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass

        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def generate_audiobook_from_json(self, json_path: str, output_dir: str) -> List[str]:
        """Generate audiobook from chunked JSON file."""
        self.stats['start_time'] = time.time()

        os.makedirs(output_dir, exist_ok=True)

        # Create temp directory structure
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Create a global status file to track overall progress
        global_status_file = os.path.join(temp_dir, "audiobook_status.json")
        processed_chapters = []

        # Check if we're resuming a previous run
        if os.path.exists(global_status_file):
            try:
                with open(global_status_file, 'r') as f:
                    global_status = json.load(f)
                    if 'processed_chapters' in global_status:
                        processed_chapters = global_status['processed_chapters']
                        self.logger.info(f"Resuming audiobook generation. Already processed: {processed_chapters}")
            except Exception as e:
                self.logger.warning(f"Could not load global status file, starting fresh: {e}")

        self.logger.info("="*80)
        self.logger.info("STARTING JSON TO AUDIOBOOK CONVERSION")
        self.logger.info("="*80)

        json_data = self.load_chunked_json(json_path)
        chapters = json_data.get('chapters', {})
        metadata = json_data.get('metadata', {})

        self.stats['total_chapters'] = len(chapters)
        self.stats['total_chunks'] = metadata.get('total_chunks', 0)

        audiobook_files = []

        # Sort chapters by number
        sorted_chapters = sorted(chapters.items(), key=lambda x: int(x[0]))

        for i, (chapter_num, chapter_data) in enumerate(sorted_chapters):
            # Skip already processed chapters unless the final file is missing
            # Use string formatting for chapter number (not numeric formatting)
            chapter_filename = f"Chapter_{chapter_num}_{self.sanitize_filename(chapter_data['title'])}.wav"
            chapter_path = os.path.join(output_dir, chapter_filename)

            if chapter_num in processed_chapters and os.path.exists(chapter_path):
                self.logger.info(f"Skipping already processed Chapter {chapter_num}")
                audiobook_files.append(chapter_path)
                continue

            try:
                audio_file = self.process_chapter(chapter_num, chapter_data, output_dir)
                if audio_file:
                    audiobook_files.append(audio_file)
                    if chapter_num not in processed_chapters:
                        processed_chapters.append(chapter_num)
                        # Update global status file
                        with open(global_status_file, 'w') as f:
                            json.dump({'processed_chapters': processed_chapters}, f)

                # Progress update
                progress = (i + 1) / len(chapters) * 100
                elapsed = time.time() - self.stats['start_time']
                self.logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(chapters)}) - Elapsed: {elapsed:.0f}s")

            except Exception as e:
                self.logger.error(f"Error processing chapter {chapter_num}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue

        # Final statistics
        total_time = time.time() - self.stats['start_time']

        self.logger.info("="*80)
        self.logger.info("AUDIOBOOK CONVERSION COMPLETED!")
        self.logger.info("="*80)
        self.logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        self.logger.info(f"Chapters processed: {self.stats['processed_chapters']}/{self.stats['total_chapters']}")
        self.logger.info(f"Generated {len(audiobook_files)} audiobook files")

        if audiobook_files:
            total_size = sum(os.path.getsize(f) for f in audiobook_files if os.path.exists(f))
            total_size_mb = total_size / (1024 * 1024)
            self.logger.info(f"Total audiobook size: {total_size_mb:.1f} MB")

        return audiobook_files


def main():
    """Main function to run the JSON to audiobook conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert chunked JSON to audiobook using Gemini TTS")
    parser.add_argument("json_path", help="Path to chunked JSON file")
    parser.add_argument("output_dir", help="Output directory for audiobook files")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Set logging level")
    parser.add_argument("--log-file", help="Optional log file path")

    args = parser.parse_args()

    # Setup file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(getattr(logging, args.log_level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger = logging.getLogger("AudiobookGenerator")
        logger.addHandler(file_handler)

    try:
        generator = AudiobookGenerator(api_key=args.api_key, log_level=args.log_level)
        audiobook_files = generator.generate_audiobook_from_json(args.json_path, args.output_dir)

        print(f"\n🎉 Audiobook generation completed!")
        print(f"📚 Generated {len(audiobook_files)} audiobook files")
        if audiobook_files:
            print(f"📁 Files saved to: {args.output_dir}")
            print(f"📁 Temp files saved to: {os.path.join(args.output_dir, 'temp')}")
            for audio_file in audiobook_files:
                print(f"  - {os.path.basename(audio_file)}")

    except KeyboardInterrupt:
        print("\n❌ Conversion interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
