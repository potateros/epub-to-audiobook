# To run this code you need to install the following dependencies:
# pip install ebooklib tiktoken beautifulsoup4

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import List, Dict
import ebooklib
from ebooklib import epub
import tiktoken
from bs4 import BeautifulSoup


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("EPUBProcessor")
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


class EPUBProcessor:
    """Handles EPUB extraction and chunking into JSON format."""

    def __init__(self, log_level: str = "INFO"):
        self.logger = setup_logging(log_level)
        self.max_tokens = 7500

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.logger.info("Successfully initialized tiktoken tokenizer")
        except Exception as e:
            self.logger.warning(f"Failed to initialize tiktoken: {e}. Using fallback token counting.")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback method."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4  # Rough estimation

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def clean_text(self, text: str) -> str:
        """Clean extracted text content."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Clean up quotes using simple string replacement
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        text = text.replace(''', "'")
        text = text.replace(''', "'")

        return text.strip()

    def extract_text_from_soup(self, soup: BeautifulSoup) -> str:
        """Extract text content from BeautifulSoup."""
        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()

        # Get text with proper spacing
        text = soup.get_text(separator=' ', strip=True)

        # If we got very little text, try different approaches
        if len(text) < 50:
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)

            if len(text) < 50:
                for tag in ['div', 'main', 'article', 'section']:
                    content_div = soup.find(tag)
                    if content_div:
                        text = content_div.get_text(separator=' ', strip=True)
                        if len(text) > 50:
                            break

        return text

    def is_navigation_content(self, text: str, filename: str) -> bool:
        """Check if content appears to be navigation/TOC."""
        # Check filename patterns
        nav_filenames = ['toc', 'contents', 'nav', 'index', 'copyright', 'title']
        if any(nav in filename.lower() for nav in nav_filenames):
            return True

        # Check content patterns
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 5:
            short_lines = sum(1 for line in lines if len(line) < 50)
            if short_lines / len(lines) > 0.7:
                return True

        return False

    def extract_chapter_title(self, soup: BeautifulSoup, filename: str) -> str:
        """Extract chapter title from HTML content."""
        # Try to find title in common heading tags
        for tag in ['h1', 'h2', 'h3', 'title']:
            heading = soup.find(tag)
            if heading and heading.get_text(strip=True):
                title = heading.get_text(strip=True)
                # Clean up the title
                cleaned_title = re.sub(r'^(chapter|ch\.?)\s*\d*:?\s*', '', title, flags=re.IGNORECASE)
                if cleaned_title and len(cleaned_title) < 100:
                    return cleaned_title

        # Fallback to filename
        title = Path(filename).stem
        title = re.sub(r'^(chapter|ch\.?)\s*\d*', '', title, flags=re.IGNORECASE)
        return title.replace('_', ' ').replace('-', ' ').title()

    def extract_chapters_from_epub(self, epub_path: str) -> List[Dict]:
        """Extract chapters from EPUB file."""
        self.logger.info(f"Starting EPUB extraction from: {epub_path}")
        start_time = time.time()

        try:
            book = epub.read_epub(epub_path)
            self.logger.info("EPUB file successfully loaded")
        except Exception as e:
            self.logger.error(f"Failed to read EPUB file: {e}")
            raise

        # Analyze EPUB structure
        all_items = list(book.get_items())
        self.logger.info(f"Total items in EPUB: {len(all_items)}")

        chapters = []
        spine_items = [item[0] for item in book.spine]
        self.logger.info(f"Found {len(spine_items)} items in EPUB spine")

        chapter_num = 1

        # Process spine items first
        for i, item_id in enumerate(spine_items):
            self.logger.debug(f"Processing spine item {i+1}/{len(spine_items)}: {item_id}")

            item = book.get_item_with_id(item_id)
            if not item:
                continue

            # Accept document items and HTML files
            if (item.get_type() == ebooklib.ITEM_DOCUMENT or
                item.get_name().lower().endswith(('.html', '.xhtml', '.htm'))):

                try:
                    content = item.get_content()
                    if not content:
                        continue

                    soup = BeautifulSoup(content, 'html.parser')
                    text_content = self.extract_text_from_soup(soup)
                    text_content = self.clean_text(text_content)

                    content_length = len(text_content.strip())

                    # Skip very short content
                    if content_length < 50:
                        self.logger.debug(f"Skipping short content ({content_length} chars): {item.get_name()}")
                        continue

                    # Skip navigation content
                    if self.is_navigation_content(text_content, item.get_name()):
                        self.logger.debug(f"Skipping navigation content: {item.get_name()}")
                        continue

                    title = self.extract_chapter_title(soup, item.get_name())

                    chapters.append({
                        'number': chapter_num,
                        'title': title,
                        'content': text_content,
                        'item_id': item_id,
                        'item_name': item.get_name(),
                        'character_count': content_length,
                        'estimated_tokens': self.count_tokens(text_content)
                    })

                    self.logger.info(f"Added Chapter {chapter_num}: '{title}' ({content_length} chars)")
                    chapter_num += 1

                except Exception as e:
                    self.logger.error(f"Error processing item {item_id}: {e}")
                    continue

        # If no chapters found in spine, try all HTML items
        if len(chapters) == 0:
            self.logger.warning("No chapters found in spine, trying all HTML items...")

            for item in all_items:
                if (item.get_type() == ebooklib.ITEM_DOCUMENT or
                    item.get_name().lower().endswith(('.html', '.xhtml', '.htm'))):

                    try:
                        content = item.get_content()
                        if not content:
                            continue

                        soup = BeautifulSoup(content, 'html.parser')
                        text_content = self.extract_text_from_soup(soup)
                        text_content = self.clean_text(text_content)

                        content_length = len(text_content.strip())

                        if content_length < 30:
                            continue

                        if self.is_navigation_content(text_content, item.get_name()):
                            continue

                        title = self.extract_chapter_title(soup, item.get_name())

                        chapters.append({
                            'number': chapter_num,
                            'title': title,
                            'content': text_content,
                            'item_id': item.get_id(),
                            'item_name': item.get_name(),
                            'character_count': content_length,
                            'estimated_tokens': self.count_tokens(text_content)
                        })

                        self.logger.info(f"Added Chapter {chapter_num}: '{title}' ({content_length} chars)")
                        chapter_num += 1

                    except Exception as e:
                        self.logger.debug(f"Error processing HTML item {item.get_name()}: {e}")
                        continue

        extraction_time = time.time() - start_time
        self.logger.info(f"EPUB extraction completed in {extraction_time:.2f}s")
        self.logger.info(f"Extracted {len(chapters)} chapters")

        if chapters:
            total_chars = sum(ch['character_count'] for ch in chapters)
            total_tokens = sum(ch['estimated_tokens'] for ch in chapters)
            self.logger.info(f"Total content: {total_chars:,} characters, ~{total_tokens:,} tokens")

        return chapters

    def chunk_text_by_words(self, text: str, max_words: int = 1700) -> List[str]:
        """Chunk text by word count while maintaining complete sentences."""
        if self.count_words(text) <= max_words:
            return [text]

        chunks = []

        # Split by paragraphs first to maintain structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')

        current_chunk = ""
        current_word_count = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_words = self.count_words(paragraph)

            # If adding this paragraph would exceed the limit
            if current_word_count + paragraph_words > max_words and current_chunk:
                # Complete any partial sentence in current chunk
                current_chunk = self.complete_sentence(current_chunk)
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_word_count = paragraph_words
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_word_count += paragraph_words

        # Handle the last chunk
        if current_chunk.strip():
            current_chunk = self.complete_sentence(current_chunk)
            chunks.append(current_chunk.strip())

        # If any chunk is still too long, split by sentences
        final_chunks = []
        for chunk in chunks:
            if self.count_words(chunk) > max_words * 1.5:  # Allow 50% overflow
                sentence_chunks = self.split_by_sentences_with_word_limit(chunk, max_words)
                final_chunks.extend(sentence_chunks)
            else:
                final_chunks.append(chunk)

        self.logger.info(f"Text chunked into {len(final_chunks)} pieces (target: {max_words} words each)")
        return final_chunks

    def complete_sentence(self, text: str) -> str:
        """Ensure text ends with a complete sentence."""
        # Find the last sentence-ending punctuation
        sentence_endings = ['.', '!', '?', '."', '!"', '?"']

        for ending in sentence_endings:
            last_pos = text.rfind(ending)
            if last_pos != -1:
                return text[:last_pos + len(ending)]

        return text  # Return as-is if no sentence ending found

    def split_by_sentences_with_word_limit(self, text: str, max_words: int) -> List[str]:
        """Split text by sentences while respecting word limits."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""
        current_word_count = 0

        for sentence in sentences:
            sentence_words = self.count_words(sentence)

            if current_word_count + sentence_words > max_words and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_word_count = sentence_words
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_word_count += sentence_words

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_text_reliably(self, text: str, max_tokens: int) -> List[str]:
        """Reliably chunk text to avoid content loss."""
        initial_tokens = self.count_tokens(text)

        if initial_tokens <= max_tokens:
            return [text]

        chunks = []

        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')
        if len(paragraphs) == 1:
            paragraphs = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            test_chunk = current_chunk
            if test_chunk:
                test_chunk += "\n\n" + paragraph
            else:
                test_chunk = paragraph

            if self.count_tokens(test_chunk) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If single paragraph is too long, split by sentences
                if self.count_tokens(paragraph) > max_tokens:
                    sub_chunks = self.split_by_sentences(paragraph, max_tokens)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = paragraph
            else:
                current_chunk = test_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        self.logger.info(f"Text chunked into {len(chunks)} pieces")
        return chunks

    def split_by_sentences(self, text: str, max_tokens: int) -> List[str]:
        """Split text by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence

            if self.count_tokens(test_chunk) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Split by words as last resort
                    word_chunks = self.split_by_words(sentence, max_tokens)
                    chunks.extend(word_chunks[:-1])
                    current_chunk = word_chunks[-1] if word_chunks else ""
            else:
                current_chunk = test_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def split_by_words(self, text: str, max_tokens: int) -> List[str]:
        """Split text by words as last resort."""
        words = text.split()
        chunks = []
        current_chunk = ""

        for word in words:
            test_chunk = current_chunk + (" " if current_chunk else "") + word

            if self.count_tokens(test_chunk) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    chunks.append(word)
                    current_chunk = ""
            else:
                current_chunk = test_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def create_chunked_json(self, chapters: List[Dict], output_path: str, chunk_by_words: bool = True, max_words: int = 1700) -> str:
        """Create JSON file with chunked chapters."""
        self.logger.info(f"Creating chunked JSON file (chunk by {'words' if chunk_by_words else 'tokens'})...")

        chunked_data = {}
        total_chunks = 0

        for chapter in chapters:
            chapter_num = str(chapter['number'])
            content = chapter['content']
            title = chapter['title']

            self.logger.info(f"Chunking Chapter {chapter_num}: '{title}'")

            if chunk_by_words:
                chunks = self.chunk_text_by_words(content, max_words)
            else:
                chunks = self.chunk_text_reliably(content, self.max_tokens)

            chunked_data[chapter_num] = {
                'title': title,
                'chunks': chunks
            }

            total_chunks += len(chunks)
            chunk_info = f"~{max_words} words" if chunk_by_words else f"~{self.max_tokens} tokens"
            self.logger.info(f"Chapter {chapter_num} split into {len(chunks)} chunks ({chunk_info} each)")

        # Add metadata
        metadata = {
            'total_chapters': len(chapters),
            'total_chunks': total_chunks,
            'chunking_method': 'words' if chunk_by_words else 'tokens',
            'max_words_per_chunk': max_words if chunk_by_words else None,
            'max_tokens_per_chunk': self.max_tokens if not chunk_by_words else None,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        output_data = {
            'metadata': metadata,
            'chapters': chunked_data
        }

        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        self.logger.info(f"JSON file created: {output_path} ({file_size:.2f} MB)")
        self.logger.info(f"Total: {len(chapters)} chapters, {total_chunks} chunks")

        return output_path

    def process_epub_to_json(self, epub_path: str, json_output_path: str, chunk_by_words: bool = True, max_words: int = 1700) -> str:
        """Main method to extract EPUB and create chunked JSON."""
        self.logger.info("="*80)
        self.logger.info("STARTING EPUB TO JSON CONVERSION")
        self.logger.info("="*80)

        chapters = self.extract_chapters_from_epub(epub_path)

        if not chapters:
            raise ValueError("No chapters found in EPUB file")

        json_file = self.create_chunked_json(chapters, json_output_path, chunk_by_words, max_words)

        self.logger.info("="*80)
        self.logger.info("EPUB TO JSON CONVERSION COMPLETED")
        self.logger.info("="*80)

        return json_file


def main():
    """Main function to run the EPUB to JSON conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert EPUB to chunked JSON for TTS processing")
    parser.add_argument("epub_path", help="Path to EPUB file")
    parser.add_argument("output_path", help="Output path for JSON file")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Set logging level")
    parser.add_argument("--log-file", help="Optional log file path")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Analyze EPUB structure and exit")
    parser.add_argument("--chunk-by-tokens", action="store_true",
                       help="Chunk by token count instead of words")
    parser.add_argument("--max-words", type=int, default=1700,
                       help="Maximum words per chunk (default: 1700)")

    args = parser.parse_args()

    # Handle chunking method
    chunk_by_words = not args.chunk_by_tokens

    # Setup file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(getattr(logging, args.log_level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger = logging.getLogger("EPUBProcessor")
        logger.addHandler(file_handler)

    try:
        processor = EPUBProcessor(log_level=args.log_level)

        if args.analyze_only:
            chapters = processor.extract_chapters_from_epub(args.epub_path)

            print(f"\n📊 EPUB Analysis Results:")
            print(f"📚 Found {len(chapters)} chapters")

            if chapters:
                for ch in chapters:
                    print(f"  Chapter {ch['number']}: '{ch['title']}' ({ch['character_count']} chars)")
            else:
                print("❌ No chapters found.")
        else:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

            json_file = processor.process_epub_to_json(
                args.epub_path, args.output_path, chunk_by_words, args.max_words
            )

            print(f"\n🎉 JSON extraction completed!")
            print(f"📁 JSON file: {json_file}")

    except KeyboardInterrupt:
        print("\n❌ Conversion interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
