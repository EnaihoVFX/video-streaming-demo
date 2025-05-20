import os
import re
import json
import time
import random
import logging
import subprocess
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import whisper
import ffmpeg
import torch
from transformers import pipeline
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clipper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedYouTubeClipper:
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the clipper with configuration"""
        self.load_config(config_path)
        self.setup_directories()
        self.initialize_models()
        self.setup_http_session()
        self.processed_videos = set()
        self.running = True
        self.load_gameplay_footage()
        
    def load_config(self, config_path: str) -> None:
        """Load and validate configuration"""
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Required configurations
            self.output_dir = config['output_dir']
            self.download_dir = config['download_dir']
            
            # Enhanced configurations with defaults
            self.scan_interval = config.get('scan_interval_minutes', 60) * 60  # Convert to seconds
            self.min_clip_length = config.get('min_clip_length', 8)
            self.max_clip_length = config.get('max_clip_length', 45)
            self.max_video_length = config.get('max_video_length', 600)  # 10 minutes
            self.default_prompts = config.get('prompts', [
                "most exciting moment",
                "funniest part",
                "emotional highlight",
                "controversial statement"
            ])
            self.trending_urls = config.get('trending_urls', [
                "https://www.youtube.com/feed/trending",
                "https://www.youtube.com/feed/trending?bp=4gINGgt5dG1hX2NoYXJ0cw%3D%3D"  # Music trending
            ])
            self.gameplay_footage = config.get('gameplay_footage', [])
            self.layout_rules = config.get('layout_rules', {
                "tutorial": "side_by_side",
                "review": "picture_in_picture",
                "gaming": "fullscreen",
                "interview": "split_screen"
            })
            
            # Video processing settings
            self.video_format = config.get('video_format', 'best[height<=720][fps<=30]')
            self.max_retries = config.get('max_retries', 3)
            self.retry_delay = config.get('retry_delay', 5)
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def setup_directories(self) -> None:
        """Create necessary directories with error handling"""
        try:
            os.makedirs(self.download_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'clips'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'compilations'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
            logger.info("Directories setup complete")
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            raise

    def initialize_models(self) -> None:
        """Initialize AI models with error handling and progress logging"""
        try:
            logger.info("Initializing NLP models...")
            self.nlp = pipeline(
                "text-classification", 
                model="distilbert-base-uncased",
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            self.sentiment = pipeline(
                "sentiment-analysis",
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            
            logger.info("Initializing Whisper model...")
            self.model = whisper.load_model(
                "base",
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            logger.info("All models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def setup_http_session(self) -> None:
        """Configure HTTP session with retries and timeouts"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        # Configure retry strategy
        retry_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('http://', retry_adapter)
        self.session.mount('https://', retry_adapter)

    def load_gameplay_footage(self) -> None:
        """Preload and validate gameplay footage"""
        self.available_gameplay = []
        for footage in self.gameplay_footage:
            if os.path.exists(footage):
                self.available_gameplay.append(footage)
            else:
                logger.warning(f"Gameplay footage not found: {footage}")
        
        if not self.available_gameplay:
            logger.info("No valid gameplay footage available")

    def get_trending_videos(self) -> List[Dict]:
        """Scrape trending videos with improved parsing and error handling"""
        videos = []
        
        for url in self.trending_urls:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Fetching trending videos from {url} (attempt {attempt + 1})")
                    response = self.session.get(url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    video_elements = soup.select('ytd-video-renderer, ytd-compact-video-renderer')
                    
                    if not video_elements:
                        logger.warning("No video elements found - page structure may have changed")
                        continue
                        
                    for element in video_elements[:10]:  # Limit to top 10 per page
                        try:
                            title = element.select_one('#video-title').get_text(strip=True)
                            video_url = 'https://www.youtube.com' + element.select_one('#video-title')['href'].split('&')[0]
                            video_id = re.search(r'v=([a-zA-Z0-9_-]+)', video_url).group(1)
                            channel = element.select_one('.ytd-channel-name a').get_text(strip=True)
                            
                            # Estimate category from URL or title
                            category = self.estimate_video_category(video_url, title)
                            
                            videos.append({
                                'id': video_id,
                                'title': title,
                                'channel': channel,
                                'url': video_url,
                                'category': category,
                                'duration': None
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error parsing video element: {e}")
                            continue
                    
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to fetch trending videos from {url} after {self.max_retries} attempts")
                    time.sleep(self.retry_delay)
        
        return videos

    def estimate_video_category(self, url: str, title: str) -> str:
        """Estimate video category based on URL and title patterns"""
        title_lower = title.lower()
        
        # Check URL patterns first
        if "music" in url.lower() or "artist" in url.lower():
            return "music"
        if "gaming" in url.lower():
            return "gaming"
            
        # Check title keywords
        gaming_keywords = ["gameplay", "walkthrough", "let's play", "speedrun"]
        tutorial_keywords = ["how to", "tutorial", "guide", "step by step"]
        
        if any(kw in title_lower for kw in gaming_keywords):
            return "gaming"
        if any(kw in title_lower for kw in tutorial_keywords):
            return "tutorial"
        if "interview" in title_lower:
            return "interview"
        if "review" in title_lower:
            return "review"
            
        return "entertainment"  # Default category

    def get_video_duration(self, video_id: str) -> Optional[float]:
        """Get video duration with improved reliability"""
        for attempt in range(self.max_retries):
            try:
                url = f"https://www.youtube.com/watch?v={video_id}"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                # Try multiple patterns to find duration
                patterns = [
                    r'"approxDurationMs":"(\d+)"',  # Standard pattern
                    r'"lengthSeconds":"(\d+)"',     # Alternate pattern
                    r'itemprop="duration".*?content="PT(\d+)S"'  # HTML5 pattern
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, response.text)
                    if match:
                        return float(match.group(1))  # Already in seconds or converted
                
                logger.warning(f"Duration not found for video {video_id}")
                return None
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to get duration failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to get duration after {self.max_retries} attempts")
                time.sleep(self.retry_delay)
        
        return None

    def download_video(self, video_id: str) -> Optional[str]:
        """Download video with improved error handling and progress tracking"""
        output_path = os.path.join(self.download_dir, f"{video_id}.mp4")
        
        if os.path.exists(output_path):
            logger.info(f"Video already downloaded: {video_id}")
            return output_path
            
        # Check duration first
        duration = self.get_video_duration(video_id)
        if duration and duration > self.max_video_length:
            logger.info(f"Skipping long video ({duration}s): {video_id}")
            return None
            
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading video {video_id} (attempt {attempt + 1})")
                
                cmd = [
                    'yt-dlp',
                    '-f', self.video_format,
                    '-o', output_path,
                    '--no-continue',
                    '--no-playlist',
                    '--throttled-rate', '1M',  # Limit download speed
                    '--retries', '3',
                    url
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Monitor progress
                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logger.debug(output.strip())
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
                
                logger.info(f"Successfully downloaded video: {video_id}")
                return output_path
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Download failed (attempt {attempt + 1}): {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download video after {self.max_retries} attempts")
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected download error: {e}")
                return None
        
        return None

    def transcribe_video(self, video_path: str) -> Tuple[str, List[Dict]]:
        """Transcribe video with progress tracking and error handling"""
        try:
            logger.info(f"Starting transcription for {video_path}")
            
            # Use faster whisper options for base model
            result = self.model.transcribe(
                video_path,
                fp16=False,  # Disable for CPU/MPS
                verbose=False,
                word_timestamps=True
            )
            
            logger.info(f"Completed transcription for {video_path}")
            
            # Add sentiment analysis to segments
            for segment in result['segments']:
                try:
                    segment['sentiment'] = self.sentiment(segment['text'])[0]
                    segment['excitement'] = self.calculate_excitement_score(segment['text'])
                except Exception as e:
                    logger.warning(f"Failed to analyze segment: {e}")
                    segment['sentiment'] = {'label': 'NEUTRAL', 'score': 0.5}
                    segment['excitement'] = 0.5
            
            return result['text'], result['segments']
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def calculate_excitement_score(self, text: str) -> float:
        """Calculate a more sophisticated excitement score"""
        excitement_words = {
            'wow': 1.0, 'amazing': 0.9, 'incredible': 0.9, 'unbelievable': 0.8,
            'awesome': 0.8, 'epic': 0.7, 'crazy': 0.7, 'insane': 0.7,
            'mind-blowing': 0.9, 'spectacular': 0.8, 'holy cow': 0.9,
            'omg': 0.8, 'oh my god': 0.8, 'no way': 0.7
        }
        
        words = text.lower().split()
        score = 0.0
        max_words = min(len(words), 100)  # Limit to first 100 words
        
        for word in words[:max_words]:
            if word in excitement_words:
                score += excitement_words[word]
        
        # Normalize score (0-1 range)
        normalized_score = min(score / 5, 1.0)
        
        # Add bonus for exclamation marks
        exclamation_bonus = min(text.count('!') * 0.05, 0.2)
        
        return min(normalized_score + exclamation_bonus, 1.0)

    def find_best_segments(self, segments: List[Dict], prompt: Optional[str] = None) -> List[Dict]:
        """Improved segment finding with hybrid approach"""
        if prompt:
            return self.find_prompted_segments(segments, prompt)
        return self.find_auto_segments(segments)

    def find_prompted_segments(self, segments: List[Dict], prompt: str) -> List[Dict]:
        """Enhanced prompt-based segment finding"""
        try:
            prompt_type = self.nlp(prompt)[0]['label']
            
            relevant_segments = []
            for seg in segments:
                score = 0.0
                text = seg['text'].lower()
                
                # Score based on prompt type
                if prompt_type == "funny":
                    funny_words = ['laugh', 'funny', 'joke', 'hilarious', 'comedy']
                    score += 0.5 * sum(text.count(word) for word in funny_words)
                    if seg['sentiment']['label'] == "POSITIVE":
                        score += 0.3
                
                elif prompt_type == "exciting":
                    score += seg['excitement']
                    if seg['sentiment']['label'] == "POSITIVE":
                        score += 0.2
                
                elif prompt_type == "emotional":
                    emotional_words = ['cry', 'tears', 'heart', 'love', 'sad']
                    score += 0.4 * sum(text.count(word) for word in emotional_words)
                    if seg['sentiment']['score'] > 0.7:  # Strong sentiment
                        score += 0.3
                
                elif prompt_type == "controversial":
                    controversial_phrases = ['but', 'however', 'although', 'disagree']
                    score += 0.5 * sum(text.count(phrase) for phrase in controversial_phrases)
                    if seg['sentiment']['label'] == "NEGATIVE":
                        score += 0.2
                
                # Duration scoring
                duration = seg['end'] - seg['start']
                if self.min_clip_length <= duration <= self.max_clip_length:
                    duration_score = 1 - abs(duration - (self.min_clip_length + self.max_clip_length)/2) / 10
                    score += 0.2 * duration_score
                
                if score > 0.5:  # Threshold for relevance
                    seg['relevance_score'] = score
                    relevant_segments.append(seg)
            
            # Return top 3 segments by relevance
            return sorted(relevant_segments, key=lambda x: x['relevance_score'], reverse=True)[:3]
            
        except Exception as e:
            logger.error(f"Prompted segment finding failed: {e}")
            return []

    def find_auto_segments(self, segments: List[Dict]) -> List[Dict]:
        """Improved automatic segment finding with multiple factors"""
        scored_segments = []
        
        for seg in segments:
            try:
                duration = seg['end'] - seg['start']
                
                # Skip if duration is outside desired range
                if not (self.min_clip_length <= duration <= self.max_clip_length):
                    continue
                
                # Composite score calculation
                score = (
                    0.4 * seg['excitement'] +  # Excitement level
                    0.3 * seg['sentiment']['score'] * (1 if seg['sentiment']['label'] == "POSITIVE" else -0.5) +
                    0.2 * min(len(seg['text'].split()) / 30, 1) +  # Optimal text length
                    0.1 * random.uniform(0.8, 1.0)  # Randomness factor
                )
                
                scored_segments.append({
                    'segment': seg,
                    'score': score,
                    'duration': duration
                })
                
            except Exception as e:
                logger.warning(f"Error scoring segment: {e}")
                continue
        
        # Sort by score and return top segments
        scored_segments.sort(key=lambda x: x['score'], reverse=True)
        return [item['segment'] for item in scored_segments[:5]]

    def create_enhanced_clip(self, video_path: str, segments: List[Dict], video_meta: Dict) -> Optional[str]:
        """Create professional clip with multiple enhancements"""
        try:
            logger.info(f"Creating enhanced clip from {video_path}")
            
            # Load base video
            base_clip = VideoFileClip(video_path)
            clips = []
            
            for i, seg in enumerate(segments):
                try:
                    # Extract segment
                    clip = base_clip.subclip(seg['start'], seg['end'])
                    
                    # Add captions with improved styling
                    if random.random() > 0.3:  # 70% chance of captions
                        caption_text = self.clean_caption_text(seg['text'])
                        caption = TextClip(
                            caption_text,
                            fontsize=28,
                            color='white',
                            font='Arial-Bold',
                            stroke_color='black',
                            stroke_width=1.5,
                            size=(base_clip.w * 0.9, None),
                            method='caption',
                            align='center'
                        ).set_position(('center', 'bottom-50')).set_duration(seg['end'] - seg['start'])
                        
                        # Add semi-transparent background for better readability
                        caption_bg = caption.on_color(
                            size=(caption.w + 20, caption.h + 10),
                            color=(0, 0, 0),
                            pos=('center', 'center'),
                            col_opacity=0.6
                        )
                        
                        clip = CompositeVideoClip([clip, caption_bg])
                    
                    # Add transition effects
                    if i > 0:
                        clips[-1] = clips[-1].fx(transfx.crossfadeout, 0.5)
                        clip = clip.fx(transfx.crossfadein, 0.5)
                    
                    clips.append(clip)
                    
                except Exception as e:
                    logger.warning(f"Error processing segment {i}: {e}")
                    continue
            
            if not clips:
                logger.error("No valid segments to create clip")
                return None
            
            # Combine clips with improved composition
            final_clip = concatenate_videoclips(clips, method="compose", padding=-0.5)
            
            # Apply layout based on video category
            layout = self.layout_rules.get(video_meta['category'].lower().split()[0], "fullscreen")
            final_clip = self.apply_layout(final_clip, layout, video_meta)
            
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir, 
                'clips', 
                f"{video_meta['id']}_{timestamp}.mp4"
            )
            
            # Write with optimized settings
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate='5000k',
                threads=4,
                fps=24,
                preset='fast',
                logger=None  # Disable moviepy's logger
            )
            
            logger.info(f"Successfully created clip: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Clip creation failed: {e}")
            return None
        finally:
            if 'base_clip' in locals():
                base_clip.close()
            if 'final_clip' in locals():
                final_clip.close()

    def clean_caption_text(self, text: str, max_length: int = 50) -> str:
        """Clean and format caption text for display"""
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Shorten if needed
        if len(text) > max_length:
            words = text.split()
            shortened = []
            char_count = 0
            
            for word in words:
                if char_count + len(word) + 1 <= max_length - 3:  # Account for ellipsis
                    shortened.append(word)
                    char_count += len(word) + 1
                else:
                    shortened.append("...")
                    break
            
            text = ' '.join(shortened)
        
        return text

    def apply_layout(self, clip, layout: str, video_meta: Dict):
        """Apply different layouts to the final clip"""
        if layout == "side_by_side" and self.available_gameplay:
            try:
                gameplay_path = random.choice(self.available_gameplay)
                gameplay_clip = VideoFileClip(gameplay_path)
                
                # Match duration
                if gameplay_clip.duration > clip.duration:
                    gameplay_clip = gameplay_clip.subclip(0, clip.duration)
                else:
                    # Loop gameplay if shorter than clip
                    gameplay_clip = gameplay_clip.loop(duration=clip.duration)
                
                # Resize both clips
                main_clip = clip.resize(width=640)
                gameplay_clip = gameplay_clip.resize(width=640)
                
                # Combine side by side
                return clips_array([[main_clip, gameplay_clip]])
                
            except Exception as e:
                logger.warning(f"Failed to apply side-by-side layout: {e}")
                return clip
        
        elif layout == "picture_in_picture" and self.available_gameplay:
            try:
                gameplay_path = random.choice(self.available_gameplay)
                gameplay_clip = VideoFileClip(gameplay_path)
                
                # Match duration and resize
                if gameplay_clip.duration > clip.duration:
                    gameplay_clip = gameplay_clip.subclip(0, clip.duration)
                else:
                    gameplay_clip = gameplay_clip.loop(duration=clip.duration)
                
                gameplay_clip = gameplay_clip.resize(width=clip.w // 3)
                
                # Position in bottom-right corner
                pip_clip = gameplay_clip.set_position((
                    clip.w - gameplay_clip.w - 20,
                    clip.h - gameplay_clip.h - 20
                ))
                
                return CompositeVideoClip([clip, pip_clip])
                
            except Exception as e:
                logger.warning(f"Failed to apply PiP layout: {e}")
                return clip
        
        return clip  # Default to fullscreen

    def process_video(self, video: Dict) -> List[str]:
        """Process a single video end-to-end"""
        created_clips = []
        
        try:
            logger.info(f"Starting processing for video: {video['title']}")
            
            # Download video
            video_path = self.download_video(video['id'])
            if not video_path:
                return created_clips
            
            # Get duration from downloaded file
            try:
                with VideoFileClip(video_path) as temp_clip:
                    video['duration'] = temp_clip.duration
            except Exception as e:
                logger.warning(f"Couldn't get duration: {e}")
                video['duration'] = None
            
            # Transcribe and analyze
            full_text, segments = self.transcribe_video(video_path)
            
            # Process with default prompts
            for prompt in self.default_prompts:
                prompted_segments = self.find_prompted_segments(segments, prompt)
                if prompted_segments:
                    clip_path = self.create_enhanced_clip(video_path, prompted_segments, video)
                    if clip_path:
                        created_clips.append(clip_path)
                        logger.info(f"Created prompted clip: {clip_path}")
            
            # Process with automatic selection
            auto_segments = self.find_auto_segments(segments)
            if auto_segments:
                clip_path = self.create_enhanced_clip(video_path, auto_segments, video)
                if clip_path:
                    created_clips.append(clip_path)
                    logger.info(f"Created auto-generated clip: {clip_path}")
            
            # Clean up
            os.remove(video_path)
            logger.info(f"Finished processing video: {video['title']}")
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
        
        return created_clips

    def continuous_processing(self) -> None:
        """Main processing loop with improved scheduling and error handling"""
        logger.info("Starting continuous processing service")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Get trending videos
                videos = self.get_trending_videos()
                if not videos:
                    logger.warning("No trending videos found")
                    time.sleep(self.retry_delay)
                    continue
                
                # Process new videos
                new_videos = [v for v in videos if v['id'] not in self.processed_videos]
                
                if not new_videos:
                    logger.info("No new videos to process")
                else:
                    logger.info(f"Found {len(new_videos)} new videos to process")
                    
                    for video in new_videos:
                        if not self.running:  # Check if we should stop
                            break
                            
                        created_clips = self.process_video(video)
                        if created_clips:
                            self.processed_videos.add(video['id'])
                
                # Calculate sleep time accounting for processing duration
                processing_time = time.time() - start_time
                sleep_time = max(0, self.scan_interval - processing_time)
                
                if sleep_time > 0:
                    logger.info(f"Next check in {sleep_time/60:.1f} minutes")
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt - shutting down")
                self.running = False
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(min(60, self.scan_interval))  # Wait before retrying
        
        logger.info("Processing service stopped")

if __name__ == "__main__":
    try:
        clipper = EnhancedYouTubeClipper()
        clipper.continuous_processing()
    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)