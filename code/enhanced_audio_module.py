"""
Enhanced Audio Module integrating with Enhanced TTS Manager

This module provides backward compatibility with existing AudioProcessor
while adding advanced features like voice cloning, emotion control, and provider switching.

Key Features:
- Drop-in replacement for existing AudioProcessor
- Enhanced voice quality with emotion support
- Voice cloning from user samples
- Provider switching and A/B testing
- Quality assessment and optimization
- Real-time voice customization

Author: Enhanced by Claude Code Assistant for RealtimeVoiceChat project
"""

import asyncio
import logging
import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Callable, Optional, Generator, Dict, Any
import struct

from enhanced_tts_manager import (
    EnhancedTTSManager, TTSProvider, TTSConfig, Emotion, VoiceProfile
)
from colors import Colors

logger = logging.getLogger(__name__)

class Silence:
    """Silence timing configuration for different TTS engines"""
    def __init__(self, comma: float = 0.3, sentence: float = 0.6, default: float = 0.3):
        self.comma = comma
        self.sentence = sentence
        self.default = default

ENGINE_SILENCES = {
    "coqui_xtts_v2": Silence(comma=0.3, sentence=0.6, default=0.3),
    "elevenlabs": Silence(comma=0.4, sentence=0.7, default=0.4),
    "azure_speech": Silence(comma=0.3, sentence=0.6, default=0.3),
}

# Enhanced configuration with new features
DEFAULT_TTS_CONFIG = {
    "provider": TTSProvider.COQUI_XTTS_V2,
    "model": "multilingual-v2-v1",
    "quality": "balanced",  # balanced, fast, max_quality
    "latency_mode": "balanced",  # real_time, low_latency, high_quality
    "enable_emotion": True,
    "enable_cloning": True,
    "cache_size": 50,
    "default_emotion": Emotion.NEUTRAL,
    "voice_speed": 1.0,
    "voice_pitch": 1.0,
    "voice_profile": None,
}

class EnhancedAudioProcessor:
    """
    Enhanced Audio Processor with multi-provider TTS support, voice cloning, and emotion control.

    This class provides backward compatibility with AudioProcessor interface while
    adding advanced features for improved voice quality and customization.
    """

    def __init__(
            self,
            engine: str = "coqui_xtts_v2",
            orpheus_model: Optional[str] = None,
            config_overrides: Optional[Dict[str, Any]] = None,
            quality_mode: str = "balanced",
            enable_cloning: bool = True,
            enable_emotions: bool = True,
            custom_voice_profile: Optional[VoiceProfile] = None,
    ) -> None:
        """
        Initialize Enhanced Audio Processor with advanced TTS capabilities.

        Args:
            engine: TTS provider/engine name (backward compatibility)
            orpheus_model: Legacy parameter for backward compatibility
            config_overrides: Dictionary to override default configuration
            quality_mode: Quality preset ('fast', 'balanced', 'max_quality')
            enable_cloning: Enable voice cloning features
            enable_emotions: Enable emotion support
            custom_voice_profile: Custom voice profile to use
        """
        logger.info(f"🎙️ Initializing Enhanced Audio Processor with engine: {engine}")

        # Configuration management
        self.config = DEFAULT_TTS_CONFIG.copy()
        if config_overrides:
            self.config.update(config_overrides)

        # Map legacy engine names to new providers
        self.engine_name = self._map_legacy_engine(engine)
        self.quality_mode = quality_mode
        self.enable_cloning = enable_cloning
        self.enable_emotions = enable_emotions

        # Backward compatibility
        self.engine_name = engine
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.audio_chunks = asyncio.Queue()

        # Enhanced TTS Manager
        self.tts_manager = EnhancedTTSManager()
        self.initialization_complete = False
        self.first_audio_chunk_time = None
        self.audio_chunk_times = []
        self.current_voice_profile = custom_voice_profile

        # Quality tracking
        self.quality_metrics = {
            "ttft": None,  # Time to First Token
            "audio_latency": [],
            "total_synthesis_time": [],
            "cache_hit_rate": 0,
            "total_synthesis_count": 0,
            "cache_hits": 0
        }

        # Emotion detection (if enabled)
        self.emotion_history = []
        self.current_emotion = Emotion.NEUTRAL

        # Voice cloning state
        self.cloning_samples = []
        self.currently_cloning = False

        # Initialize TTS manager
        self._initialize_tts_manager()

    def _map_legacy_engine(self, engine: str) -> TTSProvider:
        """Map legacy engine names to new TTSProvider enum"""
        mapping = {
            "coqui": TTSProvider.COQUI_XTTS_V2,
            "kokoro": TTSProvider.COQUI_XTTS_V2,  # Use Coqui as default for Kokoro
            "orpheus": TTSProvider.COQUI_XTTS_V2,  # Use Coqui as default for Orpheus
            "coqui_xtts_v2": TTSProvider.COQUI_XTTS_V2,
            "elevenlabs": TTSProvider.ELEVENLABS,
            "azure": TTSProvider.AZURE_SPEECH,
            "azure_speech": TTSProvider.AZURE_SPEECH,
            "openai": TTSProvider.OPENAI_TTS,
            "openai_tts": TTSProvider.OPENAI_TTS,
        }
        return mapping.get(engine.lower(), TTSProvider.COQUI_XTTS_V2)

    def _initialize_tts_manager(self):
        """Initialize the enhanced TTS manager"""
        try:
            # Create TTS configuration
            tts_config = TTSConfig(
                provider=self.config["provider"],
                model=self.config.get("model"),
                voice_profile=self.current_voice_profile,
                emotion=self.config.get("default_emotion", Emotion.NEUTRAL),
                speed=self.config.get("voice_speed", 1.0),
                quality=self.quality_mode,
                latency_mode=self.config["latency_mode"],
                enable_cloning=self.enable_cloning,
                enable_emotion=self.enable_emotions,
                cache_size=self.config["cache_size"]
            )

            # Initialize the TTS manager
            init_success = asyncio.run(self.tts_manager.initialize(tts_config))
            if not init_success:
                logger.error("❌ Failed to initialize TTS manager")
                return

            self.initialization_complete = True
            logger.info("✅ Enhanced TTS Manager initialized successfully")

            # Test synthesis to measure initial latency
            asyncio.run(self._test_initial_latency())

        except Exception as e:
            logger.error(f"💥 Failed to initialize TTS manager: {e}")

    async def _test_initial_latency(self):
        """Test synthesis to measure initial latency"""
        try:
            test_text = "Hello, this is a test of the voice system."
            start_time = time.time()

            # Synthesize test text
            async for _ in self.tts_manager.synthesize(test_text):
                if self.first_audio_chunk_time is None:
                    self.first_audio_chunk_time = time.time() - start_time
                    logger.info(f"🎤️ Initial TTS latency: {self.first_audio_chunk_time:.2f}s")
                break

        except Exception as e:
            logger.error(f"❌ TTS test failed: {e}")

    def set_audio_chunk_callback(
            self,
            callback: Callable[[np.ndarray], None],
            stream_chunk_size: int = 60,
            tld: Optional[float] = None,
            log_synthesis_time: bool = True,
            log_deprecated: bool = True,
            muted=False,
    ) -> None:
        """
        Set callback for audio chunk generation (backward compatibility).

        Args:
            callback: Function to call with audio chunks
            stream_chunk_size: Size of audio chunks to generate
            tld: Time delay (legacy parameter)
            log_synthesis_time: Whether to log synthesis timing
            log_deprecated: Whether to log deprecation warnings
            muted: Whether to mute synthesis
        """
        self.audio_chunk_callback = callback
        self.stream_chunk_size = stream_chunk_size
        self.log_synthesis_time = log_synthesis_time
        self.muted = muted

        if log_deprecated:
            logger.info("📝 Using Enhanced Audio Processor with advanced features")

    async def stream(self,
                   text_stream: Generator[str, None, None],
                   partial_implementation: bool = False,
                   voice_emotion: Optional[Emotion] = None,
                   voice_profile: Optional[VoiceProfile] = None,
                   quality_preset: str = "balanced") -> None:
        """
        Stream text to audio using enhanced TTS capabilities.

        Args:
            text_stream: Generator of text chunks to synthesize
            partial_implementation: Whether this is partial implementation (legacy)
            voice_emotion: Emotion to apply to synthesis
            voice_profile: Voice profile to use for synthesis
            quality_preset: Quality preset for synthesis
        """
        if not self.initialization_complete:
            logger.warning("⚠️ TTS manager not initialized, cannot stream audio")
            return

        # Use provided emotion or current emotion
        target_emotion = voice_emotion or self.current_emotion

        # Use provided voice profile or current profile
        target_voice_profile = voice_profile or self.current_voice_profile

        logger.info(f"🎤️ Starting enhanced audio streaming with emotion: {target_emotion.value if target_emotion else 'neutral'}")

        try:
            # Combine text from stream
            text_buffer = ""
            for chunk in text_stream:
                if chunk:
                    text_buffer += chunk
                    logger.debug(f"📝 Received text chunk: {chunk[:50]}...")

            # Skip if text is empty
            if not text_buffer.strip():
                logger.debug("⏭️ Skipping empty text synthesis")
                return

            # Detect emotion if enabled and no emotion specified
            if self.enable_emotions and not voice_emotion:
                detected_emotion = await self._detect_emotion(text_buffer)
                target_emotion = detected_emotion
                logger.info(f"🎭 Detected emotion: {detected_emotion.value}")

            # Synthesize text with enhanced TTS
            synthesis_start = time.time()
            first_chunk = True
            total_samples = 0

            async for audio_chunk in self.tts_manager.synthesize(
                text=text_buffer,
                emotion=target_emotion,
                voice_profile=target_voice_profile,
                quality_preset=quality_preset
            ):
                if self.stop_event.is_set():
                    logger.info("⏹️ Audio synthesis stopped by event")
                    break

                if first_chunk:
                    self.first_audio_chunk_time = time.time() - synthesis_start
                    first_chunk = False
                    logger.info(f"🎤️ First audio chunk in {self.first_audio_chunk_time:.2f}s")

                # Process audio chunk
                chunk_start = time.time()
                chunk_size = audio_chunk.shape[0]
                total_samples += chunk_size

                # Send to audio chunks queue
                try:
                    self.audio_chunks.put_nowait(audio_chunk)
                    if hasattr(self, 'audio_chunk_callback') and self.audio_chunk_callback:
                        self.audio_chunk_callback(audio_chunk)
                except asyncio.QueueFull:
                    logger.warning("⚠️ Audio chunks queue is full, dropping chunk")

                # Track quality metrics
                self.quality_metrics["audio_latency"].append(time.time() - chunk_start)

            # Update quality metrics
            synthesis_time = time.time() - synthesis_start
            self.quality_metrics["total_synthesis_time"].append(synthesis_time)
            self.quality_metrics["total_synthesis_count"] += 1

            logger.info(f"✅ Completed audio synthesis: {len(text_buffer)} chars, {total_samples} samples, {synthesis_time:.2f}s")

        except Exception as e:
            logger.error(f"💥 Enhanced audio streaming failed: {e}")

        finally:
            self.finished_event.set()
            logger.debug("🏁 Enhanced audio streaming finished")

    async def _detect_emotion(self, text: str) -> Emotion:
        """
        Detect emotion from text using keyword analysis.

        This is a simple implementation using keyword matching.
        In production, you might want to use a more sophisticated NLP model.
        """
        if not self.enable_emotions:
            return Emotion.NEUTRAL

        # Simple keyword-based emotion detection
        emotion_keywords = {
            Emotion.HAPPY: ['happy', 'joy', 'excited', 'wonderful', 'great', 'amazing', 'fantastic'],
            Emotion.SAD: ['sad', 'sorry', 'unfortunate', 'disappointed', 'regret', 'unhappy'],
            Emotion.EXCITED: ['exciting', 'awesome', 'fantastic', 'amazing', 'wonderful', 'incredible'],
            Emotion.CALM: ['calm', 'peaceful', 'relaxed', 'tranquil', 'serene', 'gentle'],
            Emotion.ANGRY: ['angry', 'frustrated', 'annoyed', 'upset', 'furious', 'irritated'],
            Emotion.GENTLE: ['gentle', 'soft', 'kind', 'caring', 'warm', 'soothing'],
        }

        text_lower = text.lower()

        # Count emotion matches
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score

        # Return emotion with highest score or NEUTRAL if no matches
        if emotion_scores:
            detected = max(emotion_scores, key=emotion_scores.get)
            self.emotion_history.append(detected)
            return detected

        return Emotion.NEUTRAL

    async def clone_voice_from_sample(self,
                                    reference_audio: np.ndarray,
                                    voice_name: str,
                                    voice_description: Optional[str] = None) -> bool:
        """
        Clone voice from reference audio sample.

        Args:
            reference_audio: Audio sample to clone from
            voice_name: Name for the cloned voice
            voice_description: Optional description of the voice
        """
        if not self.enable_cloning:
            logger.warning("⚠️ Voice cloning is disabled")
            return False

        if self.currently_cloning:
            logger.warning("⚠️ Voice cloning already in progress")
            return False

        try:
            self.currently_cloning = True
            logger.info(f"🎤️ Starting voice cloning: {voice_name}")

            # Validate audio quality
            if not self._validate_audio_quality(reference_audio):
                logger.error("❌ Reference audio quality is insufficient for cloning")
                return False

            # Clone voice using enhanced TTS manager
            cloned_profile = await self.tts_manager.clone_voice_from_sample(
                reference_audio, voice_name
            )

            if cloned_profile:
                self.current_voice_profile = cloned_profile
                logger.info(f"✅ Voice cloning successful: {cloned_profile.name}")
                return True
            else:
                logger.error(f"❌ Voice cloning failed for: {voice_name}")
                return False

        except Exception as e:
            logger.error(f"💥 Voice cloning error: {e}")
            return False

        finally:
            self.currently_cloning = False

    def _validate_audio_quality(self, audio: np.ndarray) -> bool:
        """
        Validate audio quality for voice cloning.

        Args:
            audio: Audio array to validate

        Returns:
            True if audio quality is sufficient, False otherwise
        """
        if audio.size == 0:
            logger.error("❌ Audio is empty")
            return False

        if audio.shape[0] < 16000:  # Less than 1 second at 16kHz
            logger.error("❌ Audio too short for voice cloning (minimum 1 second)")
            return False

        # Check audio levels
        max_level = np.max(np.abs(audio))
        if max_level < 1000:  # Very quiet audio
            logger.warning("⚠️ Audio levels are very low, cloning quality may be poor")
        elif max_level > 30000:  # Potential clipping
            logger.warning("⚠️ Audio may be clipping, cloning quality may be poor")

        # Check for silence
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        if rms < 500:  # Very quiet overall
            logger.warning("⚠️ Audio is very quiet, cloning quality may be poor")

        return True

    def set_emotion(self, emotion: Emotion):
        """Set current emotion for synthesis"""
        self.current_emotion = emotion
        logger.info(f"🎭 Emotion set to: {emotion.value}")

    def set_voice_profile(self, voice_profile: VoiceProfile):
        """Set voice profile for synthesis"""
        self.current_voice_profile = voice_profile
        logger.info(f"🎤️ Voice profile set to: {voice_profile.name}")

    def set_quality_mode(self, mode: str):
        """Set quality mode for synthesis"""
        self.quality_mode = mode
        logger.info(f"⚙️ Quality mode set to: {mode}")

    async def switch_provider(self, new_provider: TTSProvider, config: Optional[TTSConfig] = None):
        """Switch to a different TTS provider"""
        try:
            logger.info(f"🔄 Switching TTS provider to: {new_provider.value}")

            # Create config for new provider
            new_config = config or TTSConfig(
                provider=new_provider,
                quality=self.quality_mode,
                enable_cloning=self.enable_cloning,
                enable_emotion=self.enable_emotions
            )

            # Switch provider
            success = await self.tts_manager.switch_provider(new_provider, new_config)
            if success:
                self.config["provider"] = new_provider
                logger.info(f"✅ Successfully switched to provider: {new_provider.value}")
                return True
            else:
                logger.error(f"❌ Failed to switch to provider: {new_provider.value}")
                return False

        except Exception as e:
            logger.error(f"💥 Provider switching error: {e}")
            return False

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics"""
        if not self.quality_metrics["total_synthesis_count"]:
            return {"message": "No synthesis performed yet"}

        total_synthesis = self.quality_metrics["total_synthesis_count"]
        cache_hits = self.quality_metrics["cache_hits"]
        cache_hit_rate = (cache_hits / total_synthesis) * 100 if total_synthesis > 0 else 0

        avg_latency = np.mean(self.quality_metrics["audio_latency"]) if self.quality_metrics["audio_latency"] else 0
        avg_synthesis_time = np.mean(self.quality_metrics["total_synthesis_time"]) if self.quality_metrics["total_synthesis_time"] else 0

        return {
            "total_synthesis_count": total_synthesis,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "average_latency": f"{avg_latency*1000:.1f}ms",
            "average_synthesis_time": f"{avg_synthesis_time:.2f}s",
            "first_chunk_latency": f"{self.first_audio_chunk_time:.2f}s" if self.first_audio_chunk_time else "N/A",
            "ttft": f"{self.first_audio_chunk_time:.2f}s" if self.first_audio_chunk_time else "N/A"
        }

    def get_available_voices(self) -> list:
        """Get available voices from current provider"""
        if not self.initialization_complete:
            logger.warning("⚠️ TTS manager not initialized")
            return []

        try:
            voices = asyncio.run(self.tts_manager.get_available_voices())
            return [{"name": v.name, "provider": v.provider.value, "is_custom": v.is_custom} for v in voices]
        except Exception as e:
            logger.error(f"❌ Failed to get available voices: {e}")
            return []

    async def cleanup(self):
        """Cleanup enhanced audio processor resources"""
        logger.info("🧹 Cleaning up Enhanced Audio Processor")

        try:
            # Cleanup TTS manager
            if self.initialization_complete:
                await self.tts_manager.cleanup()

            # Clear queues
            while not self.audio_chunks.empty():
                try:
                    self.audio_chunks.get_nowait()
                except Empty:
                    break

            # Reset state
            self.stop_event.set()
            self.finished_event.set()
            self.currently_cloning = False

            # Log final quality metrics
            metrics = self.get_quality_metrics()
            logger.info(f"📊 Final quality metrics: {metrics}")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

    # Backward compatibility methods
    def on_first_audio_chunk_synthesize(self):
        """Legacy callback for first audio chunk"""
        pass

    def on_audio_stream_start(self):
        """Legacy callback for stream start"""
        pass

    def create_silence(self, duration: float, sample_rate: int = 24000) -> np.ndarray:
        """Create silence audio chunk"""
        num_samples = int(duration * sample_rate)
        return np.zeros(num_samples, dtype=np.int16)

    def create_silence_generator(self, duration: float, sample_rate: int = 24000) -> Generator[np.ndarray, None, None]:
        """Generate silence as chunks"""
        total_samples = int(duration * sample_rate)
        chunk_size = 60  # Default chunk size
        remaining = total_samples

        while remaining > 0:
            current_chunk_size = min(chunk_size, remaining)
            yield np.zeros(current_chunk_size, dtype=np.int16)
            remaining -= current_chunk_size

# Backward compatibility factory function
def create_enhanced_audio_processor(engine: str = "coqui_xtts_v2",
                                 **kwargs) -> EnhancedAudioProcessor:
    """
    Factory function to create enhanced audio processor with backward compatibility.

    Args:
        engine: TTS engine name
        **kwargs: Additional arguments for AudioProcessor

    Returns:
        EnhancedAudioProcessor instance
    """
    return EnhancedAudioProcessor(engine=engine, **kwargs)

# Configuration constants for backward compatibility
START_ENGINE = "coqui_xtts_v2"
QUICK_ANSWER_STREAM_CHUNK_SIZE = 8
FINAL_ANSWER_STREAM_CHUNK_SIZE = 30

# Silence timing for backward compatibility
class EngineSilence:
    COMMA = 0.3
    SENTENCE = 0.6
    DEFAULT = 0.3

ENGINE_SILENCES = {
    "coqui_xtts_v2": EngineSilence(),
    "elevenlabs": EngineSilence(),
    "azure_speech": EngineSilence(),
}

if __name__ == "__main__":
    # Test enhanced audio processor
    async def test_enhanced_processor():
        """Test the enhanced audio processor with emotion support"""
        processor = EnhancedAudioProcessor(
            engine="coqui_xtts_v2",
            quality_mode="max_quality",
            enable_emotions=True,
            enable_cloning=True
        )

        # Test with different emotions
        test_texts = [
            ("Hello, this is a happy message!", Emotion.HAPPY),
            ("I'm sorry to hear that sad news.", Emotion.SAD),
            ("This is so exciting and amazing!", Emotion.EXCITED),
            ("Let's stay calm and peaceful.", Emotion.CALM),
        ]

        for text, emotion in test_texts:
            print(f"🎤️ Testing with emotion: {emotion.value}")
            print(f"📝 Text: {text}")

            # Create text stream
            def text_stream():
                yield text

            # Set audio chunk callback
            chunk_times = []
            def audio_callback(chunk):
                chunk_times.append(time.time())
                print(f"🔊 Audio chunk: {len(chunk)} samples")

            processor.set_audio_chunk_callback(audio_callback)

            # Stream audio
            await processor.stream(text_stream(), voice_emotion=emotion)

            # Wait for completion
            await processor.finished_event.wait()

            # Print metrics
            metrics = processor.get_quality_metrics()
            print(f"📊 Metrics: {metrics}\n")

        await processor.cleanup()

    # Run test
    asyncio.run(test_enhanced_processor())