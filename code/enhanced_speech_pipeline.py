"""
Enhanced Speech Pipeline Manager integration with Advanced TTS Features

This module integrates the Enhanced TTS Manager with the existing Speech Pipeline
to provide improved voice quality, emotion support, and voice cloning capabilities.

Key Features:
- Seamless integration with existing SpeechPipelineManager interface
- Emotion-aware synthesis based on conversation context
- Real-time voice switching and A/B testing
- Quality monitoring and automatic provider selection
- Voice cloning from user audio samples
- Advanced audio post-processing

Author: Enhanced by Claude Code Assistant for RealtimeVoiceChat project
"""

import asyncio
import logging
import time
import threading
from queue import Queue
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
import numpy as np

from speech_pipeline_manager import SpeechPipelineManager
from enhanced_audio_module import EnhancedAudioProcessor, create_enhanced_audio_processor
from enhanced_tts_manager import (
    TTSProvider, TTSConfig, Emotion, VoiceProfile
)
from colors import Colors

logger = logging.getLogger(__name__)

class ConversationContext(Enum):
    """Conversation context types for emotion selection"""
    QUESTION = "question"
    ANSWER = "answer"
    GREETING = "greeting"
    FAREWELL = "farewell"
    EXCITEMENT = "excitement"
    EMPATHY = "empathy"
    INSTRUCTION = "instruction"
    GENERAL = "general"

class QualityTier(Enum):
    """Quality tiers for TTS synthesis"""
    ULTRA_FAST = "ultra_fast"      # 100ms latency, lower quality
    FAST = "fast"                   # 150ms latency, good quality
    BALANCED = "balanced"            # 200ms latency, good quality
    HIGH_QUALITY = "high_quality"    # 300ms latency, high quality
    STUDIO = "studio"               # 500ms latency, maximum quality

class EnhancedSpeechPipelineManager:
    """
    Enhanced Speech Pipeline Manager with advanced TTS capabilities and intelligent voice quality.

    This class extends the functionality of SpeechPipelineManager while maintaining
    backward compatibility and adding sophisticated voice synthesis features.
    """

    def __init__(
            self,
            # Existing SpeechPipelineManager parameters (for backward compatibility)
            llm_provider: str = "ollama",
            llm_model: str = "llama3.1:8b-instruct-q8_0",
            tts_engine: str = "coqui_xtts_v2",
            system_prompt: Optional[str] = None,
            no_think: bool = False,
            orpheus_model: Optional[str] = None,
            # Enhanced TTS parameters
            quality_tier: QualityTier = QualityTier.BALANCED,
            enable_emotion_synthesis: bool = True,
            enable_voice_cloning: bool = True,
            enable_provider_switching: bool = True,
            emotion_detection_enabled: bool = True,
            audio_quality_enhancement: bool = True,
            custom_voice_profile: Optional[VoiceProfile] = None,
            tts_config_overrides: Optional[Dict[str, Any]] = None,
            # Performance tuning
            optimization_mode: str = "balanced",  # speed, balanced, quality
            adaptive_quality: bool = True,
            cache_enabled: bool = True,
            realtime_latency_target: float = 200.0,  # ms
        ) -> None:
        """
        Initialize Enhanced Speech Pipeline Manager with advanced voice synthesis features.

        Args:
            llm_provider: LLM provider backend (ollama, openai, etc.)
            llm_model: Specific LLM model to use
            tts_engine: TTS engine to use (enhanced provider names)
            system_prompt: System prompt for LLM
            no_think: Whether to disable LLM thinking
            orpheus_model: Legacy parameter for backward compatibility

            quality_tier: Quality tier for TTS synthesis
            enable_emotion_synthesis: Enable emotion-aware synthesis
            enable_voice_cloning: Enable voice cloning features
            enable_provider_switching: Enable intelligent provider switching
            emotion_detection_enabled: Enable emotion detection from conversation context
            audio_quality_enhancement: Enable advanced audio post-processing
            custom_voice_profile: Custom voice profile to use
            tts_config_overrides: Override configuration for TTS
            optimization_mode: Performance optimization mode
            adaptive_quality: Enable adaptive quality based on latency requirements
            cache_enabled: Enable audio caching for performance
            realtime_latency_target: Target latency for real-time synthesis (ms)
        """
        logger.info(f"🚀 Initializing Enhanced Speech Pipeline Manager")
        logger.info(f"🎤️ TTS Engine: {tts_engine}, Quality: {quality_tier.value}")
        logger.info(f"🧠 LLM Provider: {llm_provider}, Model: {llm_model}")

        # Store enhanced configuration
        self.enhanced_config = {
            "quality_tier": quality_tier,
            "enable_emotion_synthesis": enable_emotion_synthesis,
            "enable_voice_cloning": enable_voice_cloning,
            "enable_provider_switching": enable_provider_switching,
            "emotion_detection_enabled": emotion_detection_enabled,
            "audio_quality_enhancement": audio_quality_enhancement,
            "custom_voice_profile": custom_voice_profile,
            "optimization_mode": optimization_mode,
            "adaptive_quality": adaptive_quality,
            "cache_enabled": cache_enabled,
            "realtime_latency_target": realtime_latency_target,
        }

        # Map enhanced engines to provider configuration
        mapped_engine, mapped_config = self._map_enhanced_engine(tts_engine, tts_config_overrides)

        # Create original SpeechPipelineManager for backward compatibility
        self.original_pipeline = SpeechPipelineManager(
            llm_provider=llm_provider,
            llm_model=llm_model,
            tts_engine=mapped_engine,
            system_prompt=system_prompt,
            no_think=no_think,
            orpheus_model=orpheus_model
        )

        # Create enhanced audio processor
        self.enhanced_audio = create_enhanced_audio_processor(
            engine=mapped_engine,
            config_overrides=mapped_config,
            quality_mode=quality_tier.value,
            enable_cloning=enable_voice_cloning,
            enable_emotions=enable_emotion_synthesis,
            custom_voice_profile=custom_voice_profile,
        )

        # Enhanced state management
        self.conversation_history = []
        self.current_emotion = Emotion.NEUTRAL
        self.current_context = ConversationContext.GENERAL
        self.current_provider = self._map_to_provider(mapped_engine)
        self.quality_history = []
        self.provider_performance = {}
        self.voice_profiles = {}

        # Emotion detection and context analysis
        self.emotion_patterns = self._initialize_emotion_patterns()
        self.context_keywords = self._initialize_context_keywords()

        # Quality monitoring
        self.quality_monitoring = {
            "latency_measurements": [],
            "quality_scores": [],
            "provider_performance": {},
            "cache_hit_rates": {},
            "error_rates": {}
        }

        # Adaptive quality management
        self.adaptive_quality_enabled = adaptive_quality
        self.target_latency_ms = realtime_latency_target
        self.last_quality_adjustment = None

        # Voice cloning state
        self.cloning_session = None
        self.cloning_samples_collected = []
        self.voice_cloning_progress = 0.0

        # Thread safety
        self.emotion_lock = threading.Lock()
        self.quality_lock = threading.Lock()
        self.provider_switching_lock = threading.Lock()

        # Performance optimization flags
        self.performance_mode = optimization_mode
        self.resource_usage_tracking = {}

        logger.info("✅ Enhanced Speech Pipeline Manager initialized successfully")

    def _map_enhanced_engine(self, engine: str, config_overrides: Optional[Dict[str, Any]]) -> tuple:
        """Map enhanced engine names to backward-compatible configurations"""
        engine_mapping = {
            "coqui_xtts_v2": ("coqui", self._get_coqui_config(config_overrides)),
            "elevenlabs": ("coqui", self._get_elevenlabs_config(config_overrides)),
            "azure_speech": ("coqui", self._get_azure_config(config_overrides)),
            "openai_tts": ("coqui", self._get_openai_tts_config(config_overrides)),
        }

        return engine_mapping.get(engine.lower(), (engine, config_overrides or {}))

    def _map_to_provider(self, engine: str) -> TTSProvider:
        """Map engine string to TTSProvider enum"""
        provider_mapping = {
            "coqui": TTSProvider.COQUI_XTTS_V2,
            "coqui_xtts_v2": TTSProvider.COQUI_XTTS_V2,
            "elevenlabs": TTSProvider.ELEVENLABS,
            "azure_speech": TTSProvider.AZURE_SPEECH,
            "azure": TTSProvider.AZURE_SPEECH,
            "openai_tts": TTSProvider.OPENAI_TTS,
            "openai": TTSProvider.OPENAI_TTS,
        }
        return provider_mapping.get(engine.lower(), TTSProvider.COQUI_XTTS_V2)

    def _get_coqui_config(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get Coqui-specific configuration"""
        return {
            "provider": TTSProvider.COQUI_XTTS_V2,
            "model": "multilingual-v2-v1",
            "cache_size": 50,
            "enable_cloning": self.enhanced_config["enable_voice_cloning"],
            "enable_emotion": self.enhanced_config["enable_emotion_synthesis"],
            **(overrides or {})
        }

    def _get_elevenlabs_config(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get ElevenLabs-specific configuration"""
        return {
            "provider": TTSProvider.ELEVENLABS,
            "api_key": overrides.get("api_key") if overrides else None,
            "voice_id": overrides.get("voice_id", "rachel") if overrides else "rachel",
            "cache_size": 30,
            "enable_cloning": True,
            "enable_emotion": True,
            **(overrides or {})
        }

    def _get_azure_config(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get Azure Speech-specific configuration"""
        return {
            "provider": TTSProvider.AZURE_SPEECH,
            "api_key": overrides.get("api_key") if overrides else None,
            "region": overrides.get("region", "eastus") if overrides else "eastus",
            "voice_id": overrides.get("voice_id", "en-US-JennyNeural") if overrides else "en-US-JennyNeural",
            "cache_size": 40,
            "enable_cloning": False,
            "enable_emotion": True,
            **(overrides or {})
        }

    def _get_openai_tts_config(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get OpenAI TTS-specific configuration"""
        return {
            "provider": TTSProvider.OPENAI_TTS,
            "api_key": overrides.get("api_key") if overrides else None,
            "model": overrides.get("model", "tts-1") if overrides else "tts-1",
            "voice_id": overrides.get("voice_id", "alloy") if overrides else "alloy",
            "cache_size": 35,
            "enable_cloning": False,
            "enable_emotion": False,
            **(overrides or {})
        }

    def _initialize_emotion_patterns(self) -> Dict[Emotion, List[str]]:
        """Initialize emotion detection patterns"""
        return {
            Emotion.HAPPY: [
                "happy", "joy", "wonderful", "great", "amazing", "fantastic",
                "excited", "delighted", "pleased", "cheerful", "laugh", "smile"
            ],
            Emotion.SAD: [
                "sad", "sorry", "unfortunate", "disappointed", "regret", "unhappy",
                "sorrow", "grief", "melancholy", "depressed", "cry"
            ],
            Emotion.EXCITED: [
                "exciting", "awesome", "fantastic", "amazing", "wonderful", "incredible",
                "wow", "astonishing", "breathtaking", "thrilling", "spectacular"
            ],
            Emotion.CALM: [
                "calm", "peaceful", "relaxed", "tranquil", "serene", "gentle",
                "soothing", "comfortable", "easy", "restful", "meditative"
            ],
            Emotion.GENTLE: [
                "gentle", "soft", "kind", "caring", "warm", "soothing",
                "tender", "compassionate", "understanding", "supportive", "loving"
            ],
            Emotion.CONFIDENT: [
                "confident", "certain", "sure", "positive", "assured", "decisive",
                "strong", "capable", "competent", "reliable", "professional"
            ],
        }

    def _initialize_context_keywords(self) -> Dict[ConversationContext, List[str]]:
        """Initialize conversation context keywords"""
        return {
            ConversationContext.QUESTION: [
                "what", "why", "how", "when", "where", "who", "which",
                "?", "tell me", "explain", "describe", "ask"
            ],
            ConversationContext.ANSWER: [
                "answer", "response", "reply", "solution", "result",
                "here's", "that's", "this is", "because", "therefore"
            ],
            ConversationContext.GREETING: [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "welcome", "greetings", "how are you", "nice to meet you"
            ],
            ConversationContext.FAREWELL: [
                "goodbye", "bye", "see you", "farewell", "take care",
                "until next time", "have a good day", "good night"
            ],
            ConversationContext.EXCITEMENT: [
                "amazing", "incredible", "wow", "fantastic", "wonderful",
                "breathtaking", "spectacular", "awesome", "brilliant"
            ],
            ConversationContext.EMPATHY: [
                "understand", "empathy", "feel", "sorry to hear", "i understand",
                "that sounds", "i can imagine", "must be", "relate"
            ],
            ConversationContext.INSTRUCTION: [
                "do this", "please", "command", "instruction", "direction",
                "follow", "step", "process", "method", "how to"
            ],
        }

    # Enhanced audio processing with emotion and context awareness
    async def synthesize_with_emotion(self,
                                    text: str,
                                    context: Optional[ConversationContext] = None,
                                    override_emotion: Optional[Emotion] = None,
                                    voice_profile: Optional[VoiceProfile] = None,
                                    quality_tier: Optional[QualityTier] = None) -> None:
        """
        Enhanced synthesis with emotion and context awareness.

        Args:
            text: Text to synthesize
            context: Conversation context for appropriate emotion selection
            override_emotion: Force specific emotion instead of auto-detection
            voice_profile: Custom voice profile to use
            quality_tier: Quality tier for this synthesis
        """
        with self.emotion_lock:
            # Determine emotion
            if override_emotion:
                target_emotion = override_emotion
            elif self.enhanced_config["emotion_detection_enabled"] and context:
                target_emotion = self._detect_emotion_for_context(text, context)
            else:
                target_emotion = self.current_emotion

            # Determine quality tier
            target_quality = quality_tier or self._determine_optimal_quality(context)

            # Update state
            self.current_emotion = target_emotion
            if context:
                self.current_context = context

            logger.info(f"🎭 Synthesizing with emotion: {target_emotion.value}, quality: {target_quality.value}, context: {context.value if context else 'general'}")

            # Create text stream for enhanced audio processor
            def text_stream():
                yield text

            # Set audio chunk callback
            def audio_callback(chunk):
                # Forward to original pipeline for WebSocket sending
                if hasattr(self.original_pipeline, 'audio'):
                    self.original_pipeline.audio.audio_chunks.put_nowait(chunk)

            self.enhanced_audio.set_audio_chunk_callback(audio_callback)

            # Synthesize with enhanced TTS
            synthesis_start = time.time()
            await self.enhanced_audio.stream(
                text_stream(),
                voice_emotion=target_emotion,
                voice_profile=voice_profile or self.enhanced_config["custom_voice_profile"],
                quality_preset=target_quality.value
            )

            synthesis_time = time.time() - synthesis_start

            # Update quality metrics
            self._update_quality_metrics(target_emotion, synthesis_time, target_quality)

    def _detect_emotion_for_context(self, text: str, context: ConversationContext) -> Emotion:
        """Detect appropriate emotion based on text and conversation context"""
        text_lower = text.lower()

        # Analyze text for emotion patterns
        emotion_scores = {}
        for emotion, patterns in self.emotion_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                emotion_scores[emotion] = score

        # Context-based emotion adjustment
        context_emotion_map = {
            ConversationContext.GREETING: Emotion.GENTLE,
            ConversationContext.FAREWELL: Emotion.CALM,
            ConversationContext.EMPATHY: Emotion.GENTLE,
            ConversationContext.EXCITEMENT: Emotion.EXCITED,
            ConversationContext.QUESTION: Emotion.NEUTRAL,
            ConversationContext.ANSWER: Emotion.CONFIDENT,
            ConversationContext.INSTRUCTION: Emotion.GENTLE,
        }

        base_emotion = context_emotion_map.get(context, Emotion.NEUTRAL)

        # Combine detected emotion with context preference
        if emotion_scores:
            detected_emotion = max(emotion_scores, key=emotion_scores.get)
            # Weight detected emotion higher for excitement/empathy contexts
            if context in [ConversationContext.EXCITEMENT, ConversationContext.EMPATHY]:
                return detected_emotion
            else:
                # Blend context and detected emotion
                return detected_emotion if detected_emotion != Emotion.NEUTRAL else base_emotion
        else:
            return base_emotion

    def _determine_optimal_quality(self, context: Optional[ConversationContext]) -> QualityTier:
        """Determine optimal quality tier based on context and performance targets"""
        if not self.adaptive_quality_enabled:
            return self.enhanced_config["quality_tier"]

        # Quality selection based on context and performance requirements
        context_quality_map = {
            ConversationContext.QUESTION: QualityTier.BALANCED,
            ConversationContext.ANSWER: QualityTier.BALANCED,
            ConversationContext.GREETING: QualityTier.FAST,
            ConversationContext.FAREWELL: QualityTier.FAST,
            ConversationContext.EXCITEMENT: QualityTier.HIGH_QUALITY,
            ConversationContext.EMPATHY: QualityTier.HIGH_QUALITY,
            ConversationContext.INSTRUCTION: QualityTier.BALANCED,
            ConversationContext.GENERAL: QualityTier.BALANCED,
        }

        base_quality = context_quality_map.get(context, QualityTier.BALANCED)

        # Adjust based on current latency performance
        if self.quality_monitoring["latency_measurements"]:
            avg_latency = np.mean(self.quality_monitoring["latency_measurements"])
            if avg_latency > self.target_latency_ms / 1000.0:  # Convert to seconds
                # Too slow, use faster quality
                if base_quality == QualityTier.STUDIO:
                    return QualityTier.HIGH_QUALITY
                elif base_quality == QualityTier.HIGH_QUALITY:
                    return QualityTier.BALANCED
                elif base_quality == QualityTier.BALANCED:
                    return QualityTier.FAST
            elif avg_latency < (self.target_latency_ms * 0.7) / 1000.0:  # Much faster than target
                # Much faster than target, can use higher quality
                if base_quality == QualityTier.FAST:
                    return QualityTier.BALANCED
                elif base_quality == QualityTier.BALANCED:
                    return QualityTier.HIGH_QUALITY

        return base_quality

    def _update_quality_metrics(self, emotion: Emotion, synthesis_time: float, quality_tier: QualityTier):
        """Update quality monitoring metrics"""
        with self.quality_lock:
            self.quality_monitoring["latency_measurements"].append(synthesis_time)
            self.quality_monitoring["quality_scores"].append(quality_tier.value)

            # Keep only recent measurements
            max_measurements = 100
            if len(self.quality_monitoring["latency_measurements"]) > max_measurements:
                self.quality_monitoring["latency_measurements"] = self.quality_monitoring["latency_measurements"][-max_measurements:]
                self.quality_monitoring["quality_scores"] = self.quality_monitoring["quality_scores"][-max_measurements:]

    async def clone_voice_from_conversation_sample(self,
                                               conversation_audio: np.ndarray,
                                               voice_name: str,
                                               description: Optional[str] = None) -> bool:
        """
        Clone voice from conversation audio sample.

        Args:
            conversation_audio: Audio sample from conversation
            voice_name: Name for the cloned voice
            description: Optional description of the voice characteristics
        """
        if not self.enhanced_config["enable_voice_cloning"]:
            logger.warning("⚠️ Voice cloning is disabled in configuration")
            return False

        try:
            logger.info(f"🎤️ Starting voice cloning from conversation sample: {voice_name}")
            self.voice_cloning_progress = 0.0

            # Clone voice using enhanced audio processor
            success = await self.enhanced_audio.clone_voice_from_sample(
                conversation_audio, voice_name
            )

            if success:
                self.voice_cloning_progress = 100.0
                logger.info(f"✅ Voice cloning completed successfully: {voice_name}")
                return True
            else:
                logger.error(f"❌ Voice cloning failed: {voice_name}")
                return False

        except Exception as e:
            logger.error(f"💥 Voice cloning error: {e}")
            return False

    def set_conversation_context(self, context: ConversationContext, text: Optional[str] = None):
        """Set conversation context for intelligent emotion selection"""
        self.current_context = context
        logger.debug(f"🎭 Conversation context set to: {context.value}")

        if text and self.enhanced_config["emotion_detection_enabled"]:
            detected_emotion = self._detect_emotion_for_context(text, context)
            self.set_emotion(detected_emotion)

    def set_emotion(self, emotion: Emotion):
        """Set current emotion for synthesis"""
        with self.emotion_lock:
            self.current_emotion = emotion
            self.enhanced_audio.set_emotion(emotion)
            logger.info(f"🎭 Emotion set to: {emotion.value}")

    def set_voice_profile(self, voice_profile: VoiceProfile):
        """Set voice profile for synthesis"""
        self.enhanced_audio.set_voice_profile(voice_profile)
        logger.info(f"🎤️ Voice profile set to: {voice_profile.name}")

    async def switch_to_provider(self, provider: TTSProvider, config: Optional[TTSConfig] = None):
        """Switch to a different TTS provider for comparison"""
        if not self.enhanced_config["enable_provider_switching"]:
            logger.warning("⚠️ Provider switching is disabled")
            return False

        with self.provider_switching_lock:
            try:
                logger.info(f"🔄 Switching to TTS provider: {provider.value}")

                # Create default config if none provided
                if not config:
                    config = TTSConfig(
                        provider=provider,
                        quality=self.enhanced_config["quality_tier"].value,
                        enable_cloning=self.enhanced_config["enable_voice_cloning"],
                        enable_emotion=self.enhanced_config["enable_emotion_synthesis"],
                        cache_size=self.enhanced_config.get("cache_size", 50)
                    )

                # Switch provider
                success = await self.enhanced_audio.switch_provider(provider, config)
                if success:
                    self.current_provider = provider
                    logger.info(f"✅ Successfully switched to provider: {provider.value}")
                    return True
                else:
                    logger.error(f"❌ Failed to switch to provider: {provider.value}")
                    return False

            except Exception as e:
                logger.error(f"💥 Provider switching error: {e}")
                return False

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive quality and performance metrics"""
        metrics = {
            "enhanced_tts_metrics": self.enhanced_audio.get_quality_metrics(),
            "current_emotion": self.current_emotion.value if self.current_emotion else "neutral",
            "current_context": self.current_context.value if self.current_context else "general",
            "current_provider": self.current_provider.value if self.current_provider else "none",
            "voice_cloning_enabled": self.enhanced_config["enable_voice_cloning"],
            "emotion_synthesis_enabled": self.enhanced_config["enable_emotion_synthesis"],
            "provider_switching_enabled": self.enhanced_config["enable_provider_switching"],
            "adaptive_quality_enabled": self.adaptive_quality_enabled,
            "target_latency_ms": self.target_latency_ms,
        }

        # Add quality monitoring data
        if self.quality_monitoring["latency_measurements"]:
            latencies = self.quality_monitoring["latency_measurements"]
            metrics["quality_monitoring"] = {
                "average_latency_ms": np.mean(latencies) * 1000,
                "min_latency_ms": min(latencies) * 1000,
                "max_latency_ms": max(latencies) * 1000,
                "recent_measurements": len(latencies),
                "latency_consistency": np.std(latencies) * 1000,  # Standard deviation
            }

        return metrics

    def get_available_voices_and_providers(self) -> Dict[str, Any]:
        """Get available voices and providers information"""
        return {
            "available_voices": self.enhanced_audio.get_available_voices(),
            "current_provider": self.current_provider.value if self.current_provider else "none",
            "supported_providers": [provider.value for provider in TTSProvider],
            "current_voice_profile": self.current_voice_profile.name if hasattr(self, 'current_voice_profile') and self.current_voice_profile else "default",
            "cloning_enabled": self.enhanced_config["enable_voice_cloning"],
        }

    # Backward compatibility methods
    def process_text(self, text: str) -> None:
        """Backward compatibility: process text with enhanced features"""
        # Detect context from text
        context = self._detect_context_from_text(text)

        # Use enhanced synthesis with emotion detection
        asyncio.create_task(self.synthesize_with_emotion(text, context))

    def _detect_context_from_text(self, text: str) -> ConversationContext:
        """Detect conversation context from text content"""
        text_lower = text.lower()

        context_scores = {}
        for context, keywords in self.context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                context_scores[context] = score

        if context_scores:
            return max(context_scores, key=context_scores.get)
        else:
            return ConversationContext.GENERAL

    # Delegation methods for backward compatibility
    def __getattr__(self, name):
        """Delegate undefined methods to original pipeline for backward compatibility"""
        if hasattr(self.original_pipeline, name):
            return getattr(self.original_pipeline, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    async def cleanup(self):
        """Cleanup enhanced speech pipeline resources"""
        logger.info("🧹 Cleaning up Enhanced Speech Pipeline Manager")

        try:
            # Cleanup enhanced audio processor
            await self.enhanced_audio.cleanup()

            # Cleanup original pipeline
            if hasattr(self.original_pipeline, 'cleanup'):
                await self.original_pipeline.cleanup()

            # Clear state
            self.conversation_history.clear()
            self.voice_profiles.clear()
            self.quality_monitoring.clear()

            logger.info("✅ Enhanced Speech Pipeline Manager cleanup completed")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Factory function for easy initialization
def create_enhanced_speech_pipeline(
    quality_tier: QualityTier = QualityTier.BALANCED,
    enable_emotion: bool = True,
    enable_cloning: bool = True,
    **kwargs
) -> EnhancedSpeechPipelineManager:
    """
    Factory function to create enhanced speech pipeline with sensible defaults.

    Args:
        quality_tier: Quality tier for TTS synthesis
        enable_emotion: Enable emotion-aware synthesis
        enable_cloning: Enable voice cloning features
        **kwargs: Additional arguments for SpeechPipelineManager

    Returns:
        EnhancedSpeechPipelineManager instance
    """
    return EnhancedSpeechPipelineManager(
        quality_tier=quality_tier,
        enable_emotion_synthesis=enable_emotion,
        enable_voice_cloning=enable_cloning,
        **kwargs
    )

if __name__ == "__main__":
    # Test enhanced speech pipeline
    async def test_enhanced_pipeline():
        """Test the enhanced speech pipeline with emotion support"""
        pipeline = EnhancedSpeechPipelineManager(
            quality_tier=QualityTier.BALANCED,
            enable_emotion_synthesis=True,
            enable_voice_cloning=True,
            adaptive_quality=True
        )

        # Test with different emotions and contexts
        test_cases = [
            ("Hello! How can I help you today?", ConversationContext.GREETING, None),
            ("That's amazing news! I'm so excited for you!", ConversationContext.EXCITEMENT, None),
            ("I understand how you're feeling. Let me help you with that.", ConversationContext.EMPATHY, None),
            ("What would you like to know about this topic?", ConversationContext.QUESTION, None),
            ("Here's the solution you're looking for.", ConversationContext.ANSWER, None),
        ]

        for text, context, emotion in test_cases:
            print(f"🎭 Testing: {text[:50]}...")
            print(f"📝 Context: {context.value}, Emotion: {emotion.value if emotion else 'auto'}")

            await pipeline.synthesize_with_emotion(text, context, emotion)
            await asyncio.sleep(2)  # Wait for synthesis

            # Get metrics
            metrics = pipeline.get_enhanced_metrics()
            print(f"📊 Metrics: {metrics}")
            print("-" * 50)

        await pipeline.cleanup()

    # Run test
    asyncio.run(test_enhanced_pipeline())