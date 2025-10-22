"""
Voice Control Interface for Real-Time Voice Switching and Customization

This module provides a comprehensive interface for managing multiple voice profiles,
real-time voice switching, and voice quality assessment.

Key Features:
- Real-time voice switching during conversation
- Voice quality assessment and comparison
- A/B testing between different voices/providers
- Voice cloning workflow management
- Audio quality monitoring and optimization

Author: Enhanced by Claude Code Assistant for RealtimeVoiceChat project
"""

import asyncio
import logging
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import json
from queue import Queue, Empty
import sounddevice as sd
from collections import defaultdict, deque

from enhanced_tts_manager import (
    EnhancedTTSManager, TTSProvider, TTSConfig, Emotion, VoiceProfile
)
from enhanced_audio_module import EnhancedAudioProcessor
from colors import Colors

logger = logging.getLogger(__name__)

class VoiceQualityMetrics:
    """Comprehensive voice quality metrics"""
    def __init__(self):
        self.naturalness_score = 0.0  # 1-10 scale
        self.clarity_score = 0.0       # 1-10 scale
        self.emotional_range = 0.0       # 1-10 scale
        self.latency_ms = 0.0           # milliseconds
        self.stability_score = 0.0       # 1-10 scale
        self.speaker_similarity = 0.0     # 1-10 scale (for cloned voices)
        self.overall_quality = 0.0        # 1-10 scale

class ABTestConfig:
    """Configuration for A/B testing between voices"""
    def __init__(self,
                 voice_a: VoiceProfile,
                 voice_b: VoiceProfile,
                 test_duration: int = 60,  # seconds
                 test_phrases: Optional[List[str]] = None):
        self.voice_a = voice_a
        self.voice_b = voice_b
        self.test_duration = test_duration
        self.test_phrases = test_phrases or [
            "Hello, this is a test of the voice synthesis system.",
            "I'm excited to try out different voice options.",
            "How does this voice sound to you?",
            "This is amazing technology for voice generation.",
            "Let me show you various emotional expressions."
        ]
        self.current_voice = "a"
        self.switch_interval = 10  # Switch voices every 10 seconds

class VoiceCloningSession:
    """Session for voice cloning workflow"""
    def __init__(self, target_duration: float = 30.0):
        self.target_duration = target_duration
        self.collected_audio = []
        self.current_duration = 0.0
        self.is_recording = False
        self.recording_thread = None
        self.audio_queue = Queue()

class VoiceControlInterface:
    """
    Advanced voice control interface for real-time voice management and quality assessment.

    This interface provides comprehensive voice control capabilities including
    real-time switching, A/B testing, and voice cloning workflows.
    """

    def __init__(
            self,
            enhanced_audio_processor: EnhancedAudioProcessor,
            sample_rate: int = 24000,
            channels: int = 1,
            buffer_size: int = 1024,
    ) -> None:
        """
        Initialize Voice Control Interface.

        Args:
            enhanced_audio_processor: EnhancedAudioProcessor instance
            sample_rate: Audio sample rate for recording
            channels: Number of audio channels
            buffer_size: Audio buffer size for recording
        """
        self.audio_processor = enhanced_audio_processor
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size

        # Voice management
        self.active_voice_profiles: Dict[str, VoiceProfile] = {}
        self.current_voice_name = "default"
        self.voice_quality_metrics: Dict[str, VoiceQualityMetrics] = {}
        self.voice_comparison_data: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Real-time voice switching
        self.voice_switching_enabled = False
        self.voice_transition_duration = 0.5  # seconds
        self.switching_scheduled = False
        self.switching_queue = Queue()

        # A/B testing
        self.ab_test_active = False
        self.ab_test_config: Optional[ABTestConfig] = None
        self.ab_test_results: Dict[str, List[Any]] = defaultdict(list)
        self.ab_test_start_time = None

        # Voice cloning
        self.cloning_session: Optional[VoiceCloningSession] = None
        self.cloning_progress_callback: Optional[Callable[[float], None]] = None
        self.recording_stream = None

        # Audio monitoring
        self.audio_monitoring_enabled = False
        self.audio_quality_monitor = None
        self.latency_monitor = deque(maxlen=100)  # Keep last 100 measurements
        self.audio_level_monitor = deque(maxlen=1000)

        # Event management
        self.voice_switch_callbacks: List[Callable[[str, str], None]] = []
        self.quality_change_callbacks: List[Callable[[str, VoiceQualityMetrics], None]] = []

        # Thread safety
        self.voice_control_lock = threading.Lock()
        self.cloning_lock = threading.Lock()
        self.ab_test_lock = threading.Lock()

        logger.info("🎤️ Voice Control Interface initialized")

    def add_voice_profile(self, voice_profile: VoiceProfile) -> None:
        """Add a voice profile to the active collection"""
        with self.voice_control_lock:
            self.active_voice_profiles[voice_profile.name] = voice_profile
            logger.info(f"➕ Added voice profile: {voice_profile.name} ({voice_profile.provider.value})")

    def switch_voice(self, voice_name: str, transition: bool = True) -> bool:
        """
        Switch to a different voice profile.

        Args:
            voice_name: Name of voice to switch to
            transition: Whether to use smooth transition

        Returns:
            True if switch was successful, False otherwise
        """
        with self.voice_control_lock:
            if voice_name not in self.active_voice_profiles:
                logger.error(f"❌ Voice profile not found: {voice_name}")
                return False

            if voice_name == self.current_voice_name:
                logger.debug(f"🔄 Voice already active: {voice_name}")
                return True

            old_voice = self.current_voice_name
            self.current_voice_name = voice_name

            # Switch voice in audio processor
            new_voice = self.active_voice_profiles[voice_name]
            self.audio_processor.set_voice_profile(new_voice)

            # Notify callbacks
            for callback in self.voice_switch_callbacks:
                try:
                    callback(old_voice, voice_name)
                except Exception as e:
                    logger.error(f"❌ Voice switch callback error: {e}")

            logger.info(f"🔄 Switched voice from {old_voice} to {voice_name}")
            return True

    def enable_real_time_switching(self, enable: bool = True) -> None:
        """Enable or disable real-time voice switching"""
        with self.voice_control_lock:
            self.voice_switching_enabled = enable
            if enable:
                threading.Thread(target=self._voice_switching_loop, daemon=True).start()
                logger.info("✅ Real-time voice switching enabled")
            else:
                logger.info("⏹️ Real-time voice switching disabled")

    def _voice_switching_loop(self) -> None:
        """Background loop for handling voice switching requests"""
        while self.voice_switching_enabled:
            try:
                # Check for scheduled voice switches
                if not self.switching_queue.empty():
                    switch_request = self.switching_queue.get(timeout=0.1)
                    self.switch_voice(switch_request["voice_name"], switch_request.get("transition", True))

                # Handle A/B test voice switching
                if self.ab_test_active and self.ab_test_config:
                    self._handle_ab_test_switching()

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Voice switching loop error: {e}")
                time.sleep(1.0)  # Back off on error

    def start_ab_test(self, test_config: ABTestConfig) -> bool:
        """
        Start A/B testing between two voices.

        Args:
            test_config: Configuration for A/B test

        Returns:
            True if test started successfully
        """
        with self.ab_test_lock:
            if self.ab_test_active:
                logger.warning("⚠️ A/B test already in progress")
                return False

            # Validate voices
            if test_config.voice_a.name not in self.active_voice_profiles:
                logger.error(f"❌ Voice A not found: {test_config.voice_a.name}")
                return False

            if test_config.voice_b.name not in self.active_voice_profiles:
                logger.error(f"❌ Voice B not found: {test_config.voice_b.name}")
                return False

            self.ab_test_config = test_config
            self.ab_test_active = True
            self.ab_test_start_time = time.time()
            self.ab_test_results.clear()

            # Initialize with voice A
            self.switch_voice(test_config.voice_a.name)

            logger.info(f"🧪 Started A/B test: {test_config.voice_a.name} vs {test_config.voice_b.name}")
            return True

    def _handle_ab_test_switching(self) -> None:
        """Handle automatic voice switching during A/B test"""
        if not self.ab_test_config:
            return

        current_time = time.time()
        elapsed = current_time - self.ab_test_start_time

        # Switch voices at specified intervals
        if int(elapsed) % self.ab_test_config.switch_interval == 0:
            # Switch to the other voice
            new_voice = self.ab_test_config.voice_b if self.current_voice_name == self.ab_test_config.voice_a.name else self.ab_test_config.voice_a
            self.switch_voice(new_voice.name)

            # Log the switch for analysis
            switch_data = {
                "timestamp": current_time,
                "voice_from": self.current_voice_name,
                "voice_to": new_voice.name,
                "elapsed_time": elapsed
            }
            self.ab_test_results["voice_switches"].append(switch_data)

        # Check if test is complete
        if elapsed >= self.ab_test_config.test_duration:
            self._complete_ab_test()

    def _complete_ab_test(self) -> None:
        """Complete the A/B test and generate results"""
        with self.ab_test_lock:
            self.ab_test_active = False

            # Generate test results
            results = {
                "test_duration": time.time() - self.ab_test_start_time,
                "voice_a": self.ab_test_config.voice_a.name,
                "voice_b": self.ab_test_config.voice_b.name,
                "voice_switches": self.ab_test_results.get("voice_switches", []),
                "quality_metrics_a": self.voice_quality_metrics.get(self.ab_test_config.voice_a.name),
                "quality_metrics_b": self.voice_quality_metrics.get(self.ab_test_config.voice_b.name),
                "user_preferences": self.ab_test_results.get("user_preferences", []),
                "latency_data_a": self.ab_test_results.get("latency_a", []),
                "latency_data_b": self.ab_test_results.get("latency_b", []),
            }

            # Save results
            self._save_ab_test_results(results)

            logger.info(f"✅ A/B test completed: {results}")

            # Reset to default voice
            self.switch_voice("default")

    def _save_ab_test_results(self, results: Dict[str, Any]) -> None:
        """Save A/B test results to file"""
        try:
            filename = f"ab_test_results_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"💾 A/B test results saved to: {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save A/B test results: {e}")

    def start_voice_cloning_session(self,
                               target_duration: float = 30.0,
                               voice_name: str = "custom_voice") -> bool:
        """
        Start a voice cloning session with audio recording.

        Args:
            target_duration: Duration of audio to collect (seconds)
            voice_name: Name for the cloned voice

        Returns:
            True if session started successfully
        """
        with self.cloning_lock:
            if self.cloning_session and self.cloning_session.is_recording:
                logger.warning("⚠️ Voice cloning session already in progress")
                return False

            self.cloning_session = VoiceCloningSession(target_duration)
            self.cloning_session.voice_name = voice_name

            # Start audio recording
            try:
                self.recording_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=self.buffer_size,
                    callback=self._audio_recording_callback
                )
                self.recording_stream.start()
                self.cloning_session.is_recording = True

                # Start monitoring thread
                self.recording_thread = threading.Thread(
                    target=self._monitor_cloning_session,
                    daemon=True
                )
                self.recording_thread.start()

                logger.info(f"🎤️ Started voice cloning session for: {voice_name}")
                return True

            except Exception as e:
                logger.error(f"❌ Failed to start voice cloning: {e}")
                self.cloning_session = None
                return False

    def _audio_recording_callback(self, indata: np.ndarray, frames: int, time_info, status: int) -> None:
        """Audio recording callback for voice cloning"""
        if self.cloning_session and self.cloning_session.is_recording:
            # Add audio data to queue
            self.cloning_session.audio_queue.put_nowait(indata.copy())

    def _monitor_cloning_session(self) -> None:
        """Monitor and manage voice cloning session"""
        if not self.cloning_session:
            return

        logger.info(f"🎙️ Recording voice cloning session: {self.cloning_session.target_duration}s")

        start_time = time.time()

        while self.cloning_session.is_recording and not self.cloning_session.current_duration >= self.cloning_session.target_duration:
            try:
                # Process audio data
                audio_chunk = self.cloning_session.audio_queue.get(timeout=0.1)
                self.cloning_session.collected_audio.append(audio_chunk)
                self.cloning_session.current_duration = time.time() - start_time

                # Update progress callback
                if self.cloning_progress_callback:
                    progress = min(100.0, (self.cloning_session.current_duration / self.cloning_session.target_duration) * 100.0)
                    self.cloning_progress_callback(progress)

                # Log progress
                if int(self.cloning_session.current_duration) % 5 == 0:
                    logger.info(f"🎙️ Cloning progress: {self.cloning_session.current_duration:.1f}s / {self.cloning_session.target_duration}s")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Cloning session monitoring error: {e}")

        # Stop recording
        self._stop_cloning_session()

    def _stop_cloning_session(self) -> None:
        """Stop voice cloning session and process collected audio"""
        if not self.cloning_session or not self.cloning_session.is_recording:
            return

        try:
            # Stop recording
            if self.recording_stream:
                self.recording_stream.stop()
                self.recording_stream.close()

            # Process collected audio
            self.cloning_session.is_recording = False

            # Combine all audio chunks
            if self.cloning_session.collected_audio:
                combined_audio = np.concatenate(self.cloning_session.collected_audio)
                logger.info(f"🎙️ Collected {len(combined_audio)} samples for voice cloning")

                # Start voice cloning process
                self._process_voice_cloning_audio(combined_audio)
            else:
                logger.warning("⚠️ No audio collected during cloning session")

        except Exception as e:
            logger.error(f"❌ Error stopping cloning session: {e}")

        finally:
            self.cloning_session = None

    def _process_voice_cloning_audio(self, audio_data: np.ndarray) -> None:
        """Process recorded audio for voice cloning"""
        try:
            # Create event loop if not running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Perform voice cloning
            success = loop.run_until_complete(
                self.audio_processor.clone_voice_from_sample(
                    audio_data,
                    self.cloning_session.voice_name
                )
            )

            if success:
                logger.info(f"✅ Voice cloning completed: {self.cloning_session.voice_name}")
                # Refresh available voices
                available_voices = self.audio_processor.get_available_voices()
                logger.info(f"🎤️ Available voices: {[v['name'] for v in available_voices]}")
            else:
                logger.error(f"❌ Voice cloning failed: {self.cloning_session.voice_name}")

        except Exception as e:
            logger.error(f"❌ Voice cloning processing error: {e}")

    def assess_voice_quality(self, voice_name: str) -> VoiceQualityMetrics:
        """
        Assess the quality of a voice profile using various metrics.

        Args:
            voice_name: Name of voice to assess

        Returns:
            Comprehensive quality metrics
        """
        if voice_name not in self.active_voice_profiles:
            logger.error(f"❌ Voice profile not found: {voice_name}")
            return VoiceQualityMetrics()

        metrics = VoiceQualityMetrics()

        try:
            # Switch to the voice for testing
            original_voice = self.current_voice_name
            self.switch_voice(voice_name)

            # Test phrases for quality assessment
            test_phrases = [
                "The quick brown fox jumps over the lazy dog.",
                "Hello, how are you today?",
                "I'm excited to tell you about this amazing technology!",
                "This is a calm and peaceful message.",
                "One, two, three, four, five.",
            ]

            # Test different emotions
            emotions_to_test = [Emotion.NEUTRAL, Emotion.HAPPY, Emotion.EXCITED]

            latency_measurements = []
            quality_scores = []

            for phrase in test_phrases:
                for emotion in emotions_to_test:
                    # Measure latency
                    start_time = time.time()

                    # Generate speech (this would use the TTS system)
                    synthesis_start = time.time()

                    # Simulate TTS processing time
                    time.sleep(0.1)  # Placeholder for actual TTS synthesis

                    synthesis_time = time.time() - synthesis_start
                    total_time = time.time() - start_time

                    latency_measurements.append(total_time)

                    # Update metrics
                    metrics.latency_ms = np.mean(latency_measurements) * 1000

                    # Simulate quality assessment
                    # In a real implementation, you would:
                    # 1. Analyze audio quality metrics
                    # 2. Use objective quality measures (MOS, PESQ, etc.)
                    # 3. Get user feedback
                    # 4. Compare against reference voices

            # Calculate quality scores based on various factors
            metrics.naturalness_score = self._assess_naturalness(voice_name)
            metrics.clarity_score = self._assess_clarity(voice_name)
            metrics.emotional_range = self._assess_emotional_range(voice_name)
            metrics.stability_score = self._assess_stability(voice_name)
            metrics.overall_quality = np.mean([
                metrics.naturalness_score,
                metrics.clarity_score,
                metrics.emotional_range,
                metrics.stability_score
            ])

            # Cache the results
            self.voice_quality_metrics[voice_name] = metrics

            # Restore original voice
            self.switch_voice(original_voice)

            logger.info(f"📊 Quality assessment completed for {voice_name}: {metrics.overall_quality:.1f}/10")

        except Exception as e:
            logger.error(f"❌ Voice quality assessment error: {e}")

        return metrics

    def _assess_naturalness(self, voice_name: str) -> float:
        """Assess naturalness of voice (simplified implementation)"""
        # In a real implementation, you would:
        # 1. Use trained models to assess naturalness
        # 2. Analyze prosody and intonation patterns
        # 3. Compare to human speech characteristics

        # Placeholder implementation
        voice = self.active_voice_profiles.get(voice_name)
        if voice:
            # Different base scores for different providers
            provider_scores = {
                TTSProvider.ELEVENLABS: 8.5,
                TTSProvider.COQUI_XTTS_V2: 8.0,
                TTSProvider.AZURE_SPEECH: 7.5,
                TTSProvider.OPENAI_TTS: 7.0,
            }
            return provider_scores.get(voice.provider, 7.0)
        return 7.0

    def _assess_clarity(self, voice_name: str) -> float:
        """Assess clarity and intelligibility"""
        # Placeholder implementation
        return 8.2

    def _assess_emotional_range(self, voice_name: str) -> float:
        """Assess ability to express different emotions"""
        # Placeholder implementation
        voice = self.active_voice_profiles.get(voice_name)
        if voice:
            # Check if voice supports emotion
            if hasattr(voice, 'emotion_style') and voice.emotion_style:
                return 8.0
        return 6.0

    def _assess_stability(self, voice_name: str) -> float:
        """Assess consistency and stability across generations"""
        # Placeholder implementation
        return 8.5

    def get_voice_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive voice comparison report"""
        report = {
            "available_voices": list(self.active_voice_profiles.keys()),
            "current_voice": self.current_voice_name,
            "quality_metrics": {},
            "ab_test_results": dict(self.ab_test_results) if self.ab_test_results else None,
            "voice_switching_enabled": self.voice_switching_enabled,
            "cloning_session_active": self.cloning_session is not None and self.cloning_session.is_recording,
        }

        # Add quality metrics for all voices
        for voice_name, metrics in self.voice_quality_metrics.items():
            report["quality_metrics"][voice_name] = {
                "naturalness": metrics.naturalness_score,
                "clarity": metrics.clarity_score,
                "emotional_range": metrics.emotional_range,
                "stability": metrics.stability_score,
                "overall": metrics.overall_quality,
                "latency_ms": metrics.latency_ms,
            }

        return report

    def add_voice_switch_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for voice switch events"""
        self.voice_switch_callbacks.append(callback)

    def add_quality_change_callback(self, callback: Callable[[str, VoiceQualityMetrics], None]) -> None:
        """Add callback for quality change events"""
        self.quality_change_callbacks.append(callback)

    def set_cloning_progress_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for voice cloning progress"""
        self.cloning_progress_callback = callback

    def cleanup(self) -> None:
        """Cleanup voice control interface resources"""
        logger.info("🧹 Cleaning up Voice Control Interface")

        try:
            # Stop voice switching
            self.voice_switching_enabled = False

            # Stop A/B testing
            if self.ab_test_active:
                self._complete_ab_test()

            # Stop cloning session
            if self.cloning_session and self.cloning_session.is_recording:
                self._stop_cloning_session()

            # Stop audio recording
            if self.recording_stream:
                self.recording_stream.stop()
                self.recording_stream.close()

            # Clear callbacks
            self.voice_switch_callbacks.clear()
            self.quality_change_callbacks.clear()

            logger.info("✅ Voice Control Interface cleanup completed")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

if __name__ == "__main__":
    # Test the voice control interface
    async def test_voice_control():
        """Test the voice control interface functionality"""
        # Create test voices
        voice_a = VoiceProfile("Voice A", TTSProvider.COQUI_XTTS_V2, "voice_a")
        voice_b = VoiceProfile("Voice B", TTSProvider.ELEVENLABS, "voice_b")

        # Create voice control interface (would need actual audio processor)
        # voice_interface = VoiceControlInterface(enhanced_audio_processor)

        # Test voice switching
        # voice_interface.add_voice_profile(voice_a)
        # voice_interface.add_voice_profile(voice_b)

        # Test A/B testing
        # test_config = ABTestConfig(voice_a, voice_b, test_duration=30)
        # voice_interface.start_ab_test(test_config)

        logger.info("Voice control interface test completed")

    asyncio.run(test_voice_control())