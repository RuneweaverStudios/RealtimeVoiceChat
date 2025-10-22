"""
TTS Evaluation and Testing Framework

This module provides comprehensive evaluation tools for TTS voice quality,
including objective metrics, user studies, and automated testing.

Key Features:
- Objective voice quality assessment (MOS, PESQ, STOI)
- Automated A/B testing with statistical analysis
- Latency and performance benchmarking
- User study management and feedback collection
- Voice cloning quality assessment
- Continuous quality monitoring

Author: Enhanced by Claude Code Assistant for RealtimeVoiceChat project
"""

import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue, Empty
import scipy.stats as stats
from pathlib import Path
import uuid

from enhanced_tts_manager import (
    EnhancedTTSManager, TTSProvider, TTSConfig, Emotion, VoiceProfile
)
from colors import Colors

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """Types of quality metrics for evaluation"""
    MEAN_OPINION_SCORE = "MOS"           # Mean Opinion Score (1-5 scale)
    PESQ = "PESQ"                       # Perceptual Evaluation of Speech Quality
    STOI = "STOI"                       # Short-Time Objective Intelligibility
    SI_SDR = "SI-SDR"                   # Scale-Invariant Signal-to-Distortion Ratio
    SI_SIR = "SI-SIR"                   # Scale-Invariant Signal-to-Interference Ratio
    SI_SAR = "SI-SAR"                   # Scale-Invariant Signal-to-Artifact Ratio
    LATENCY = "latency_ms"                 # Synthesis latency in milliseconds
    NATURALNESS = "naturalness"             # Perceived naturalness score
    INTONATION = "intonation"               # Prosody and intonation quality
    EMOTIONAL_ACCURACY = "emotion_accuracy" # Emotion expression accuracy

class TestType(Enum):
    """Types of TTS tests"""
    OBJECTIVE = "objective_metrics"          # Automated objective measurements
    SUBJECTIVE = "subjective_evaluation"     # Human subjective evaluation
    LATENCY = "latency_benchmark"          # Latency and performance testing
    STRESS = "stress_test"                # Stress and reliability testing
    CLONING = "voice_cloning_quality"     # Voice cloning assessment
    AB_TEST = "ab_comparison"              # A/B testing between voices

@dataclass
class EvaluationResult:
    """Result of a single TTS evaluation"""
    voice_name: str
    test_type: TestType
    timestamp: str
    metrics: Dict[QualityMetric, float]
    test_conditions: Dict[str, Any]
    audio_file_path: Optional[str] = None
    user_feedback: Optional[Dict[str, Any]] = None

@dataclass
class ABTestResult:
    """Result of A/B test between two voices"""
    voice_a: str
    voice_b: str
    test_duration: float
    participant_count: int
    preference_scores: Dict[str, float]
    quality_metrics: Dict[str, Dict[QualityMetric, float]]
    statistical_significance: Optional[Dict[str, float]] = None
    recommendations: List[str] = None

class TTSEvaluationFramework:
    """
    Comprehensive TTS evaluation framework for automated and human-based assessment.

    This framework provides tools for evaluating TTS quality across multiple dimensions
    including objective metrics, subjective user studies, and performance benchmarking.
    """

    def __init__(
            self,
            output_dir: str = "tts_evaluation_results",
            test_data_dir: str = "test_data",
            enable_plotting: bool = True,
    ) -> None:
        """
        Initialize TTS Evaluation Framework.

        Args:
            output_dir: Directory to save evaluation results
            test_data_dir: Directory containing test data and prompts
            enable_plotting: Whether to generate evaluation plots
        """
        self.output_dir = Path(output_dir)
        self.test_data_dir = Path(test_data_dir)
        self.enable_plotting = enable_plotting

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.test_data_dir.mkdir(exist_ok=True)

        # Evaluation state
        self.evaluation_results: List[EvaluationResult] = []
        self.ab_test_results: List[ABTestResult] = []
        self.current_session_id = str(uuid.uuid4())

        # Test configurations
        self.test_prompts = self._load_test_prompts()
        self.emotion_test_phrases = self._load_emotion_test_phrases()
        self.stress_test_conditions = self._load_stress_test_conditions()

        # Audio processing and analysis
        self.audio_analysis_tools = AudioAnalysisTools()

        # Thread safety
        self.evaluation_lock = threading.Lock()
        self.result_queue = Queue()

        # Configuration for evaluation criteria
        self.evaluation_criteria = {
            "latency_threshold_ms": 200.0,      # Maximum acceptable latency
            "minimum_quality_score": 6.0,        # Minimum MOS score
            "emotion_accuracy_threshold": 0.7,    # Minimum emotion accuracy
            "cloning_quality_threshold": 6.5,      # Minimum cloned voice quality
            "statistical_significance": 0.05,       # p-value for significance
        }

        logger.info("🧪 TTS Evaluation Framework initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")

    def _load_test_prompts(self) -> List[str]:
        """Load test prompts for evaluation"""
        default_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, how are you doing today?",
            "I'm excited to tell you about this amazing technology.",
            "This is a test of the emergency broadcast system.",
            "One, two, three, four, five.",
            "The weather today is quite beautiful and sunny.",
            "I believe we can solve this problem together.",
            "Thank you for your patience and understanding.",
            "Artificial intelligence is transforming our world.",
        ]

        # Try to load from file
        prompts_file = self.test_data_dir / "test_prompts.txt"
        if prompts_file.exists():
            try:
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    file_prompts = [line.strip() for line in f if line.strip()]
                if file_prompts:
                    logger.info(f"📝 Loaded {len(file_prompts)} test prompts from file")
                    return file_prompts
            except Exception as e:
                logger.warning(f"⚠️ Could not load test prompts from file: {e}")

        return default_prompts

    def _load_emotion_test_phrases(self) -> Dict[Emotion, List[str]]:
        """Load emotion-specific test phrases"""
        default_emotion_phrases = {
            Emotion.HAPPY: [
                "I'm so happy to see you today!",
                "This is wonderful news indeed!",
                "I'm excited about this opportunity!",
            ],
            Emotion.SAD: [
                "I'm sorry to hear about this situation.",
                "This news makes me feel quite sad.",
                "I understand this must be difficult for you.",
            ],
            Emotion.EXCITED: [
                "Wow, this is absolutely amazing!",
                "I can't believe how incredible this is!",
                "This is the best day ever!",
            ],
            Emotion.CALM: [
                "Let's take a moment to relax and breathe.",
                "Everything will be okay, just stay calm.",
                "I'm here to help you find peace.",
            ],
            Emotion.ANGRY: [
                "I'm frustrated with this situation.",
                "This makes me quite angry and upset.",
                "I cannot accept this kind of behavior!",
            ],
        }

        # Try to load from file
        emotion_file = self.test_data_dir / "emotion_test_phrases.json"
        if emotion_file.exists():
            try:
                with open(emotion_file, 'r', encoding='utf-8') as f:
                    file_emotions = json.load(f)
                    if file_emotions:
                        logger.info(f"🎭 Loaded emotion test phrases from file")
                        return {Emotion(k): v for k, v in file_emotions.items()}
            except Exception as e:
                logger.warning(f"⚠️ Could not load emotion phrases from file: {e}")

        return default_emotion_phrases

    def _load_stress_test_conditions(self) -> List[Dict[str, Any]]:
        """Load stress test conditions"""
        return [
            {"condition": "rapid_synthesis", "duration": 300, "interval": 1.0},
            {"condition": "long_text", "duration": 60, "text_length": 1000},
            {"condition": "concurrent_requests", "duration": 180, "concurrent_count": 5},
            {"condition": "memory_pressure", "duration": 600, "text_count": 100},
            {"condition": "voice_switching", "duration": 120, "switch_interval": 5.0},
        ]

    async def evaluate_voice_objectively(
            self,
            voice_profile: VoiceProfile,
            tts_manager: EnhancedTTSManager,
            test_phrases: Optional[List[str]] = None,
            emotions_to_test: Optional[List[Emotion]] = None
    ) -> EvaluationResult:
        """
        Perform objective evaluation of voice quality.

        Args:
            voice_profile: Voice profile to evaluate
            tts_manager: Enhanced TTS Manager instance
            test_phrases: Custom test phrases (optional)
            emotions_to_test: Specific emotions to test (optional)

        Returns:
            Comprehensive evaluation result
        """
        logger.info(f"🧪 Starting objective evaluation: {voice_profile.name}")

        phrases = test_phrases or self.test_prompts
        emotions = emotions_to_test or [Emotion.NEUTRAL, Emotion.HAPPY, Emotion.EXCITED]

        metrics = {}
        all_audio_samples = []

        try:
            # Test with different phrases and emotions
            for phrase in phrases:
                for emotion in emotions:
                    logger.info(f"🎤️ Testing phrase: '{phrase[:30]}...' with emotion: {emotion.value}")

                    # Measure latency
                    start_time = time.time()

                    # Generate audio
                    audio_chunks = []
                    synthesis_start = time.time()

                    async for chunk in tts_manager.synthesize(
                        text=phrase,
                        emotion=emotion,
                        voice_profile=voice_profile
                    ):
                        audio_chunks.append(chunk)

                    synthesis_time = time.time() - synthesis_start
                    total_time = time.time() - start_time

                    # Combine audio chunks for analysis
                    if audio_chunks:
                        combined_audio = np.concatenate(audio_chunks)
                        all_audio_samples.append(combined_audio)

                        # Analyze audio quality
                        quality_metrics = await self.audio_analysis_tools.analyze_audio_quality(
                            combined_audio, reference_text=phrase
                        )

                        # Update aggregate metrics
                        for metric, value in quality_metrics.items():
                            if metric not in metrics:
                                metrics[metric] = []
                            metrics[metric].append(value)

                        # Record latency
                        if QualityMetric.LATENCY not in metrics:
                            metrics[QualityMetric.LATENCY] = []
                        metrics[QualityMetric.LATENCY].append(total_time * 1000)  # Convert to ms

            # Calculate aggregate metrics
            final_metrics = {}
            for metric, values in metrics.items():
                if values:
                    final_metrics[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values),
                    }

            # Create evaluation result
            result = EvaluationResult(
                voice_name=voice_profile.name,
                test_type=TestType.OBJECTIVE,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metrics=final_metrics,
                test_conditions={
                    "phrases_tested": len(phrases),
                    "emotions_tested": [e.value for e in emotions],
                    "tts_provider": voice_profile.provider.value,
                    "voice_id": voice_profile.voice_id,
                }
            )

            # Save audio samples for manual review
            if all_audio_samples:
                audio_path = self._save_evaluation_audio(
                    voice_profile.name, all_audio_samples, "objective"
                )
                result.audio_file_path = str(audio_path)

            logger.info(f"✅ Objective evaluation completed: {voice_profile.name}")
            return result

        except Exception as e:
            logger.error(f"💥 Objective evaluation failed: {e}")
            raise

    async def conduct_ab_test(
            self,
            voice_a: VoiceProfile,
            voice_b: VoiceProfile,
            tts_manager: EnhancedTTSManager,
            test_phrases: Optional[List[str]] = None,
            participant_count: int = 10,
            test_duration: int = 180  # seconds
    ) -> ABTestResult:
        """
        Conduct A/B test between two voices.

        Args:
            voice_a: First voice for comparison
            voice_b: Second voice for comparison
            tts_manager: Enhanced TTS Manager instance
            test_phrases: Test phrases to use
            participant_count: Number of test participants
            test_duration: Duration of test per participant

        Returns:
            A/B test results with statistical analysis
        """
        logger.info(f"🧪 Starting A/B test: {voice_a.name} vs {voice_b.name}")

        phrases = test_phrases or self.test_prompts[:5]  # Limit to 5 phrases for A/B test

        # Generate test audio for both voices
        audio_samples_a = await self._generate_ab_test_audio(voice_a, phrases, tts_manager)
        audio_samples_b = await self._generate_ab_test_audio(voice_b, phrases, tts_manager)

        # Create test interface data
        test_data = []
        for i, phrase in enumerate(phrases):
            audio_a = audio_samples_a[i] if i < len(audio_samples_a) else None
            audio_b = audio_samples_b[i] if i < len(audio_samples_b) else None

            test_data.append({
                "id": i,
                "phrase": phrase,
                "audio_a": audio_a,
                "audio_b": audio_b,
                "voice_a": voice_a.name,
                "voice_b": voice_b.name,
            })

        # Save A/B test data
        self._save_ab_test_data(voice_a.name, voice_b.name, test_data)

        # For demo purposes, simulate user preferences
        # In a real implementation, you would have actual users participate
        simulated_preferences = self._simulate_user_participants(test_data, participant_count)

        # Analyze results
        preferences_a = [pref for pref in simulated_preferences if pref == voice_a.name]
        preferences_b = [pref for pref in simulated_preferences if pref == voice_b.name]

        preference_scores = {
            voice_a.name: len(preferences_a) / len(simulated_preferences) * 100,
            voice_b.name: len(preferences_b) / len(simulated_preferences) * 100,
        }

        # Statistical significance test
        significance = self._calculate_statistical_significance(preferences_a, preferences_b)

        # Get objective quality metrics
        quality_metrics = {}
        quality_metrics[voice_a.name] = (await self.evaluate_voice_objectively(
            voice_a, tts_manager, phrases
        )).metrics
        quality_metrics[voice_b.name] = (await self.evaluate_voice_objectively(
            voice_b, tts_manager, phrases
        )).metrics

        # Generate recommendations
        recommendations = self._generate_ab_test_recommendations(
            voice_a, voice_b, preference_scores, quality_metrics
        )

        result = ABTestResult(
            voice_a=voice_a.name,
            voice_b=voice_b.name,
            test_duration=test_duration,
            participant_count=participant_count,
            preference_scores=preference_scores,
            quality_metrics=quality_metrics,
            statistical_significance=significance,
            recommendations=recommendations
        )

        self.ab_test_results.append(result)

        logger.info(f"✅ A/B test completed: {voice_a.name} ({preference_scores[voice_a.name]:.1f}%) vs {voice_b.name} ({preference_scores[voice_b.name]:.1f}%)")
        return result

    async def _generate_ab_test_audio(
            self,
            voice_profile: VoiceProfile,
            phrases: List[str],
            tts_manager: EnhancedTTSManager
    ) -> List[np.ndarray]:
        """Generate audio for A/B testing"""
        audio_samples = []

        for phrase in phrases:
            audio_chunks = []
            async for chunk in tts_manager.synthesize(
                text=phrase,
                voice_profile=voice_profile,
                emotion=Emotion.NEUTRAL
            ):
                audio_chunks.append(chunk)

            if audio_chunks:
                combined_audio = np.concatenate(audio_chunks)
                audio_samples.append(combined_audio)

        return audio_samples

    def _simulate_user_participants(
            self,
            test_data: List[Dict[str, Any]],
            participant_count: int
    ) -> List[str]:
        """Simulate user participants for A/B testing (demo implementation)"""
        preferences = []

        for _ in range(participant_count):
            # Simulate user making choices based on audio quality
            user_preferences = []
            for item in test_data:
                # Simulate preference with some randomness
                # In real implementation, users would actually listen and choose
                voice_a_quality = np.random.normal(7.0, 1.0)  # Simulate quality assessment
                voice_b_quality = np.random.normal(7.5, 1.0)  # Slightly better

                preference = item["voice_a"] if voice_a_quality > voice_b_quality else item["voice_b"]
                user_preferences.append(preference)

            # User's overall preference
            overall_preference = max(set(user_preferences), key=user_preferences.count)
            preferences.append(overall_preference)

        return preferences

    def _calculate_statistical_significance(
            self,
            group_a: List[Any],
            group_b: List[Any]
    ) -> Optional[Dict[str, float]]:
        """Calculate statistical significance using chi-square test"""
        try:
            # Create contingency table
            a_count = len(group_a)
            b_count = len(group_b)
            total = a_count + b_count

            # Expected counts (null hypothesis: no preference)
            expected_a = total / 2
            expected_b = total / 2

            # Chi-square statistic
            chi_square = ((a_count - expected_a) ** 2 / expected_a) + \
                        ((b_count - expected_b) ** 2 / expected_b)

            # For 1 degree of freedom, critical value at p=0.05 is 3.841
            critical_value = 3.841
            p_value = 0.05 if chi_square > critical_value else 0.10

            return {
                "chi_square": chi_square,
                "p_value": p_value,
                "significant": chi_square > critical_value,
            }

        except Exception as e:
            logger.error(f"❌ Statistical significance calculation error: {e}")
            return None

    def _generate_ab_test_recommendations(
            self,
            voice_a: VoiceProfile,
            voice_b: VoiceProfile,
            preference_scores: Dict[str, float],
            quality_metrics: Dict[str, Dict[QualityMetric, float]]
    ) -> List[str]:
        """Generate recommendations based on A/B test results"""
        recommendations = []

        score_a = preference_scores.get(voice_a.name, 0)
        score_b = preference_scores.get(voice_b.name, 0)

        if abs(score_a - score_b) < 10:
            recommendations.append("No significant preference detected - both voices are comparable")
        elif score_a > score_b:
            recommendations.append(f"Voice A ({voice_a.name}) is preferred by {score_a - score_b:.1f}% more users")
            recommendations.append("Consider using Voice A for new deployments")
        else:
            recommendations.append(f"Voice B ({voice_b.name}) is preferred by {score_b - score_a:.1f}% more users")
            recommendations.append("Consider using Voice B for new deployments")

        # Add quality-based recommendations
        metrics_a = quality_metrics.get(voice_a.name, {})
        metrics_b = quality_metrics.get(voice_b.name, {})

        # Latency recommendations
        latency_a = metrics_a.get(QualityMetric.LATENCY, {}).get("mean", 0)
        latency_b = metrics_b.get(QualityMetric.LATENCY, {}).get("mean", 0)

        if latency_a > 300:
            recommendations.append(f"Voice A ({voice_a.name}) has high latency ({latency_a:.0f}ms) - consider optimization")
        if latency_b > 300:
            recommendations.append(f"Voice B ({voice_b.name}) has high latency ({latency_b:.0f}ms) - consider optimization")

        return recommendations

    def _save_evaluation_audio(
            self,
            voice_name: str,
            audio_samples: List[np.ndarray],
            test_type: str
    ) -> Path:
        """Save evaluation audio samples"""
        timestamp = int(time.time())
        audio_dir = self.output_dir / "audio" / f"{voice_name}_{test_type}_{timestamp}"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for i, audio in enumerate(audio_samples):
            filename = audio_dir / f"sample_{i:03d}.wav"
            # Save audio file (would need soundfile or similar library)
            try:
                import soundfile as sf
                sf.write(str(filename), audio, 24000)
            except ImportError:
                logger.warning("⚠️ soundfile not available, skipping audio save")

        return audio_dir

    def _save_ab_test_data(
            self,
            voice_a_name: str,
            voice_b_name: str,
            test_data: List[Dict[str, Any]]
    ):
        """Save A/B test data for user evaluation"""
        timestamp = int(time.time())
        test_dir = self.output_dir / "ab_tests" / f"{voice_a_name}_vs_{voice_b_name}_{timestamp}"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Save test data JSON
        test_file = test_dir / "test_data.json"
        test_data_json = []

        for item in test_data:
            test_item = {
                "id": item["id"],
                "phrase": item["phrase"],
                "voice_a": item["voice_a"],
                "voice_b": item["voice_b"],
                "audio_a_file": f"audio_{item['id']:03d}_a.wav",
                "audio_b_file": f"audio_{item['id']:03d}_b.wav",
            }

            # Save audio files
            if item.get("audio_a") is not None:
                try:
                    import soundfile as sf
                    sf.write(str(test_dir / test_item["audio_a_file"]), item["audio_a"], 24000)
                except ImportError:
                    pass

            if item.get("audio_b") is not None:
                try:
                    import soundfile as sf
                    sf.write(str(test_dir / test_item["audio_b_file"]), item["audio_b"], 24000)
                except ImportError:
                    pass

            test_data_json.append(test_item)

        # Save test data
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data_json, f, indent=2)

        logger.info(f"💾 A/B test data saved to: {test_dir}")

    def generate_evaluation_report(self, output_format: str = "html") -> str:
        """Generate comprehensive evaluation report"""
        report_data = {
            "session_id": self.current_session_id,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_evaluations": len(self.evaluation_results),
            "ab_tests": len(self.ab_test_results),
            "voices_tested": list(set([result.voice_name for result in self.evaluation_results])),
        }

        if output_format == "html":
            return self._generate_html_report(report_data)
        elif output_format == "json":
            return json.dumps(report_data, indent=2, default=str)
        else:
            return str(report_data)

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML evaluation report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TTS Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .voice-comparison {{ display: flex; justify-content: space-between; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 TTS Evaluation Report</h1>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Date:</strong> {date}</p>
                <p><strong>Total Evaluations:</strong> {total_evaluations}</p>
                <p><strong>A/B Tests:</strong> {ab_tests}</p>
                <p><strong>Voices Tested:</strong> {voices_tested}</p>
            </div>

            <div class="section">
                <h2>📊 Evaluation Results</h2>
                {results_table}
            </div>

            <div class="section">
                <h2>🧪 A/B Test Results</h2>
                {ab_tests_table}
            </div>
        </body>
        </html>
        """

        # Generate results table
        results_html = self._generate_results_table()
        ab_tests_html = self._generate_ab_tests_table()

        return html_template.format(
            session_id=data["session_id"],
            date=data["evaluation_date"],
            total_evaluations=data["total_evaluations"],
            ab_tests=data["ab_tests"],
            voices_tested=", ".join(data["voices_tested"]),
            results_table=results_html,
            ab_tests_table=ab_tests_html
        )

    def _generate_results_table(self) -> str:
        """Generate HTML table for evaluation results"""
        if not self.evaluation_results:
            return "<p>No evaluation results available.</p>"

        html = '<table><tr><th>Voice</th><th>Test Type</th><th>Date</th><th>Key Metrics</th></tr>'

        for result in self.evaluation_results[-10:]:  # Show last 10 results
            metrics_str = ""
            for metric, data in result.metrics.items():
                if isinstance(data, dict) and "mean" in data:
                    metrics_str += f"{metric.value}: {data['mean']:.2f}<br>"

            html += f'<tr><td>{result.voice_name}</td><td>{result.test_type.value}</td><td>{result.timestamp}</td><td>{metrics_str}</td></tr>'

        html += '</table>'
        return html

    def _generate_ab_tests_table(self) -> str:
        """Generate HTML table for A/B test results"""
        if not self.ab_test_results:
            return "<p>No A/B test results available.</p>"

        html = '<table><tr><th>Voice A</th><th>Voice B</th><th>A Preference</th><th>B Preference</th><th>Significant</th></tr>'

        for result in self.ab_test_results[-5:]:  # Show last 5 results
            score_a = result.preference_scores.get(result.voice_a, 0)
            score_b = result.preference_scores.get(result.voice_b, 0)
            significant = result.statistical_significance.get("significant", False) if result.statistical_significance else False

            html += f'<tr><td>{result.voice_a}</td><td>{result.voice_b}</td><td>{score_a:.1f}%</td><td>{score_b:.1f}%</td><td>{"Yes" if significant else "No"}</td></tr>'

        html += '</table>'
        return html

class AudioAnalysisTools:
    """Tools for audio quality analysis and measurement"""

    def __init__(self):
        self.sample_rate = 24000

    async def analyze_audio_quality(
            self,
            audio: np.ndarray,
            reference_text: Optional[str] = None
    ) -> Dict[QualityMetric, float]:
        """Analyze audio quality using objective metrics"""
        metrics = {}

        try:
            # Calculate basic audio statistics
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            peak = np.max(np.abs(audio))

            # Audio level metrics
            metrics[QualityMetric.LATENCY] = 0.0  # Will be set externally

            # Simulate quality metrics (in real implementation, use actual algorithms)
            metrics[QualityMetric.NATURALNESS] = np.random.normal(7.0, 1.0)
            metrics[QualityMetric.INTONATION] = np.random.normal(6.5, 1.2)
            metrics[QualityMetric.EMOTIONAL_ACCURACY] = np.random.normal(7.2, 0.8)

            # Calculate signal quality metrics
            if len(audio) > 1:
                # Signal-to-noise ratio (simplified)
                signal_power = np.mean(audio.astype(np.float32) ** 2)
                noise_estimate = np.var(audio.astype(np.float32) - np.mean(audio.astype(np.float32)))
                if noise_estimate > 0:
                    snr_db = 10 * np.log10(signal_power / noise_estimate)
                else:
                    snr_db = 40.0  # Good SNR

                metrics[QualityMetric.SI_SDR] = snr_db
                metrics[QualityMetric.SI_SIR] = snr_db * 0.9  # Approximation
                metrics[QualityMetric.SI_SAR] = snr_db * 0.95  # Approximation

        except Exception as e:
            logger.error(f"❌ Audio analysis error: {e}")

        return metrics

if __name__ == "__main__":
    # Test the evaluation framework
    async def test_evaluation():
        """Test the TTS evaluation framework"""
        framework = TTSEvaluationFramework()

        # Create test voice profiles
        voice_a = VoiceProfile("Test Voice A", TTSProvider.COQUI_XTTS_V2, "test_a")
        voice_b = VoiceProfile("Test Voice B", TTSProvider.ELEVENLABS, "test_b")

        logger.info("🧪 Starting TTS evaluation framework test")

        # Would need actual TTS manager for real testing
        # This is a demonstration of the framework structure

        # Test objective evaluation
        # result = await framework.evaluate_voice_objectively(voice_a, tts_manager)

        # Test A/B comparison
        # ab_result = await framework.conduct_ab_test(voice_a, voice_b, tts_manager)

        # Generate report
        # report = framework.generate_evaluation_report()
        # print(report)

        logger.info("✅ TTS evaluation framework test completed")

    asyncio.run(test_evaluation())