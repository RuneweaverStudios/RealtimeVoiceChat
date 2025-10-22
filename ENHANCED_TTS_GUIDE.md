# Enhanced TTS Implementation Guide

This guide explains how to use the enhanced Text-to-Speech (TTS) features in RealtimeVoiceChat, including voice cloning, emotion support, and provider switching.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/RealtimeVoiceChat.git
cd RealtimeVoiceChat

# Install enhanced TTS dependencies
pip install -r requirements_enhanced.txt

# For GPU acceleration (NVIDIA)
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# For Coqui XTTS v2 (latest)
pip install git+https://github.com/coqui-ai/TTS.git@dev

# For voice cloning support
pip install sounddevice pydub librosa noisereduce
```

### 2. Basic Usage

```python
from enhanced_speech_pipeline import EnhancedSpeechPipelineManager, QualityTier

# Create enhanced speech pipeline
pipeline = EnhancedSpeechPipelineManager(
    quality_tier=QualityTier.BALANCED,
    enable_emotion_synthesis=True,
    enable_voice_cloning=True,
    adaptive_quality=True
)

# Use emotion-aware synthesis
await pipeline.synthesize_with_emotion(
    text="Hello! I'm excited to meet you!",
    context=ConversationContext.GREETING,
    emotion=Emotion.EXCITED
)
```

## 🎛️ Enhanced Features

### Voice Cloning

Create custom voices from audio samples:

```python
# Clone voice from recording
success = await pipeline.clone_voice_from_conversation_sample(
    conversation_audio=recorded_audio,
    voice_name="Custom Assistant",
    description="Friendly and professional voice"
)

if success:
    print("✅ Voice cloned successfully!")
    # The new voice is automatically available
```

#### Voice Cloning Quality Tips

1. **Audio Quality Requirements:**
   - Minimum 3 seconds of clear audio
   - Sample rate: 24000 Hz or higher
   - Low background noise
   - Clear pronunciation and natural speech

2. **Recording Best Practices:**
   ```python
   # Use the voice cloning interface
   from voice_control_interface import VoiceControlInterface

   voice_interface = VoiceControlInterface(pipeline.enhanced_audio)

   # Start cloning session
   await voice_interface.start_voice_cloning_session(
       target_duration=30.0,  # 30 seconds recommended
       voice_name="My Voice"
   )

   # Progress callback
   def cloning_progress(progress):
       print(f"Cloning progress: {progress:.1f}%")

   voice_interface.set_cloning_progress_callback(cloning_progress)
   ```

### Emotion-Aware Synthesis

Different emotions are automatically detected or can be explicitly set:

```python
from enhanced_tts_manager import Emotion

# Available emotions:
emotions = [
    Emotion.NEUTRAL,    # Normal speaking voice
    Emotion.HAPPY,       # Cheerful, positive tone
    Emotion.SAD,         # Sympathetic, gentle tone
    Emotion.EXCITED,     # Enthusiastic, energetic
    Emotion.CALM,        # Peaceful, soothing tone
    Emotion.GENTLE,      # Warm, caring voice
    Emotion.CONFIDENT,    # Assured, professional tone
    Emotion.ANGRY,       # Upset, frustrated tone
    Emotion.SURPRISED    # Startled, amazed
]

# Use emotion in synthesis
await pipeline.synthesize_with_emotion(
    text="This is wonderful news!",
    context=ConversationContext.EXCITEMENT,
    emotion=Emotion.EXCITED
)
```

### Multi-Provider Support

Switch between different TTS providers for quality or cost optimization:

```python
from enhanced_tts_manager import TTSProvider

# Available providers:
providers = {
    TTSProvider.COQUI_XTTS_V2:    "High quality, local processing",
    TTSProvider.ELEVENLABS:          "Premium quality, API-based",
    TTSProvider.AZURE_SPEECH:         "Enterprise grade",
    TTSProvider.OPENAI_TTS:           "Good quality, simple API",
}

# Switch providers
success = await pipeline.switch_to_provider(
    new_provider=TTSProvider.ELEVENLABS,
    config=TTSConfig(
        provider=TTSProvider.ELEVENLABS,
        api_key="your-api-key",
        voice_id="rachel",
        quality="high_quality"
    )
)
```

### Real-Time Voice Switching

Switch voices during conversation for different contexts:

```python
from voice_control_interface import VoiceControlInterface

# Create voice control interface
voice_control = VoiceControlInterface(pipeline.enhanced_audio)

# Add multiple voice profiles
voice_control.add_voice_profile(custom_voice_1)
voice_control.add_voice_profile(custom_voice_2)

# Enable real-time switching
voice_control.enable_real_time_switching(True)

# Schedule voice changes
voice_control.switching_queue.put({
    "voice_name": "custom_voice_1",
    "transition": True
})

# Add callback for voice changes
def on_voice_change(old_voice, new_voice):
    print(f"Voice switched from {old_voice} to {new_voice}")

voice_control.add_voice_switch_callback(on_voice_change)
```

## 🎛️ Quality Control

### Quality Tiers

Choose quality based on your needs:

| Tier | Latency | Quality | Use Case |
|-------|----------|---------|-----------|
| `ULTRA_FAST` | ~100ms | Basic | Real-time interactions |
| `FAST` | ~150ms | Good | Interactive applications |
| `BALANCED` | ~200ms | Good | General use (default) |
| `HIGH_QUALITY` | ~300ms | High | Professional applications |
| `STUDIO` | ~500ms | Maximum | Content creation |

```python
# Set quality mode
pipeline.set_quality_mode("high_quality")

# Or use during synthesis
await pipeline.synthesize_with_emotion(
    text="Professional announcement",
    quality_tier=QualityTier.STUDIO
)
```

### Audio Enhancement

Built-in audio processing for better voice quality:

```python
# Enable audio enhancement (default: enabled)
enhanced_config = {
    "audio_quality_enhancement": True,
    "noise_reduction": True,
    "equalization": True,
    "compression": True,  # Dynamic range compression
    "reverb_removal": True,
}
```

## 🧪 A/B Testing Framework

Compare different voices systematically:

```python
from tts_evaluation_framework import TTSEvaluationFramework

# Create evaluation framework
framework = TTSEvaluationFramework(
    output_dir="voice_evaluation_results",
    enable_plotting=True
)

# Define voices to compare
voice_a = VoiceProfile("Voice A", TTSProvider.COQUI_XTTS_V2, "voice_a")
voice_b = VoiceProfile("Voice B", TTSProvider.ELEVENLABS, "voice_b")

# Run A/B test
ab_result = await framework.conduct_ab_test(
    voice_a=voice_a,
    voice_b=voice_b,
    test_phrases=[
        "Hello, this is a voice comparison test.",
        "How would you like me to help you today?",
        "I'm excited to show you this amazing feature!"
    ],
    participant_count=20,
    test_duration=180  # 3 minutes
)

print(f"A/B Test Results:")
print(f"Voice A preference: {ab_result.preference_scores['voice_a']:.1f}%")
print(f"Voice B preference: {ab_result.preference_scores['voice_b']:.1f}%")
print(f"Statistical significance: {ab_result.statistical_significance}")
```

### Voice Quality Assessment

Automatically evaluate voice quality:

```python
# Assess voice quality
quality_metrics = await framework.evaluate_voice_objectively(
    voice_profile=custom_voice,
    tts_manager=pipeline.enhanced_audio
)

print(f"Quality Assessment Results:")
print(f"Naturalness: {quality_metrics.metrics[QualityMetric.NATURALNESS]:.1f}/10")
print(f"Clarity: {quality_metrics.metrics[QualityMetric.CLARITY]:.1f}/10")
print(f"Emotional Range: {quality_metrics.metrics[QualityMetric.EMOTIONAL_RANGE]:.1f}/10")
print(f"Latency: {quality_metrics.metrics[QualityMetric.LATENCY]:.1f}ms")
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys for different providers
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
export AZURE_SPEECH_KEY="your-azure-speech-key"
export AZURE_REGION="eastus"

# TTS Configuration
export TTS_QUALITY_TIER="balanced"
export TTS_ENABLE_EMOTIONS="true"
export TTS_ENABLE_CLONING="true"
export TTS_CACHE_SIZE="100"

# Audio Settings
export AUDIO_SAMPLE_RATE="24000"
export AUDIO_CHANNELS="1"
export AUDIO_BUFFER_SIZE="1024"
```

### Configuration File

Create `tts_config.json`:

```json
{
  "providers": {
    "coqui_xtts_v2": {
      "model": "multilingual-v2-v1",
      "cache_size": 50,
      "enable_emotion": true,
      "enable_cloning": true,
      "streaming": true
    },
    "elevenlabs": {
      "api_key": "${ELEVENLABS_API_KEY}",
      "voice_id": "rachel",
      "model": "eleven_monolingual_v1",
      "cache_size": 30
    },
    "azure_speech": {
      "api_key": "${AZURE_SPEECH_KEY}",
      "region": "${AZURE_REGION}",
      "voice_id": "en-US-JennyNeural",
      "cache_size": 40
    }
  },
  "quality": {
    "default_tier": "balanced",
    "adaptive_quality": true,
    "target_latency_ms": 200.0,
    "audio_enhancement": true,
    "noise_reduction": true,
    "equalization": true
  },
  "performance": {
    "enable_caching": true,
    "cache_size": 100,
    "optimization_mode": "balanced",
    "gpu_acceleration": true
  },
  "emotions": {
    "enable_detection": true,
    "auto_apply": true,
    "strength": 0.7
  },
  "cloning": {
    "enable": true,
    "minimum_duration": 3.0,
    "target_duration": 30.0,
    "quality_threshold": 6.0
  }
}
```

## 🎯 Use Cases and Examples

### 1. Customer Service Bot

```python
customer_service_pipeline = EnhancedSpeechPipelineManager(
    quality_tier=QualityTier.BALANCED,  # Responsive but good quality
    enable_emotion_synthesis=True,
    adaptive_quality=True,
    realtime_latency_target=150.0  # Fast for customer service
)

# Set emotion based on customer sentiment
if customer_sentiment == "frustrated":
    await pipeline.synthesize_with_emotion(
        text="I understand your frustration. Let me help you resolve this.",
        context=ConversationContext.EMPATHY,
        emotion=Emotion.GENTLE
    )
elif customer_sentiment == "happy":
    await pipeline.synthesize_with_emotion(
        text="I'm so glad I could help you today!",
        context=ConversationContext.EXCITEMENT,
        emotion=Emotion.HAPPY
    )
```

### 2. Educational Assistant

```python
educational_pipeline = EnhancedSpeechPipelineManager(
    quality_tier=QualityTier.HIGH_QUALITY,  # High quality for learning
    enable_emotion_synthesis=True,
    adaptive_quality=False,  # Consistent quality
)

# Use gentle, encouraging voice for learning
await pipeline.synthesize_with_emotion(
    text="That's a great question! Let me explain this step by step.",
    context=ConversationContext.INSTRUCTION,
    emotion=Emotion.GENTLE
)
```

### 3. Content Creation

```python
content_pipeline = EnhancedSpeechPipelineManager(
    quality_tier=QualityTier.STUDIO,  # Maximum quality
    enable_voice_cloning=True,
    audio_quality_enhancement=True
)

# Use custom cloned voice for content
await pipeline.synthesize_with_emotion(
    text="Welcome to my channel! Today we're exploring amazing AI technology.",
    context=ConversationContext.GREETING,
    voice_profile=custom_content_voice,
    quality_tier=QualityTier.STUDIO
)
```

### 4. Voice Assistant

```python
assistant_pipeline = EnhancedSpeechPipelineManager(
    quality_tier=QualityTier.BALANCED,
    enable_emotion_synthesis=True,
    enable_voice_cloning=True,
    adaptive_quality=True,
)

# Adaptive responses based on context
def respond_to_user(user_text, user_context):
    if user_context.get("time_of_day") == "morning":
        return pipeline.synthesize_with_emotion(
            text="Good morning! How can I help you start your day?",
            context=ConversationContext.GREETING,
            emotion=Emotion.HAPPY
        )
    elif user_context.get("mood") == "stressed":
        return pipeline.synthesize_with_emotion(
            text="I can help you relax. Let's take this one step at a time.",
            context=ConversationContext.EMPATHY,
            emotion=Emotion.CALM
        )
    else:
        return pipeline.synthesize_with_emotion(
            text="How can I assist you today?",
            context=ConversationContext.GENERAL,
            emotion=Emotion.NEUTRAL
        )
```

## 🐛 Troubleshooting

### Common Issues

1. **Voice Cloning Fails**
   ```python
   # Check audio quality
   def validate_cloning_audio(audio):
       if len(audio) < 24000 * 3:  # Less than 3 seconds
           print("❌ Audio too short - need at least 3 seconds")
           return False
       if np.max(np.abs(audio)) < 1000:
           print("⚠️ Audio levels are too low")
           return False
       return True
   ```

2. **High Latency**
   ```python
   # Check provider and quality settings
   if latency_ms > 300:
       # Try lower quality tier
       pipeline.set_quality_mode("fast")
       # Or switch to faster provider
       await pipeline.switch_to_provider(TTSProvider.COQUI_XTTS_V2)
   ```

3. **Emotion Not Working**
   ```python
   # Ensure emotion synthesis is enabled
   if not pipeline.enhanced_config["enable_emotion_synthesis"]:
       # Enable emotions
       pipeline.enable_emotion_synthesis = True

   # Check if provider supports emotions
   provider = pipeline.current_provider
   if provider not in [TTSProvider.ELEVENLABS, TTSProvider.COQUI_XTTS_V2]:
       print("⚠️ Current provider doesn't support emotion synthesis")
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for TTS-specific debugging
logger = logging.getLogger("enhanced_tts_manager")
logger.setLevel(logging.DEBUG)
```

## 📊 Performance Optimization

### GPU Acceleration

```python
# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
    # Enable GPU in TTS configuration
    config = TTSConfig(
        provider=TTSProvider.COQUI_XTTS_V2,
        device="cuda",
        optimize_for_gpu=True
    )
else:
    print("⚠️ GPU not available, using CPU")
```

### Caching Strategy

```python
# Optimize cache for your use case
if production_environment:
    config = TTSConfig(
        cache_size=200,  # Larger cache for production
        enable_persistent_cache=True,
        cache_ttl=3600  # 1 hour
    )
else:
    config = TTSConfig(
        cache_size=50,   # Smaller cache for development
        enable_persistent_cache=False
    )
```

### Memory Management

```python
# Monitor memory usage
import psutil
import gc

def monitor_memory():
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 80:
        print("⚠️ High memory usage, triggering cleanup")
        gc.collect()

# Call periodically
monitor_memory()
```

## 📚 API Reference

### EnhancedSpeechPipelineManager

```python
class EnhancedSpeechPipelineManager:
    def __init__(
        self,
        quality_tier: QualityTier = QualityTier.BALANCED,
        enable_emotion_synthesis: bool = True,
        enable_voice_cloning: bool = True,
        adaptive_quality: bool = True,
        # ... other parameters
    ):
        """Initialize enhanced speech pipeline"""

    async def synthesize_with_emotion(
        self,
        text: str,
        context: Optional[ConversationContext] = None,
        override_emotion: Optional[Emotion] = None,
        voice_profile: Optional[VoiceProfile] = None,
        quality_tier: Optional[QualityTier] = None
    ) -> None:
        """Enhanced synthesis with emotion and context awareness"""

    async def clone_voice_from_conversation_sample(
        self,
        conversation_audio: np.ndarray,
        voice_name: str,
        description: Optional[str] = None
    ) -> bool:
        """Clone voice from conversation audio sample"""

    def set_emotion(self, emotion: Emotion):
        """Set current emotion for synthesis"""

    def set_voice_profile(self, voice_profile: VoiceProfile):
        """Set voice profile for synthesis"""

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive quality and performance metrics"""
```

### VoiceControlInterface

```python
class VoiceControlInterface:
    def add_voice_profile(self, voice_profile: VoiceProfile):
        """Add a voice profile to the active collection"""

    def switch_voice(self, voice_name: str, transition: bool = True) -> bool:
        """Switch to a different voice profile"""

    def enable_real_time_switching(self, enable: bool = True):
        """Enable or disable real-time voice switching"""

    def start_ab_test(self, test_config: ABTestConfig) -> bool:
        """Start A/B testing between two voices"""

    def start_voice_cloning_session(
        self,
        target_duration: float = 30.0,
        voice_name: str = "custom_voice"
    ) -> bool:
        """Start a voice cloning session with audio recording"""
```

## 🔗 External Resources

- [Coqui TTS Documentation](https://coqui.ai/docs/)
- [ElevenLabs API Documentation](https://elevenlabs.io/docs/)
- [Azure Speech Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/)
- [OpenAI TTS Documentation](https://platform.openai.com/docs/guides/text-to-speech)

## 📄 License

This enhanced TTS implementation extends the original RealtimeVoiceChat project. Please refer to the original project's license and comply with individual provider licensing requirements.

For commercial use, ensure you have appropriate licenses for:
- Coqui TTS (Apache 2.0)
- ElevenLabs API (Commercial terms)
- Azure Speech Services (Azure pricing tier)
- OpenAI API (Usage terms)

---

**Happy coding! 🎤️✨**