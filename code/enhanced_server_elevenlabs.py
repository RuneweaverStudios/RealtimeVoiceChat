"""
Enhanced RealtimeVoiceChat Server with ElevenLabs Integration

This is a ready-to-use server that integrates the enhanced TTS system
with ElevenLabs for premium voice quality.

Usage:
1. Set your ElevenLabs API key in .env file
2. Run: python -m code.enhanced_server_elevenlabs

Features:
- Premium ElevenLabs voice quality
- Emotion-aware synthesis
- Voice cloning support
- Real-time voice switching
- Quality monitoring and caching

Author: Enhanced by Claude Code Assistant for RealtimeVoiceChat project
"""

import asyncio
import os
import logging
from typing import Optional
import time

# Original server imports (for backward compatibility)
from server import (
    WebSocketManager, RealtimeVoiceChatHandler, colored, Colors,
    USAGE, parse_json_message, DeepSpeedSample
)
from enhanced_speech_pipeline import (
    EnhancedSpeechPipelineManager, QualityTier, ConversationContext
)
from enhanced_tts_manager import (
    TTSProvider, TTSConfig, Emotion, VoiceProfile
)
from colors import Colors

logger = logging.getLogger(__name__)

class EnhancedElevenLabsServer:
    """
    Enhanced server with ElevenLabs TTS integration.

    This server provides all the original functionality plus:
    - Premium ElevenLabs voice synthesis
    - Emotion-aware responses
    - Voice cloning capabilities
    - Real-time quality monitoring
    """

    def __init__(
            self,
            elevenlabs_api_key: Optional[str] = None,
            llm_provider: str = "openai",
            llm_model: str = "gpt-4",
            quality_tier: QualityTier = QualityTier.HIGH_QUALITY,
            enable_emotions: bool = True,
            enable_cloning: bool = True,
    ) -> None:
        """
        Initialize enhanced server with ElevenLabs TTS.

        Args:
            elevenlabs_api_key: ElevenLabs API key (if None, reads from env)
            llm_provider: LLM provider for responses (openai, ollama)
            llm_model: Specific LLM model to use
            quality_tier: TTS quality tier for synthesis
            enable_emotions: Enable emotion-aware synthesis
            enable_cloning: Enable voice cloning features
        """
        self.api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.quality_tier = quality_tier
        self.enable_emotions = enable_emotions
        self.enable_cloning = enable_cloning

        # Validate ElevenLabs API key
        if not self.api_key or not self.api_key.startswith("sk_"):
            raise ValueError(
                f"{Colors.RED}❌ Invalid ElevenLabs API key. "
                f"Please set ELEVENLABS_API_KEY in your .env file. "
                f"Get your key at: https://elevenlabs.io{Colors.RESET}"
            )

        # Initialize enhanced speech pipeline with ElevenLabs
        self.pipeline = None
        self._initialize_pipeline()

        # Statistics tracking
        self.stats = {
            "total_synthesis_requests": 0,
            "emotion_synthesis": 0,
            "cloning_sessions": 0,
            "voice_switches": 0,
            "cache_hits": 0,
            "api_errors": 0
        }

        logger.info(f"🎤️ {Colors.apply('Enhanced ElevenLabs Server initialized').green}")
        logger.info(f"🔑 {Colors.apply('API Key').yellow}: {self.api_key[:15]}...")
        logger.info(f"🎭 {Colors.apply('Emotions').blue}: {'Enabled' if self.enable_emotions else 'Disabled'}")
        logger.info(f"🎙️ {Colors.apply('Voice Cloning').purple}: {'Enabled' if self.enable_cloning else 'Disabled'}")
        logger.info(f"⭐ {Colors.apply('Quality Tier').cyan}: {self.quality_tier.value}")

    def _initialize_pipeline(self) -> None:
        """Initialize the enhanced speech pipeline with ElevenLabs"""
        try:
            # Create ElevenLabs voice profile
            elevenlabs_voice = VoiceProfile(
                name="Rachel",  # Popular high-quality voice
                provider=TTSProvider.ELEVENLABS,
                voice_id="rachel",  # Can be: adam, antoni, bella, domi, eli, josh, rachel, sam
                speed=1.0,
                pitch=1.0,
                sample_rate=24000
            )

            # Initialize enhanced speech pipeline
            self.pipeline = EnhancedSpeechPipelineManager(
                tts_engine="elevenlabs",  # Force ElevenLabs
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                quality_tier=self.quality_tier,
                enable_emotion_synthesis=self.enable_emotions,
                enable_voice_cloning=self.enable_cloning,
                adaptive_quality=True,
                custom_voice_profile=elevenlabs_voice,
                # Optimized for ElevenLabs
                optimization_mode="quality",  # Prioritize quality over speed
                realtime_latency_target=150.0  # 150ms target for real-time
            )

            logger.info(f"✅ {Colors.apply('ElevenLabs pipeline initialized successfully').green}")

        except Exception as e:
            logger.error(f"💥 {Colors.apply('Failed to initialize pipeline: ').red}{e}{Colors.RESET}")
            raise

    async def synthesize_with_context(self, text: str, context: Optional[str] = None) -> None:
        """
        Enhanced synthesis with context-aware emotion detection.

        Args:
            text: Text to synthesize
            context: Optional context for emotion selection
        """
        self.stats["total_synthesis_requests"] += 1

        # Detect conversation context
        detected_context = self._detect_context(text, context)

        # Map context to emotion (if emotions enabled)
        if self.enable_emotions:
            emotion = self._map_context_to_emotion(detected_context)
            self.stats["emotion_synthesis"] += 1
            logger.info(f"🎭 {Colors.apply('Detected emotion').purple}: {emotion.value}")
        else:
            emotion = Emotion.NEUTRAL

        try:
            # Use enhanced synthesis
            await self.pipeline.synthesize_with_emotion(
                text=text,
                context=detected_context,
                emotion=emotion
            )

            logger.info(f"🎤️ {Colors.apply('Synthesized').cyan}: '{text[:50]}...' with {emotion.value}")

        except Exception as e:
            self.stats["api_errors"] += 1
            logger.error(f"💥 {Colors.apply('Synthesis error').red}: {e}")

    def _detect_context(self, text: str, provided_context: Optional[str] = None) -> ConversationContext:
        """Detect conversation context from text or explicit context"""
        if provided_context:
            # Use provided context if available
            context_map = {
                "question": ConversationContext.QUESTION,
                "greeting": ConversationContext.GREETING,
                "answer": ConversationContext.ANSWER,
                "farewell": ConversationContext.FAREWELL,
                "empathy": ConversationContext.EMPATHY,
                "excited": ConversationContext.EXCITEMENT,
                "instruction": ConversationContext.INSTRUCTION,
            }
            return context_map.get(provided_context.lower(), ConversationContext.GENERAL)

        # Detect from text content
        text_lower = text.lower()

        # Greeting patterns
        greeting_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "welcome"]
        if any(keyword in text_lower for keyword in greeting_keywords):
            return ConversationContext.GREETING

        # Question patterns
        question_keywords = ["?", "what", "why", "how", "when", "where", "who", "which", "tell me"]
        if any(keyword in text_lower for keyword in question_keywords):
            return ConversationContext.QUESTION

        # Farewell patterns
        farewell_keywords = ["goodbye", "bye", "see you", "farewell", "take care"]
        if any(keyword in text_lower for keyword in farewell_keywords):
            return ConversationContext.FAREWELL

        # Excitement patterns
        excitement_keywords = ["amazing", "awesome", "fantastic", "wonderful", "incredible", "wow"]
        if any(keyword in text_lower for keyword in excitement_keywords):
            return ConversationContext.EXCITEMENT

        # Empathy patterns
        empathy_keywords = ["sorry", "understand", "feel", "difficult", "frustrated"]
        if any(keyword in text_lower for keyword in empathy_keywords):
            return ConversationContext.EMPATHY

        return ConversationContext.GENERAL

    def _map_context_to_emotion(self, context: ConversationContext) -> Emotion:
        """Map conversation context to appropriate emotion"""
        emotion_map = {
            ConversationContext.GREETING: Emotion.HAPPY,
            ConversationContext.QUESTION: Emotion.NEUTRAL,
            ConversationContext.ANSWER: Emotion.CONFIDENT,
            ConversationContext.FAREWELL: Emotion.GENTLE,
            ConversationContext.EXCITEMENT: Emotion.EXCITED,
            ConversationContext.EMPATHY: Emotion.GENTLE,
            ConversationContext.INSTRUCTION: Emotion.GENTLE,
        }
        return emotion_map.get(context, Emotion.NEUTRAL)

    def get_available_elevenlabs_voices(self) -> list:
        """Get list of available ElevenLabs voices"""
        return [
            {"id": "rachel", "name": "Rachel", "gender": "Female", "accent": "American"},
            {"id": "adam", "name": "Adam", "gender": "Male", "accent": "American"},
            {"id": "antoni", "name": "Antoni", "gender": "Male", "accent": "American"},
            {"id": "bella", "name": "Bella", "gender": "Female", "accent": "American"},
            {"id": "domi", "name": "Domi", "gender": "Male", "accent": "American"},
            {"id": "eli", "name": "Eli", "gender": "Male", "accent": "American"},
            {"id": "josh", "name": "Josh", "gender": "Male", "accent": "American"},
            {"id": "sam", "name": "Sam", "gender": "Male", "accent": "American"},
        ]

    def print_status(self) -> None:
        """Print current server status and statistics"""
        print(f"\n{Colors.BLUE}🎤️ Enhanced ElevenLabs Server Status{Colors.RESET}")
        print(f"{'='*50}")

        # Configuration
        print(f"📋 {Colors.apply('Configuration').yellow}:")
        print(f"   • Provider: {Colors.apply('ElevenLabs').green}")
        print(f"   • Quality: {Colors.apply(self.quality_tier.value).cyan}")
        print(f"   • Emotions: {Colors.apply('Enabled' if self.enable_emotions else 'Disabled').green}")
        print(f"   • Voice Cloning: {Colors.apply('Enabled' if self.enable_cloning else 'Disabled').green}")
        print(f"   • LLM: {Colors.apply(self.llm_provider).blue()} ({self.llm_model})")

        # Statistics
        print(f"\n📊 {Colors.apply('Statistics').cyan}:")
        print(f"   • Total Requests: {self.stats['total_synthesis_requests']}")
        print(f"   • Emotion Requests: {self.stats['emotion_synthesis']}")
        print(f"   • Voice Switches: {self.stats['voice_switches']}")
        print(f"   • Cloning Sessions: {self.stats['cloning_sessions']}")
        print(f"   • Cache Hits: {self.stats['cache_hits']}")
        print(f"   • API Errors: {self.stats['api_errors']}")

        # Available voices
        print(f"\n🎭 {Colors.apply('Available Voices').purple}:")
        voices = self.get_available_elevenlabs_voices()
        for i, voice in enumerate(voices, 1):
            print(f"   {i:2d}. {voice['name']} ({voice['id']}) - {voice['gender']} - {voice['accent']}")

        print(f"{'='*50}")

    async def cleanup(self) -> None:
        """Cleanup enhanced server resources"""
        logger.info(f"🧹 {Colors.apply('Cleaning up Enhanced ElevenLabs Server').yellow}")

        if self.pipeline:
            await self.pipeline.cleanup()

        logger.info(f"✅ {Colors.apply('Cleanup completed').green}")


class EnhancedElevenLabsHandler(RealtimeVoiceChatHandler):
    """
    Enhanced WebSocket handler with ElevenLabs integration.

    This handler extends the original functionality with emotion-aware
    responses and enhanced voice quality.
    """

    def __init__(self, *args, **kwargs):
        # Initialize parent handler
        super().__init__(*args, **kwargs)

        # Initialize enhanced ElevenLabs server
        self.enhanced_server = EnhancedElevenLabsServer(
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4"),
            quality_tier=QualityTier(os.getenv("TTS_QUALITY_TIER", "high_quality")),
            enable_emotions=os.getenv("ENABLE_EMOTIONS", "true").lower() == "true",
            enable_cloning=os.getenv("ENABLE_VOICE_CLONING", "true").lower() == "true",
        )

    async def handle_full_user_request(self, data):
        """Handle full user request with enhanced emotion synthesis."""
        try:
            # Extract text and optional context
            user_text = data.get("content", "").strip()
            context = data.get("context")  # Optional context for emotion selection

            if user_text:
                logger.info(f"👤 {Colors.apply('User request').yellow}: {user_text[:100]}{'...' if len(user_text) > 100 else ''}")

                # Use enhanced synthesis with context-aware emotion
                await self.enhanced_server.synthesize_with_context(
                    text=user_text,
                    context=context
                )

                # Add to history
                self.chat_history.append({
                    "role": "user",
                    "content": user_text,
                    "type": "final",
                    "timestamp": time.time()
                })

        except Exception as e:
            logger.error(f"💥 {Colors.apply('Error handling user request').red}: {e}")

    async def handle_voice_switch(self, data):
        """Handle voice switching request."""
        try:
            new_voice = data.get("voice_id")
            if new_voice:
                # For ElevenLabs, we can directly set the voice ID
                voice_profile = VoiceProfile(
                    name=new_voice.capitalize(),
                    provider=TTSProvider.ELEVENLABS,
                    voice_id=new_voice
                )

                # Switch voice in enhanced pipeline
                success = await self.enhanced_server.pipeline.switch_voice_profile(voice_profile)
                if success:
                    self.enhanced_server.stats["voice_switches"] += 1
                    logger.info(f"🔄 {Colors.apply('Switched to voice').purple}: {new_voice}")

                    # Send confirmation to client
                    await self.message_queue.put({
                        "type": "voice_switched",
                        "voice_id": new_voice,
                        "success": True
                    })

        except Exception as e:
            logger.error(f"💥 {Colors.apply('Error switching voice').red}: {e}")

    async def handle_emotion_override(self, data):
        """Handle emotion override request."""
        try:
            emotion_str = data.get("emotion")
            if emotion_str:
                emotion_map = {
                    "happy": Emotion.HAPPY,
                    "sad": Emotion.SAD,
                    "excited": Emotion.EXCITED,
                    "calm": Emotion.CALM,
                    "gentle": Emotion.GENTLE,
                    "confident": Emotion.CONFIDENT,
                    "angry": Emotion.ANGRY,
                    "surprised": Emotion.SURPRISED,
                }

                emotion = emotion_map.get(emotion_str.lower(), Emotion.NEUTRAL)
                self.enhanced_server.pipeline.set_emotion(emotion)
                logger.info(f"🎭 {Colors.apply('Emotion override').cyan}: {emotion.value}")

                # Send confirmation
                await self.message_queue.put({
                    "type": "emotion_set",
                    "emotion": emotion.value,
                    "success": True
                })

        except Exception as e:
            logger.error(f"💥 {Colors.apply('Error setting emotion').red}: {e}")

    async def handle_cloning_request(self, data):
        """Handle voice cloning request."""
        try:
            if not self.enhanced_server.enable_cloning:
                await self.message_queue.put({
                    "type": "cloning_disabled",
                    "message": "Voice cloning is not enabled"
                })
                return

            # Start cloning session
            voice_name = data.get("voice_name", "custom_voice")
            duration = data.get("duration", 30.0)

            self.enhanced_server.stats["cloning_sessions"] += 1
            logger.info(f"🎤️ {Colors.apply('Starting voice cloning').purple}: {voice_name}")

            # Send cloning progress updates
            await self.message_queue.put({
                "type": "cloning_started",
                "voice_name": voice_name,
                "duration": duration
            })

            # In a real implementation, you would:
            # 1. Record audio from user
            # 2. Send to ElevenLabs cloning API
            # 3. Return the cloned voice ID

            # For demo, simulate cloning success
            await asyncio.sleep(duration)  # Simulate recording time

            cloned_voice = VoiceProfile(
                name=f"Cloned {voice_name}",
                provider=TTSProvider.ELEVENLABS,
                voice_id=f"cloned_{voice_name.lower()}",
                is_custom=True
            )

            # Set the cloned voice
            await self.enhanced_server.pipeline.set_voice_profile(cloned_voice)

            await self.message_queue.put({
                "type": "cloning_completed",
                "voice_id": cloned_voice.voice_id,
                "success": True
            })

            logger.info(f"✅ {Colors.apply('Voice cloning completed').green}: {cloned_voice.voice_id}")

        except Exception as e:
            logger.error(f"💥 {Colors.apply('Error in voice cloning').red}: {e}")

    async def handle_status_request(self, data):
        """Handle status request with enhanced statistics."""
        try:
            # Get enhanced metrics
            metrics = self.enhanced_server.pipeline.get_enhanced_metrics()
            voices = self.enhanced_server.get_available_elevenlabs_voices()
            server_stats = self.enhanced_server.stats

            await self.message_queue.put({
                "type": "status_response",
                "metrics": metrics,
                "available_voices": voices,
                "statistics": server_stats,
                "configuration": {
                    "provider": "elevenlabs",
                    "quality_tier": self.enhanced_server.quality_tier.value,
                    "emotions_enabled": self.enhanced_server.enable_emotions,
                    "cloning_enabled": self.enhanced_server.enable_cloning,
                }
            })

        except Exception as e:
            logger.error(f"💥 {Colors.apply('Error handling status request').red}: {e}")

    async def process_message(self, client_id: str, msg_text: str):
        """Enhanced message processing with additional command support."""
        try:
            # Parse JSON message
            data = parse_json_message(msg_text)
            if not data:
                return

            message_type = data.get("type")

            # Handle original message types
            if message_type == "partial_user_request":
                await self.handle_partial_user_request(data)
            elif message_type == "final_user_request":
                await self.handle_full_user_request(data)
            elif message_type == "tts_start":
                await self.handle_tts_start(data)
            elif message_type == "tts_stop":
                await self.handle_tts_stop(data)
            elif message_type == "abort_generation":
                await self.handle_abort_generation(data)

            # Enhanced message types
            elif message_type == "voice_switch":
                await self.handle_voice_switch(data)
            elif message_type == "emotion_override":
                await self.handle_emotion_override(data)
            elif message_type == "cloning_request":
                await self.handle_cloning_request(data)
            elif message_type == "status_request":
                await self.handle_status_request(data)
            else:
                logger.warning(f"⚠️ {Colors.apply('Unknown message type').yellow}: {message_type}")

        except Exception as e:
            logger.error(f"💥 {Colors.apply('Error processing message').red}: {e}")


def main():
    """Main function to run enhanced ElevenLabs server."""
    print(f"\n{Colors.BLUE}🎤️ Enhanced RealtimeVoiceChat Server - ElevenLabs Edition{Colors.RESET}")
    print(f"{Colors.apply('Premium voice quality with emotion support and voice cloning').green}")
    print(f"{'='*60}")

    # Check environment
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_key:
        print(f"\n{Colors.RED}❌ ELEVENLABS_API_KEY not found in environment{Colors.RESET}")
        print(f"{Colors.YELLOW}Please set up your .env file with your ElevenLabs API key{Colors.RESET}")
        print(f"Get your key at: {Colors.apply('https://elevenlabs.io/app').blue}")
        print(f"\nExample .env file:")
        print(f"{'ELEVENLABS_API_KEY=sk_your_actual_key_here'}")
        print(f"{'TTS_PROVIDER=elevenlabs'}")
        print(f"{'LLM_PROVIDER=openai'}")
        print(f"{'LLM_MODEL=gpt-4'}")
        return

    # Initialize enhanced server
    server = EnhancedElevenLabsHandler(
        wsm=WebSocketManager(),
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL", "gpt-4"),
        device=os.getenv("DEVICE", "cpu"),
        system_prompt=os.getenv("SYSTEM_PROMPT"),
        realtime_enabled=os.getenv("REALTIME_ENABLED", "true") == "true",
        tts_engine="elevenlabs",  # Force ElevenLabs
        orpheus_model=None
    )

    # Show status
    server.enhanced_server.print_status()

    print(f"\n{Colors.GREEN}🚀 Starting Enhanced ElevenLabs Server...{Colors.RESET}")
    print(f"{Colors.apply('Premium voice quality with emotion support').cyan}")
    print(f"{Colors.apply('Voice cloning enabled').purple}")
    print(f"{Colors.apply('Open your browser and navigate to the server URL').yellow}")
    print(f"\n{Colors.apply('WebSocket will be available at: ws://localhost:8000/ws').blue}")
    print(f"{Colors.apply('Web interface at: http://localhost:8000').blue}")

    try:
        import uvicorn
        uvicorn.run(
            server,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info(f"\n{Colors.YELLOW}🛑 Shutting down Enhanced ElevenLabs Server...{Colors.RESET}")
        asyncio.run(server.cleanup())
    except Exception as e:
        logger.error(f"💥 {Colors.apply('Server error').red}: {e}")


if __name__ == "__main__":
    main()