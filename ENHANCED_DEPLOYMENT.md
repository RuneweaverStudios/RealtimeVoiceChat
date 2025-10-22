# Enhanced RealtimeVoiceChat Deployment Guide

This guide covers deployment of the enhanced TTS system with voice cloning, emotion support, and multi-provider capabilities.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced RealtimeVoiceChat                │
├─────────────────────────────────────────────────────────────┤
│  Frontend (HTML/JS)                                  │
│  - Real-time audio streaming                              │
│  - Voice control interface                                │
│  - Emotion and quality controls                         │
└─────────────────────────────────────────────────────────────┘
                             │ WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 Enhanced Speech Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│  EnhancedSpeechPipelineManager                           │
│  - Multi-provider TTS support                        │
│  - Emotion-aware synthesis                            │
│  - Voice cloning workflow                             │
│  - Quality monitoring                                │
│                                                      │
│  Components:                                         │
│  • EnhancedAudioProcessor                           │
│  • VoiceControlInterface                            │
│  • TTSEvaluationFramework                         │
│  • EnhancedTTSManager                            │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   TTS Providers                         │
├─────────────────────────────────────────────────────────────┤
│  • Coqui XTTS v2 (Local)                          │
│  • ElevenLabs (Cloud API)                          │
│  • Azure Speech (Cloud API)                          │
│  • OpenAI TTS (Cloud API)                          │
└─────────────────────────────────────────────────────────────┘
```

## 🐳 Docker Deployment

### 1. Enhanced Dockerfile

```dockerfile
# Enhanced RealtimeVoiceChat with Advanced TTS Support
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    git \
    portaudio19-dev \
    pkg-config

# Install Python dependencies
COPY requirements_enhanced.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_enhanced.txt

# Install Coqui TTS from source (latest)
RUN pip install --no-cache-dir \
    git+https://github.com/coqui-ai/TTS.git@dev \
    --upgrade

# Create app directory
WORKDIR /app

# Copy application code
COPY code/ /app/
COPY static/ /app/static/
COPY models/ /app/models/  # If you have pre-downloaded models

# Create directories
RUN mkdir -p /app/logs /app/temp /app/models/tts /app/models/cloning

# Environment variables
ENV PYTHONPATH=/app
ENV TTS_CACHE_DIR=/app/models/tts
ENV VOICE_CLONING_DIR=/app/models/cloning
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "server"]
```

### 2. Docker Compose with Enhanced Services

```yaml
# docker-compose.enhanced.yml
version: '3.8'

services:
  # Main application with enhanced TTS
  realtime-voice-chat:
    build:
      context: .
      dockerfile: Dockerfile.enhanced
    container_name: enhanced-voice-chat
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - TTS_PROVIDER=${TTS_PROVIDER:-coqui_xtts_v2}
      - TTS_QUALITY_TIER=${TTS_QUALITY_TIER:-balanced}
      - ENABLE_EMOTIONS=${ENABLE_EMOTIONS:-true}
      - ENABLE_CLONING=${ENABLE_CLONING:-true}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY:-}
      - AZURE_SPEECH_KEY=${AZURE_SPEECH_KEY:-}
      - AZURE_REGION=${AZURE_REGION:-eastus}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config:/app/config
      - ./tts_evaluation_results:/app/evaluation_results
    restart: unless-stopped
    depends_on:
      - redis
      - nginx

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: voice-chat-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --aof-use-rdb-preamble no

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: voice-chat-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - realtime-voice-chat
    restart: unless-stopped

  # Monitoring with Prometheus (optional)
  prometheus:
    image: prom/prometheus:latest
    container_name: voice-chat-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  # Grafana dashboard (optional)
  grafana:
    image: grafana/grafana:latest
    container_name: voice-chat-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

### 3. GPU-Enabled Docker

```dockerfile
# Dockerfile.gpu - For NVIDIA GPU support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    git \
    portaudio19-dev

# Install Python dependencies
COPY requirements_enhanced.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements_enhanced.txt

# Install GPU-optimized packages
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

# Copy application
COPY code/ /app/
COPY static/ /app/static/

# GPU environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

CMD ["python3", "-m", "server"]
```

## 🌐 Production Deployment

### 1. Kubernetes Deployment

```yaml
# k8s-enhanced-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enhanced-voice-chat
spec:
  replicas: 3
  selector:
    matchLabels:
      app: enhanced-voice-chat
  template:
    metadata:
      labels:
        app: enhanced-voice-chat
    spec:
      containers:
      - name: voice-chat-app
        image: your-registry/realtime-voice-chat:enhanced
        ports:
        - containerPort: 8000
        env:
        - name: TTS_PROVIDER
          value: "coqui_xtts_v2"
        - name: TTS_QUALITY_TIER
          value: "balanced"
        - name: ENABLE_EMOTIONS
          value: "true"
        - name: ELEVENLABS_API_KEY
          valueFrom:
            secretKeyRef:
              name: voice-chat-secrets
              key: elevenlabs-api-key
        - name: AZURE_SPEECH_KEY
          valueFrom:
            secretKeyRef:
              name: voice-chat-secrets
              key: azure-speech-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: enhanced-voice-chat-service
spec:
  selector:
    app: enhanced-voice-chat
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: enhanced-voice-chat-ingress
spec:
  tls:
  - hosts:
    - voice-chat.yourdomain.com
    secretName: voice-chat-tls
  rules:
  - host: voice-chat.yourdomain.com
    http:
      paths:
      - path: /
        backend:
          serviceName: enhanced-voice-chat-service
          servicePort: 80
```

### 2. Environment Configuration

```yaml
# production.env
# Core Configuration
TTS_PROVIDER=coqui_xtts_v2
TTS_QUALITY_TIER=balanced
ENABLE_EMOTIONS=true
ENABLE_CLONING=true
ENABLE_PROVIDER_SWITCHING=true

# Provider API Keys
ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
AZURE_SPEECH_KEY=${AZURE_SPEECH_KEY}
AZURE_REGION=eastus
OPENAI_API_KEY=${OPENAI_API_KEY}

# Performance Settings
CACHE_SIZE=200
LATENCY_TARGET_MS=200
ADAPTIVE_QUALITY=true
GPU_ACCELERATION=true

# Security
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
API_RATE_LIMIT=100
WEBSOCKET_RATE_LIMIT=50

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_DASHBOARD_ENABLED=true
LOG_LEVEL=INFO
```

### 3. Monitoring Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'enhanced-voice-chat'
    static_configs:
      - targets: ['voice-chat:8000']
    metrics_path: /metrics
    scrape_interval: 5s

rule_files:
  - "voice_chat_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## 🔒 Security Configuration

### 1. Enhanced Security Headers

```python
# enhanced_security.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import jwt
import secrets

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["voice-chat.yourdomain.com", "localhost"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voice-chat.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Rate-Limit-Remaining"]
)

# JWT Authentication
SECRET_KEY = secrets.token_urlsafe(32)

async def verify_websocket_token(websocket, token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        # Validate token claims
        if payload.get("service") != "voice-chat":
            return False
        if payload.get("type") != "websocket":
            return False
        return True
    except jwt.InvalidTokenError:
        return False
```

### 2. Rate Limiting

```python
# rate_limiter.py
from fastapi import HTTPException, Request, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.websocket("/ws")
@limiter.limit("10/minute")  # 10 connections per minute
async def websocket_endpoint(websocket: WebSocket):
    # Verify authentication token
    token = websocket.query_params.get("token")
    if not await verify_websocket_token(websocket, token):
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Rate limiting by user
    user_id = get_user_id_from_token(token)
    await check_connection_limit(user_id)
```

## 📊 Monitoring and Analytics

### 1. Metrics Collection

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
REQUEST_COUNT = Counter('voice_chat_requests_total', 'Total voice chat requests', ['method', 'status'])
TTS_LATENCY = Histogram('tts_synthesis_duration_seconds', 'TTS synthesis latency', ['provider', 'voice'])
VOICE_CLONING_ATTEMPTS = Counter('voice_cloning_attempts_total', 'Voice cloning attempts', ['success'])
WEBSOCKET_CONNECTIONS = Gauge('websocket_connections_active', 'Active WebSocket connections')
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'TTS cache hit rate')

async def record_tts_metrics(provider: str, voice: str, duration: float, success: bool):
    TTS_LATENCY.labels(provider=provider, voice=voice).observe(duration)
    status = "success" if success else "error"
    REQUEST_COUNT.labels(method="tts_synthesis", status=status).inc()
```

### 2. Health Checks

```python
# health.py
from fastapi import APIRouter
import psutil
import asyncio

health_router = APIRouter()

@health_router.get("/health")
async def health_check():
    """Comprehensive health check"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0-enhanced",
        "services": {}
    }

    # Check TTS providers
    tts_status = await check_tts_providers()
    status["services"]["tts_providers"] = tts_status

    # Check audio system
    audio_status = await check_audio_system()
    status["services"]["audio_system"] = audio_status

    # Check system resources
    status["services"]["system"] = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
    }

    # Determine overall health
    all_healthy = all(
        service["status"] == "healthy"
        for service in status["services"].values()
    )

    if not all_healthy:
        status["status"] = "degraded"
        return JSONResponse(
            content=status,
            status_code=503
        )

    return status

async def check_tts_providers():
    """Check TTS provider health"""
    providers = {}

    # Check Coqui XTTS
    try:
        # Test synthesis
        await test_coqui_provider()
        providers["coqui_xtts_v2"] = {"status": "healthy", "latency": 0.15}
    except Exception as e:
        providers["coqui_xtts_v2"] = {"status": "unhealthy", "error": str(e)}

    # Check ElevenLabs
    if os.getenv("ELEVENLABS_API_KEY"):
        try:
            await test_elevenlabs_provider()
            providers["elevenlabs"] = {"status": "healthy", "latency": 0.12}
        except Exception as e:
            providers["elevenlabs"] = {"status": "unhealthy", "error": str(e)}

    return providers

@health_router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

## 🚀 Scaling Strategies

### 1. Horizontal Scaling

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: enhanced-voice-chat-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: enhanced-voice-chat
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 60
```

### 2. Provider-Based Scaling

```python
# provider_load_balancer.py
class ProviderLoadBalancer:
    def __init__(self):
        self.providers = {
            TTSProvider.COQUI_XTTS_V2: ProviderInstance(),
            TTSProvider.ELEVENLABS: ProviderInstance(),
            TTSProvider.AZURE_SPEECH: ProviderInstance(),
        }
        self.current_load = {p: 0 for p in self.providers}
        self.max_load_per_provider = 100

    async def get_provider_for_request(self, quality_tier: QualityTier):
        """Select provider based on current load and quality requirements"""
        # Filter providers that support quality tier
        suitable_providers = [
            p for p in self.providers
            if p.supports_quality_tier(quality_tier)
        ]

        # Select provider with lowest load
        best_provider = min(
            suitable_providers,
            key=lambda p: self.current_load[p]
        )

        # Update load
        self.current_load[best_provider] += 1

        return best_provider

    async def release_provider(self, provider: TTSProvider):
        """Release provider after request completion"""
        if self.current_load[provider] > 0:
            self.current_load[provider] -= 1
```

### 3. Caching Strategy

```python
# enhanced_cache.py
import redis
import pickle
import hashlib

class EnhancedTTSCache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 hour

    async def get_cached_audio(self, text: str, emotion: str, voice: str) -> Optional[np.ndarray]:
        """Get cached audio for given text, emotion, and voice"""
        cache_key = f"tts:{hashlib.md5(f'{text}|{emotion}|{voice}'.encode()).hexdigest()}"

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")

        return None

    async def cache_audio(self, text: str, emotion: str, voice: str, audio: np.ndarray):
        """Cache generated audio"""
        cache_key = f"tts:{hashlib.md5(f'{text}|{emotion}|{voice}'.encode()).hexdigest()}"

        try:
            serialized_audio = pickle.dumps(audio)
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                serialized_audio
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
```

## 🛠️ Deployment Commands

### Development Setup

```bash
# Clone and setup
git clone https://github.com/your-repo/RealtimeVoiceChat.git
cd RealtimeVoiceChat

# Copy enhanced configuration
cp config/enhanced_config.example.json config/enhanced_config.json

# Install dependencies
pip install -r requirements_enhanced.txt

# Download TTS models
python -m code.download_tts_models

# Run development server
python -m server --config=config/enhanced_config.json --debug
```

### Production Deployment

```bash
# Build and deploy with Docker
docker build -f Dockerfile.enhanced -t realtime-voice-chat:enhanced .
docker-compose -f docker-compose.enhanced.yml up -d

# Kubernetes deployment
kubectl apply -f k8s-enhanced-deployment.yaml

# Monitor deployment
kubectl get pods -l app=enhanced-voice-chat
kubectl logs -f deployment/enhanced-voice-chat
```

### Monitoring Setup

```bash
# Deploy monitoring stack
kubectl apply -f monitoring/
helm install prometheus prometheus-community/prometheus
helm install grafana grafana/grafana

# Setup Grafana dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana-dashboard.json
```

## 📈 Performance Optimization

### 1. Database Optimizations

```sql
-- Redis configuration for TTS caching
maxmemory 2gb
maxmemory-policy allkeys-lru

-- PostgreSQL for user data and voice profiles (if using)
CREATE TABLE voice_profiles (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    voice_name VARCHAR(255) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    voice_config JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_voice_profiles_user_id ON voice_profiles(user_id);
CREATE INDEX idx_voice_profiles_provider ON voice_profiles(provider);
```

### 2. CDN Configuration

```nginx
# Enhanced Nginx configuration for static assets
server {
    listen 443 ssl http2;
    server_name voice-chat.yourdomain.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;

    # TTS audio streaming
    location /tts/stream/ {
        proxy_pass http://voice-chat:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_cache_bypass $http_pragma;

        # WebSocket support for audio streaming
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static asset optimization
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        gzip_static on;
        gzip_types application/javascript application/json text/css audio/wav;
    }

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
```

## 🔍 Troubleshooting Production Issues

### Common Production Problems

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats realtime-voice-chat

   # Fix: Implement audio cleanup and reduce cache size
   ```

2. **TTS Latency Spikes**
   ```bash
   # Monitor TTS latency
   curl http://localhost:8000/metrics | grep tts_synthesis_duration_seconds

   # Fix: Enable adaptive quality and provider switching
   ```

3. **WebSocket Connection Issues**
   ```bash
   # Check WebSocket connections
   netstat -an | grep :8000

   # Fix: Adjust WebSocket timeout and load balancing
   ```

### Emergency Procedures

```bash
# Quick restart
docker-compose -f docker-compose.enhanced.yml restart

# Full service reset
docker-compose -f docker-compose.enhanced.yml down
docker-compose -f docker-compose.enhanced.yml up -d

# Emergency rollback to previous version
kubectl set image deployment/realtime-voice-chat realtime-voice-chat=your-registry/realtime-voice-chat:previous-stable
```

## 📞 Support and Maintenance

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Backup voice profiles and configurations
DATE=$(date +%Y%m%d_%H%M%S)

# Backup voice profiles
tar -czf backups/voice_profiles_$DATE.tar.gz /app/models/voice_profiles/

# Backup configurations
tar -czf backups/config_$DATE.tar.gz /app/config/

# Backup Redis cache
redis-cli --rdb /tmp/redis_backup_$DATE.rdb
mv /tmp/redis_backup_$DATE.rdb backups/

# Cleanup old backups (keep 30 days)
find backups/ -name "*.tar.gz" -mtime +30 -delete
```

### Update Procedures

```bash
#!/bin/bash
# update.sh - Rolling update procedure

echo "🔄 Starting rolling update..."

# Pull new version
docker pull your-registry/realtime-voice-chat:enhanced-$NEW_VERSION

# Update deployment one replica at a time
for replica in $(kubectl get pods -l app=enhanced-voice-chat -o jsonpath='{.items[*].metadata.name}'); do
    echo "Updating $replica..."
    kubectl set image pod/$replica realtime-voice-chat=your-registry/realtime-voice-chat:enhanced-$NEW_VERSION

    # Wait for pod to be ready
    kubectl wait --for=condition=ready pod/$replica --timeout=300s
    echo "✅ $replica updated successfully"
done

echo "✅ Rolling update completed"
```

---

**Deployment completed successfully! 🚀**

For production deployments, ensure you:
1. Set up proper monitoring and alerting
2. Configure backup and disaster recovery
3. Implement rate limiting and security
4. Test failover procedures
5. Monitor performance metrics continuously