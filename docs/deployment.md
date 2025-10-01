# Production Deployment Guide

This guide covers deploying the RAG Production System to various production environments.

## Deployment Options

- [Docker Deployment](#docker-deployment) - Containerized deployment
- [Kubernetes Deployment](#kubernetes-deployment) - Orchestrated container deployment
- [Cloud Deployment](#cloud-deployment) - AWS, GCP, Azure
- [Traditional Server](#traditional-server-deployment) - VM or bare metal

## Pre-Deployment Checklist

### Infrastructure Requirements

- [ ] **Compute Resources**
  - CPU: 4+ cores recommended
  - RAM: 8GB+ recommended
  - Storage: 20GB+ for application and logs

- [ ] **External Services**
  - Redis server (6.0+)
  - Vector database (Pinecone/Weaviate/ChromaDB)
  - Load balancer (for multiple instances)

- [ ] **Network & Security**
  - HTTPS certificates
  - Firewall rules configured
  - VPN/VPC setup (if required)

### Configuration Validation

```bash
# Validate configuration
python -c "
from rag_system.config import validate_config
errors = validate_config()
if errors:
    print('❌ Configuration errors:')
    for error in errors: print(f'  - {error}')
else:
    print('✅ Configuration valid')
"

# Test API keys
python -c "
import openai
import os
try:
    openai.Model.list()
    print('✅ OpenAI API key valid')
except:
    print('❌ OpenAI API key invalid')
"
```

### Quality Gate Validation

```bash
# Run quality gates before deployment
python -c "
from rag_system.evaluation import QualityGate
from rag_system.api.dependencies import ProductionRAGService
import asyncio

async def validate():
    service = ProductionRAGService()
    await service.initialize()

    quality_gate = QualityGate()
    result = quality_gate.run_health_tests(service)

    if result.passed:
        print('✅ Quality gate passed - ready for deployment')
    else:
        print('❌ Quality gate failed - deployment blocked')
        for issue in result.details.get('issues', []):
            print(f'  - {issue}')

    await service.cleanup()

asyncio.run(validate())
"
```

## Docker Deployment

### Single Container Deployment

```bash
# Build production image
docker build -f docker/Dockerfile --target production -t rag-system:latest .

# Run with environment variables
docker run -d \
  --name rag-system \
  -p 8000:8000 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e PINECONE_API_KEY="${PINECONE_API_KEY}" \
  -e PINECONE_ENVIRONMENT="${PINECONE_ENVIRONMENT}" \
  -e REDIS_URL="redis://redis:6379" \
  --restart unless-stopped \
  rag-system:latest
```

### Docker Compose Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  rag-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
      target: production
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

volumes:
  redis_data:
```

```bash
# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs rag-api

# Scale API instances
docker-compose -f docker-compose.prod.yml up -d --scale rag-api=3
```

### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream rag_backend {
        server rag-api:8000;
        # Add more servers for load balancing
        # server rag-api-2:8000;
        # server rag-api-3:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name yourdomain.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000";

        # API endpoints
        location / {
            limit_req zone=api burst=20 nodelay;

            proxy_pass http://rag_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check (no rate limiting)
        location /health {
            proxy_pass http://rag_backend;
            access_log off;
        }

        # Metrics (restrict access)
        location /metrics {
            allow 10.0.0.0/8;   # Internal networks
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;

            proxy_pass http://rag_backend;
        }
    }
}
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  REDIS_URL: "redis://redis-service:6379"
  VECTOR_STORE_BACKEND: "pinecone"
  EMBEDDING_MODEL: "text-embedding-3-small"
  LLM_MODEL: "gpt-4.1"
```

### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
data:
  # Base64 encoded values
  OPENAI_API_KEY: <base64-encoded-openai-key>
  PINECONE_API_KEY: <base64-encoded-pinecone-key>
  API_KEYS: <base64-encoded-api-keys>
```

```bash
# Create secrets
kubectl create secret generic rag-secrets \
  --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
  --from-literal=PINECONE_API_KEY="${PINECONE_API_KEY}" \
  --from-literal=API_KEYS="${API_KEYS}" \
  -n rag-system
```

### Redis Deployment

```yaml
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: rag-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
        command: ["redis-server", "--appendonly", "yes"]
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: rag-system
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: rag-system
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

### RAG API Deployment

```yaml
# k8s/rag-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: rag-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: OPENAI_API_KEY
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: PINECONE_API_KEY
        - name: API_KEYS
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: API_KEYS
        envFrom:
        - configMapRef:
            name: rag-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
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
  name: rag-api-service
  namespace: rag-system
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-system
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - yourdomain.com
    secretName: rag-tls
  rules:
  - host: yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 3
  maxReplicas: 10
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
```

### Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n rag-system
kubectl get services -n rag-system
kubectl get ingress -n rag-system

# View logs
kubectl logs -f deployment/rag-api -n rag-system

# Scale deployment
kubectl scale deployment rag-api --replicas=5 -n rag-system
```

## Cloud Deployment

### AWS ECS Deployment

```json
{
  "family": "rag-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "rag-api",
      "image": "your-registry/rag-system:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://your-elasticache-endpoint:6379"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-system",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### GCP Cloud Run Deployment

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: rag-system
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/your-project/rag-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: production
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
```

```bash
# Deploy to Cloud Run
gcloud run deploy rag-system \
  --image gcr.io/your-project/rag-system:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=production \
  --set-secrets OPENAI_API_KEY=openai-secret:latest
```

### Azure Container Instances

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group rag-rg \
  --name rag-system \
  --image your-registry.azurecr.io/rag-system:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    ENVIRONMENT=production \
    REDIS_URL=redis://your-redis.redis.cache.windows.net:6380 \
  --secure-environment-variables \
    OPENAI_API_KEY=$OPENAI_API_KEY \
    PINECONE_API_KEY=$PINECONE_API_KEY \
  --restart-policy Always
```

## Traditional Server Deployment

### Systemd Service

```ini
# /etc/systemd/system/rag-system.service
[Unit]
Description=RAG Production System
After=network.target redis.service

[Service]
Type=simple
User=rag
Group=rag
WorkingDirectory=/opt/rag-system
Environment=PATH=/opt/rag-system/venv/bin
EnvironmentFile=/opt/rag-system/.env
ExecStart=/opt/rag-system/venv/bin/uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000 --workers 4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/opt/rag-system/logs /opt/rag-system/data

[Install]
WantedBy=multi-user.target
```

```bash
# Install and start service
sudo systemctl daemon-reload
sudo systemctl enable rag-system
sudo systemctl start rag-system

# Check status
sudo systemctl status rag-system

# View logs
sudo journalctl -u rag-system -f
```

### Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Create ecosystem file
cat > ecosystem.config.js << EOF
module.exports = {
  apps: [{
    name: 'rag-system',
    script: 'uvicorn',
    args: 'rag_system.api.main:app --host 0.0.0.0 --port 8000',
    instances: 4,
    exec_mode: 'cluster',
    env: {
      ENVIRONMENT: 'production'
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true
  }]
}
EOF

# Start with PM2
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save
pm2 startup
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'rag-system'
  static_configs:
  - targets: ['rag-api:8000']
  metrics_path: /metrics
  scrape_interval: 30s

- job_name: 'redis'
  static_configs:
  - targets: ['redis:6379']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RAG System Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# alerts.yml
groups:
- name: rag-system
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"

  - alert: ServiceDown
    expr: up{job="rag-system"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "RAG system is down"
```

## Blue-Green Deployment

```bash
#!/bin/bash
# blue-green-deploy.sh

NEW_VERSION=$1
CURRENT_COLOR=$(kubectl get service rag-api-service -o jsonpath='{.spec.selector.version}')

if [ "$CURRENT_COLOR" = "blue" ]; then
    NEW_COLOR="green"
else
    NEW_COLOR="blue"
fi

echo "Deploying version $NEW_VERSION to $NEW_COLOR environment..."

# Update deployment with new version
kubectl set image deployment/rag-api-$NEW_COLOR rag-api=rag-system:$NEW_VERSION

# Wait for rollout
kubectl rollout status deployment/rag-api-$NEW_COLOR

# Run health checks
kubectl exec deployment/rag-api-$NEW_COLOR -- curl -f http://localhost:8000/health

# Switch traffic
kubectl patch service rag-api-service -p '{"spec":{"selector":{"version":"'$NEW_COLOR'"}}}'

echo "Deployment complete. Traffic switched to $NEW_COLOR"
```

## Rollback Procedures

### Docker Rollback

```bash
# Tag current version as rollback
docker tag rag-system:latest rag-system:rollback

# Deploy previous version
docker run -d --name rag-system-new rag-system:previous-version

# Switch traffic (update load balancer)
# Stop old container
docker stop rag-system
docker rm rag-system
docker rename rag-system-new rag-system
```

### Kubernetes Rollback

```bash
# View rollout history
kubectl rollout history deployment/rag-api

# Rollback to previous version
kubectl rollout undo deployment/rag-api

# Rollback to specific revision
kubectl rollout undo deployment/rag-api --to-revision=2

# Check rollback status
kubectl rollout status deployment/rag-api
```

## Security Considerations

### Production Security Checklist

- [ ] **Network Security**
  - Use HTTPS/TLS encryption
  - Configure firewalls and security groups
  - Implement VPN/VPC isolation
  - Enable DDoS protection

- [ ] **Application Security**
  - Rotate API keys regularly
  - Use strong authentication
  - Implement rate limiting
  - Validate all inputs
  - Enable security headers

- [ ] **Infrastructure Security**
  - Use least privilege access
  - Enable audit logging
  - Implement secrets management
  - Regular security updates
  - Vulnerability scanning

### Secrets Management

```bash
# Using HashiCorp Vault
vault kv put secret/rag-system \
  openai_api_key="$OPENAI_API_KEY" \
  pinecone_api_key="$PINECONE_API_KEY"

# Using AWS Secrets Manager
aws secretsmanager create-secret \
  --name rag-system/openai-key \
  --secret-string "$OPENAI_API_KEY"

# Using Kubernetes secrets
kubectl create secret generic rag-secrets \
  --from-literal=openai-key="$OPENAI_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Backup and Disaster Recovery

### Data Backup

```bash
# Backup Redis data
redis-cli --rdb /backup/redis-$(date +%Y%m%d).rdb

# Backup vector database
# (Implementation depends on your vector store)

# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz .env config.yml k8s/
```

### Disaster Recovery Plan

1. **Recovery Time Objective (RTO)**: 15 minutes
2. **Recovery Point Objective (RPO)**: 1 hour
3. **Backup frequency**: Daily
4. **Geographic redundancy**: Multi-region deployment

## Performance Optimization

### Production Tuning

```bash
# Optimize for production
export UVICORN_WORKERS=4
export UVICORN_WORKER_CONNECTIONS=1000
export UVICORN_KEEPALIVE=2

# Redis optimization
echo 'vm.overcommit_memory = 1' >> /etc/sysctl.conf
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf

# File descriptor limits
echo '* soft nofile 65535' >> /etc/security/limits.conf
echo '* hard nofile 65535' >> /etc/security/limits.conf
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Basic load test
ab -n 1000 -c 10 http://localhost:8000/health

# Using wrk for more advanced testing
wrk -t12 -c400 -d30s http://localhost:8000/health
```

## Troubleshooting

### Common Issues

```bash
# Check application logs
kubectl logs -f deployment/rag-api
docker logs rag-system
journalctl -u rag-system -f

# Check resource usage
kubectl top pods
docker stats
htop

# Check network connectivity
kubectl exec -it pod/rag-api -- curl http://redis-service:6379
telnet redis-host 6379

# Check API endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:8000/metrics
```

### Performance Issues

```bash
# Monitor response times
curl -w "%{time_total}\n" -o /dev/null -s http://localhost:8000/health

# Check memory usage
kubectl exec -it pod/rag-api -- ps aux
kubectl exec -it pod/rag-api -- free -h

# Profile application
python -m cProfile -o profile.stats your_app.py
```

This comprehensive deployment guide covers all major deployment scenarios. Choose the option that best fits your infrastructure and requirements.