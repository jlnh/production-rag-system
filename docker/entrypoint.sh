#!/bin/bash

# Production entrypoint script for RAG system

set -e

echo "Starting RAG Production System..."

# Wait for dependencies to be ready
echo "Waiting for dependencies..."

# Wait for Redis
if [ "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    while ! nc -z redis 6379; do
        sleep 1
    done
    echo "Redis is ready!"
fi

# Run database migrations if needed
echo "Running system initialization..."

# Start the application
echo "Starting application with command: $@"
exec "$@"