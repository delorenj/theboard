# Container Deployment Issues - Phase 2

## Summary

TheBoard container successfully builds with Bloodbank integration but fails at runtime due to Docker DNS resolution issues preventing database connectivity.

## Completed Successfully

### Container Build
- ✅ Bloodbank dependency added to container image
- ✅ Build context adjusted to include both `theboard/` and `bloodbank/` directories
- ✅ Docker build completes without errors (54.6s build time)
- ✅ All dependencies installed in container `.venv`

### Infrastructure Services
- ✅ PostgreSQL container running and healthy (port 5433)
- ✅ Redis container running and healthy (port 6380)
- ✅ RabbitMQ container running and healthy (ports 5673, 15673)
- ✅ Qdrant container running (ports 6335, 6336 - changed to avoid conflicts)

### Code Integration
- ✅ Lifecycle event emission implemented in `src/theboard/api.py`:
  - `service.registered` event on FastAPI startup
  - `service.health` event every 60 seconds via background task
- ✅ Bloodbank emitter methods implemented:
  - `emit_service_registered()` at line 251
  - `emit_service_health()` at line 307
- ✅ Environment variable `THEBOARD_EVENT_EMITTER=rabbitmq` configured

## Current Blocking Issue

### DNS Resolution Failure

**Symptom:**
```
psycopg.OperationalError: failed to resolve host 'postgres':
[Errno -3] Temporary failure in name resolution
```

**Impact:**
- Container cannot resolve Docker service names (`postgres`, `redis`, `rabbitmq`)
- Both Alembic migrations and API startup fail immediately
- Container enters crash-loop restart cycle

**Context:**
- All containers on same Docker network: `trunk-main_theboard-network`
- Other services resolve each other successfully
- Issue specific to theboard-app container
- DNS works correctly on host system

## Attempted Fixes

1. **Removed healthcheck dependency for Qdrant** - Healthcheck used `wget` which doesn't exist in image
2. **Changed Qdrant ports** - 6333→6335, 6334→6336 to avoid host conflicts
3. **Fixed build context** - Changed from `.` to `../..` to include bloodbank directory
4. **Removed volume mount** - Eliminated `.:/app` volume that caused venv conflicts
5. **Changed command from `uv run` to direct venv activation** - Avoids runtime dependency checks
6. **Restarted Docker daemon** - Cleared stale networking state
7. **Removed Alembic migrations from startup** - Isolated DNS issue (still fails)

## System-Level Investigation Needed

### Potential Root Causes

1. **systemd-resolved issues** - DNS stub resolver may be interfering with Docker's DNS
2. **Docker daemon state corruption** - Networking subsystem may need full reset
3. **iptables rules** - Previous containers may have left stale rules
4. **Docker network driver issues** - Bridge network may need recreation with different subnet

### Diagnostics to Run

```bash
# Check Docker DNS configuration
docker exec theboard-app cat /etc/resolv.conf

# Test DNS from within container
docker exec theboard-app getent hosts postgres
docker exec theboard-app nslookup postgres

# Inspect network configuration
docker network inspect trunk-main_theboard-network

# Check iptables rules
sudo iptables -L -n -v | grep -i docker

# Verify systemd-resolved status
systemctl status systemd-resolved
resolvectl status
```

## Current Configuration Files

### compose.yml (relevant sections)

```yaml
services:
  theboard:
    build:
      context: ../..  # Build from 33GOD root to access bloodbank
      dockerfile: theboard/trunk-main/Dockerfile
    environment:
      - DATABASE_URL=postgresql+psycopg://theboard:theboard_dev_pass@postgres:5432/theboard
      - REDIS_URL=redis://:theboard_redis_pass@redis:6379
      - RABBITMQ_URL=amqp://theboard:theboard_rabbit_pass@rabbitmq:5672
      - THEBOARD_EVENT_EMITTER=rabbitmq
    command: >
      sh -c "
        . /app/.venv/bin/activate &&
        uvicorn theboard.api:app --host 0.0.0.0 --port 8000 --reload
      "
    networks:
      - theboard-network
```

### Dockerfile (relevant sections)

```dockerfile
# Copy Bloodbank dependency first (from sibling directory)
COPY bloodbank/trunk-main /bloodbank

# Copy application code
COPY theboard/trunk-main .

# Install dependencies (Bloodbank will be installed from /bloodbank)
RUN uv sync --frozen
```

### pyproject.toml

```toml
dependencies = [
  # ... other deps
  "bloodbank @ file:///bloodbank",
]
```

## Workaround: Local Development

While container deployment is blocked, lifecycle events can be tested locally:

```bash
# Install Bloodbank for local dev
cd /home/delorenj/code/33GOD/bloodbank/trunk-main
uv pip install -e .

# Start TheBoard API locally
cd /home/delorenj/code/33GOD/theboard/trunk-main
THEBOARD_EVENT_EMITTER=rabbitmq uv run uvicorn theboard.api:app --host 0.0.0.0 --port 8001
```

**Limitation:** Local development requires different Bloodbank dependency path in pyproject.toml:
- Container: `file:///bloodbank`
- Local: `file:///home/delorenj/code/33GOD/bloodbank/trunk-main`

## Verification Checklist (Once DNS Fixed)

When container deployment is working:

- [ ] Container starts without errors
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] Bloodbank status shows "connected" in health response
- [ ] RabbitMQ receives `theboard.service.registered` event
- [ ] RabbitMQ receives `theboard.service.health` events every 60 seconds
- [ ] Candybar visualizes TheBoard node in service graph
- [ ] Meeting trigger consumer can create meetings via Bloodbank events

## Next Steps

1. **System-level DNS investigation** - Run diagnostic commands above to identify root cause
2. **Consider alternative networking** - Test with `host` network mode or custom bridge
3. **File upstream issue** - If Docker bug, report to Docker GitHub with reproduction steps
4. **Document container-vs-local divergence** - Create guide for managing different dependency paths

## Related Documentation

- Phase 2 completion commit: Tag `phase-2-complete`
- Developer retrospective: `docs/33GOD_SERVICE_INTEGRATION_RETROSPECTIVE.md`
- Service registry: `/home/delorenj/code/33GOD/services/registry.yaml`
