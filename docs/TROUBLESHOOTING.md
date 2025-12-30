# TheBoard Troubleshooting Guide

**Version:** 1.0
**Last Updated:** 2025-12-30

This guide helps you diagnose and fix common issues with TheBoard.

## Table of Contents

1. [Docker and Services](#docker-and-services)
2. [Database Issues](#database-issues)
3. [LLM API Issues](#llm-api-issues)
4. [Meeting Execution Problems](#meeting-execution-problems)
5. [Performance Issues](#performance-issues)
6. [Export and Data Issues](#export-and-data-issues)

---

## Docker and Services

### Problem: "Docker not running" Error

**Symptoms:**
```
Error: Cannot connect to the Docker daemon. Is the docker daemon running?
```

**Solutions:**

1. **Check Docker status:**
   ```bash
   docker --version
   docker ps
   ```

2. **Start Docker:**
   - **Linux**: `sudo systemctl start docker`
   - **macOS**: Open Docker Desktop app
   - **Windows**: Open Docker Desktop app

3. **Verify Docker Compose:**
   ```bash
   docker-compose --version
   ```

4. **Restart services:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### Problem: Container Fails to Start

**Symptoms:**
```
Error: Container exited with code 1
```

**Solutions:**

1. **Check logs:**
   ```bash
   docker-compose logs postgres
   docker-compose logs redis
   docker-compose logs rabbitmq
   ```

2. **Check port conflicts:**
   ```bash
   # PostgreSQL (5432)
   lsof -i :5432

   # Redis (6379)
   lsof -i :6379

   # RabbitMQ (5672, 15672)
   lsof -i :5672
   lsof -i :15672
   ```

3. **Kill conflicting processes or change ports in `compose.yml`:**
   ```yaml
   ports:
     - "5433:5432"  # Use 5433 instead of 5432
   ```

4. **Remove and recreate containers:**
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

### Problem: "Connection refused" to Services

**Symptoms:**
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.
```

**Solutions:**

1. **Verify services are running:**
   ```bash
   docker-compose ps
   ```

2. **Check service health:**
   ```bash
   docker-compose exec postgres pg_isready
   docker-compose exec redis redis-cli ping
   ```

3. **Verify .env configuration:**
   ```bash
   cat .env | grep -E "POSTGRES_HOST|REDIS_HOST|RABBITMQ_HOST"
   ```

4. **Ensure hosts match Docker network:**
   - If running in Docker: use service names (`postgres`, `redis`, `rabbitmq`)
   - If running locally: use `localhost` or `127.0.0.1`

---

## Database Issues

### Problem: "Meeting not found" Error

**Symptoms:**
```
ValueError: Meeting not found: 12345678-1234-1234-1234-123456789012
```

**Solutions:**

1. **List all meetings:**
   ```bash
   board run  # Interactive selector shows all meetings
   ```

2. **Check meeting ID format:**
   - Must be valid UUID: `12345678-1234-1234-1234-123456789012`
   - Case-sensitive

3. **Verify database connection:**
   ```bash
   docker-compose exec postgres psql -U theboard -d theboard -c "SELECT id, topic FROM meetings LIMIT 5;"
   ```

4. **Check for database corruption:**
   ```bash
   docker-compose exec postgres psql -U theboard -d theboard -c "SELECT COUNT(*) FROM meetings;"
   ```

### Problem: Database Migration Issues

**Symptoms:**
```
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedTable) relation "meetings" does not exist
```

**Solutions:**

1. **Run migrations:**
   ```bash
   alembic upgrade head
   ```

2. **Reset database (WARNING: destroys all data):**
   ```bash
   docker-compose down -v
   docker-compose up -d postgres
   alembic upgrade head
   ```

3. **Check migration status:**
   ```bash
   alembic current
   alembic history
   ```

### Problem: "Too many connections" Error

**Symptoms:**
```
psycopg2.OperationalError: FATAL: too many connections
```

**Solutions:**

1. **Check connection count:**
   ```bash
   docker-compose exec postgres psql -U theboard -d theboard -c "SELECT count(*) FROM pg_stat_activity;"
   ```

2. **Kill idle connections:**
   ```bash
   docker-compose exec postgres psql -U theboard -d theboard -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle';"
   ```

3. **Increase max connections in `compose.yml`:**
   ```yaml
   command: postgres -c max_connections=200
   ```

4. **Restart PostgreSQL:**
   ```bash
   docker-compose restart postgres
   ```

---

## LLM API Issues

### Problem: Rate Limit Exceeded

**Symptoms:**
```
openai.RateLimitError: Rate limit exceeded. Please retry after 20 seconds.
```

**Solutions:**

1. **Wait and retry:**
   - Most APIs have temporary rate limits
   - Wait the specified time and rerun

2. **Check API tier:**
   - Verify your API key has sufficient quota
   - Upgrade API tier if needed

3. **Reduce agent count:**
   ```bash
   board create --topic "..." --agent-count 3  # Instead of 8
   ```

4. **Use greedy strategy:**
   ```bash
   board create --topic "..." --strategy greedy  # Fewer API calls
   ```

5. **Implement retry logic** (already in code, but check logs):
   ```bash
   # Check logs for retry attempts
   tail -f logs/theboard.log | grep -i "retry"
   ```

### Problem: Invalid API Key

**Symptoms:**
```
openai.AuthenticationError: Incorrect API key provided
```

**Solutions:**

1. **Verify .env file:**
   ```bash
   cat .env | grep LLM_API_KEY
   ```

2. **Check for trailing spaces/newlines:**
   ```bash
   # Remove and recreate
   export LLM_API_KEY="your-actual-key-here"
   echo "LLM_API_KEY=$LLM_API_KEY" >> .env
   ```

3. **Test API key directly:**
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $LLM_API_KEY"
   ```

4. **Regenerate API key** from provider dashboard

### Problem: Model Not Available

**Symptoms:**
```
openai.InvalidRequestError: The model 'gpt-5' does not exist
```

**Solutions:**

1. **Check available models:**
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $LLM_API_KEY" \
     | jq '.data[].id'
   ```

2. **Update .env with valid model:**
   ```bash
   LLM_MODEL="gpt-4-turbo-preview"
   ```

3. **Use model override:**
   ```bash
   board create --topic "..." --model "gpt-3.5-turbo"
   ```

---

## Meeting Execution Problems

### Problem: Meeting Hangs or Freezes

**Symptoms:**
- CLI shows no progress for >5 minutes
- Live progress table stops updating

**Solutions:**

1. **Check logs for errors:**
   ```bash
   tail -f logs/theboard.log
   ```

2. **Verify database connection:**
   ```bash
   docker-compose exec postgres pg_isready
   ```

3. **Check Redis connection:**
   ```bash
   docker-compose exec redis redis-cli ping
   ```

4. **Kill and restart:**
   ```bash
   # Ctrl+C to stop
   # Then rerun with --rerun flag
   board run <meeting-id> --rerun
   ```

5. **Check for deadlocks:**
   ```bash
   docker-compose exec postgres psql -U theboard -d theboard -c "SELECT * FROM pg_locks WHERE NOT granted;"
   ```

### Problem: Convergence Not Detecting

**Symptoms:**
- Meeting runs all max rounds despite repetitive content
- Novelty scores remain high

**Solutions:**

1. **Review convergence thresholds in config:**
   ```bash
   cat .env | grep -i convergence
   ```

2. **Check novelty calculation:**
   - Ensure embeddings are working (Qdrant connection)
   - Verify embedding model is loaded

3. **Manually stop and review:**
   ```bash
   # Ctrl+C to stop
   board status <meeting-id>
   # Review convergence metrics
   ```

### Problem: "No agents available" Error

**Symptoms:**
```
ValueError: No agents available for selection
```

**Solutions:**

1. **Check agent pool:**
   ```bash
   docker-compose exec postgres psql -U theboard -d theboard -c "SELECT COUNT(*) FROM agents;"
   ```

2. **Re-seed agents:**
   ```bash
   python scripts/seed_agents.py
   ```

3. **Verify agents meet criteria:**
   - For auto-select: agents must have embeddings and relevance scores
   - For manual: agents must be active

---

## Performance Issues

### Problem: Slow Execution

**Symptoms:**
- Each round takes >5 minutes
- Context size grows very large

**Solutions:**

1. **Enable compression:**
   ```bash
   # Check compression is enabled in meeting
   board status <meeting-id> | grep -i compression
   ```

2. **Reduce agent count:**
   ```bash
   board create --topic "..." --agent-count 5  # Instead of 10
   ```

3. **Use greedy strategy:**
   ```bash
   board create --topic "..." --strategy greedy
   ```

4. **Check compression metrics:**
   ```bash
   board status <meeting-id> --metrics
   ```

5. **Verify Redis caching:**
   ```bash
   docker-compose exec redis redis-cli INFO | grep connected_clients
   ```

### Problem: High Memory Usage

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce context window:**
   - Compression should handle this automatically
   - Check compression is working

2. **Clear Redis cache:**
   ```bash
   docker-compose exec redis redis-cli FLUSHALL
   ```

3. **Restart services:**
   ```bash
   docker-compose restart
   ```

4. **Increase Docker memory limits** (Docker Desktop settings)

---

## Export and Data Issues

### Problem: Export Fails

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**

1. **Specify output path explicitly:**
   ```bash
   board export <meeting-id> --format markdown --output ./output/report.md
   ```

2. **Create output directory:**
   ```bash
   mkdir -p ./output
   ```

3. **Check file permissions:**
   ```bash
   ls -la ./output
   chmod 755 ./output
   ```

### Problem: Template Not Found

**Symptoms:**
```
jinja2.exceptions.TemplateNotFound: custom-template.j2
```

**Solutions:**

1. **Verify template location:**
   ```bash
   ls src/theboard/templates/exports/
   ```

2. **Use correct template name:**
   ```bash
   board export <meeting-id> --format template --template "executive-summary.j2"
   ```

3. **Create custom template** in `src/theboard/templates/exports/`

---

## Getting More Help

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
board run <meeting-id>
```

### Check System Status

```bash
# Docker
docker-compose ps

# Postgres
docker-compose exec postgres pg_isready

# Redis
docker-compose exec redis redis-cli ping

# RabbitMQ
curl -u guest:guest http://localhost:15672/api/overview

# TheBoard version
board version
```

### Collect Diagnostic Info

```bash
# System info
uname -a
docker --version
python --version

# Service logs
docker-compose logs --tail=100 > diagnostic.log

# TheBoard logs
cat logs/theboard.log >> diagnostic.log
```

### Report Issues

When reporting issues, include:

1. TheBoard version: `board version`
2. Python version: `python --version`
3. Docker version: `docker --version`
4. Error message (full stack trace)
5. Steps to reproduce
6. Relevant logs

**GitHub Issues**: <https://github.com/your-repo/theboard/issues>

---

**Still stuck?** Check the [User Guide](./USER_GUIDE.md) or [Developer Documentation](./DEVELOPER.md).
