# Meeting Logging

Logs are now automatically generated for every meeting.

- **Location:** `logs/meetings/`
- **Format:** Markdown (`meeting_YYYYMMDD_HHMMSS_{topic}_{id}.md`)
- **Content:** Full transcript, metrics, and summary.

You can also manually export logs using:
```bash
board export <meeting_id> --format markdown
```
