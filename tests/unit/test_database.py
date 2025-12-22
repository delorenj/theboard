"""Unit tests for database connection and session management."""

from unittest.mock import MagicMock, patch

import pytest

from theboard.database import get_sync_db, init_db


class TestDatabaseSessions:
    """Test database session management."""

    def test_get_sync_db_context_manager(self):
        """Test that get_sync_db returns a context manager."""
        with patch("theboard.database.SyncSessionLocal") as mock_session_factory:
            mock_session = MagicMock()
            mock_session_factory.return_value = mock_session

            # Use as context manager
            with get_sync_db() as db:
                assert db is mock_session

            # Verify session was closed
            mock_session.close.assert_called_once()

    def test_get_sync_db_closes_on_exception(self):
        """Test that session is closed even when exception occurs."""
        with patch("theboard.database.SyncSessionLocal") as mock_session_factory:
            mock_session = MagicMock()
            mock_session_factory.return_value = mock_session

            # Simulate exception in context
            with pytest.raises(ValueError):
                with get_sync_db() as db:
                    raise ValueError("Test error")

            # Verify session was still closed
            mock_session.close.assert_called_once()

    def test_init_db(self):
        """Test database initialization."""
        with patch("theboard.database.Base") as mock_base:
            init_db()

            # Verify metadata.create_all was called
            mock_base.metadata.create_all.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
