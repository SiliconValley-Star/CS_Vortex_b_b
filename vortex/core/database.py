"""
VORTEX Database System - V17.0 ULTIMATE
Production-grade database with complete audit trail

FEATURES:
- SQLite with async support
- Complete audit trail for authority compliance
- Finding lifecycle tracking
- Authority validation history
- Evidence integrity records
- Automated backups
"""

import asyncio
import aiosqlite
import sqlite3
import logging
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

from domain.models import AssessmentResult, AIAnalysisResult, VerificationResult
from domain.enums import VerificationStatus, FindingSeverity, FindingType
from core.exceptions import (
    DatabaseError, DatabaseConnectionError, RecordNotFoundError,
    DatabaseIntegrityError, DatabaseMigrationError
)

# V21.0 - Performance Profiling Integration
try:
    from core.metrics import global_metrics
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logging.warning("Metrics module not available")

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Production database manager with authority compliance tracking.
    
    RESPONSIBILITIES:
    - Finding storage and retrieval
    - Authority compliance audit trail
    - Evidence integrity tracking
    - State transition history
    - Performance metrics
    - Automated backups
    """
    
    def __init__(self, database_path: str = "output/database/vortex.db"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # V21.0 - Metrics integration
        self.metrics = global_metrics if METRICS_ENABLED else None
        
        # Connection pool (simple implementation)
        self._connection: Optional[aiosqlite.Connection] = None
        self._connection_lock = asyncio.Lock()
        
        # Backup configuration
        self.backup_dir = self.database_path.parent / "backup"
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_interval_hours = 24
        self._last_backup: Optional[datetime] = None
        
    async def initialize(self):
        """Initialize database with schema."""
        logger.info(f"Initializing database: {self.database_path}")
        
        try:
            async with self._get_connection() as conn:
                await self._create_schema(conn)
                await self._create_indices(conn)
                await conn.commit()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection (async context manager)."""
        async with self._connection_lock:
            if self._connection is None:
                try:
                    self._connection = await aiosqlite.connect(
                        str(self.database_path),
                        timeout=30.0
                    )
                    # Enable foreign keys
                    await self._connection.execute("PRAGMA foreign_keys = ON")
                    # WAL mode for better concurrency
                    await self._connection.execute("PRAGMA journal_mode = WAL")
                except Exception as e:
                    raise DatabaseConnectionError(f"Failed to connect: {e}")
            
            yield self._connection
    
    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    async def _create_schema(self, conn: aiosqlite.Connection):
        """Create database schema."""
        
        # Findings table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                finding_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                
                -- Heuristic detection
                heuristic_score REAL,
                confidence_source TEXT,
                
                -- Evidence
                evidence TEXT,
                evidence_determinism_score REAL,
                vulnerable_parameter TEXT,
                payload TEXT,
                
                -- AI analysis (JSON)
                ai_analysis_json TEXT,
                
                -- System verification (JSON)
                verification_result_json TEXT,
                
                -- Authority compliance
                authority_level TEXT,
                authority_validated BOOLEAN DEFAULT 0,
                authority_validator TEXT,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                submitted_at TIMESTAMP,
                
                -- Metadata
                metadata_json TEXT
            )
        """)
        
        # State transitions audit trail
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS state_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_id TEXT NOT NULL,
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                reason TEXT,
                triggered_by TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (finding_id) REFERENCES findings(id)
            )
        """)
        
        # Authority validation history
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS authority_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_id TEXT NOT NULL,
                validation_type TEXT NOT NULL,
                passed BOOLEAN NOT NULL,
                validator TEXT NOT NULL,
                details_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (finding_id) REFERENCES findings(id)
            )
        """)
        
        # Evidence integrity records
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_integrity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_id TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                determinism_score REAL,
                behavioral_indicators_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (finding_id) REFERENCES findings(id)
            )
        """)
        
        # Health metrics (for monitoring)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_metadata_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System events log
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_category TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT,
                severity TEXT DEFAULT 'INFO',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    async def _create_indices(self, conn: aiosqlite.Connection):
        """Create database indices for performance."""
        
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status)",
            "CREATE INDEX IF NOT EXISTS idx_findings_created_at ON findings(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity)",
            "CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(finding_type)",
            "CREATE INDEX IF NOT EXISTS idx_state_transitions_finding ON state_transitions(finding_id)",
            "CREATE INDEX IF NOT EXISTS idx_authority_validations_finding ON authority_validations(finding_id)",
            "CREATE INDEX IF NOT EXISTS idx_health_metrics_name ON health_metrics(metric_name, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type, timestamp)"
        ]
        
        for index_sql in indices:
            await conn.execute(index_sql)
    
    async def save_finding(self, finding: AssessmentResult) -> bool:
        """
        Save or update finding with complete audit trail.
        
        Returns:
            True if successful
        """
        # V21.0 - Track query metrics
        query_start = datetime.utcnow()
        
        try:
            async with self._get_connection() as conn:
                # Serialize complex objects
                ai_analysis_json = None
                if finding.ai_analysis:
                    ai_analysis_json = json.dumps({
                        'model_used': finding.ai_analysis.model_used,
                        'verdict': finding.ai_analysis.verdict,
                        'confidence': finding.ai_analysis.confidence,
                        'exploitability': finding.ai_analysis.exploitability,
                        'impact': finding.ai_analysis.impact,
                        'reportability': finding.ai_analysis.reportability,
                        'reasoning': finding.ai_analysis.reasoning,
                        'success': finding.ai_analysis.success,
                        'is_fallback_result': finding.ai_analysis.is_fallback_result
                    })
                
                verification_json = None
                if finding.verification_result:
                    verification_json = json.dumps({
                        'success': finding.verification_result.success,
                        'match_type': finding.verification_result.match_type,
                        'confidence': finding.verification_result.confidence,
                        'matched_pattern': finding.verification_result.matched_pattern,
                        'response_time': finding.verification_result.response_time
                    })
                
                # Insert or update
                await conn.execute("""
                    INSERT OR REPLACE INTO findings (
                        id, url, finding_type, severity, status,
                        heuristic_score, confidence_source,
                        evidence, evidence_determinism_score,
                        vulnerable_parameter, payload,
                        ai_analysis_json, verification_result_json,
                        authority_level, authority_validated,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(finding.id),
                    finding.url,
                    finding.finding_type.value if finding.finding_type else '',
                    finding.severity.value if finding.severity else '',
                    finding.status.value if finding.status else '',
                    finding.heuristic_score,
                    finding.confidence_source if hasattr(finding, 'confidence_source') else None,
                    finding.evidence,
                    getattr(finding, 'evidence_determinism_score', None),
                    finding.vulnerable_parameter,
                    finding.payload,
                    ai_analysis_json,
                    verification_json,
                    getattr(finding, 'authority_level', None),
                    getattr(finding, 'authority_validated', False),
                    datetime.utcnow().isoformat()
                ))
                
                await conn.commit()
                
                # V21.0 - Record metrics
                if self.metrics:
                    duration = (datetime.utcnow() - query_start).total_seconds()
                    self.metrics.record_db_query(
                        query_type='insert_finding',
                        duration=duration
                    )
                
                logger.debug(f"Saved finding {finding.id} with status {finding.status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save finding {finding.id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to save finding: {e}")
    
    async def get_finding(self, finding_id: str) -> Optional[AssessmentResult]:
        """Retrieve finding by ID."""
        # V21.0 - Track query metrics
        query_start = datetime.utcnow()
        
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM findings WHERE id = ?",
                    (finding_id,)
                )
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                # V21.0 - Record metrics
                if self.metrics:
                    duration = (datetime.utcnow() - query_start).total_seconds()
                    self.metrics.record_db_query(
                        query_type='select_finding',
                        duration=duration
                    )
                
                return self._row_to_finding(row)
                
        except Exception as e:
            logger.error(f"Failed to get finding {finding_id}: {e}")
            raise DatabaseError(f"Failed to retrieve finding: {e}")
    
    async def get_findings_by_status(self,
                                    status: VerificationStatus,
                                    limit: int = 100) -> List[AssessmentResult]:
        """Get findings by status."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM findings WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit)
                )
                rows = await cursor.fetchall()
                
                return [self._row_to_finding(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get findings by status: {e}")
            raise DatabaseError(f"Failed to retrieve findings: {e}")
    
    async def get_recent_findings(self, limit: int = 100) -> List[AssessmentResult]:
        """
        Get most recent findings regardless of status.
        
        Args:
            limit: Maximum number of findings to retrieve
            
        Returns:
            List of findings ordered by creation time (newest first)
        """
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM findings ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
                rows = await cursor.fetchall()
                
                return [self._row_to_finding(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get recent findings: {e}")
            raise DatabaseError(f"Failed to retrieve recent findings: {e}")
    
    async def record_state_transition(self,
                                     finding_id: str,
                                     from_state: VerificationStatus,
                                     to_state: VerificationStatus,
                                     reason: str,
                                     triggered_by: str = "system"):
        """Record state transition in audit trail."""
        try:
            async with self._get_connection() as conn:
                await conn.execute("""
                    INSERT INTO state_transitions 
                    (finding_id, from_state, to_state, reason, triggered_by)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    finding_id,
                    from_state.value,
                    to_state.value,
                    reason,
                    triggered_by
                ))
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to record state transition: {e}")
    
    async def record_authority_validation(self,
                                         finding_id: str,
                                         validation_type: str,
                                         passed: bool,
                                         validator: str,
                                         details: Optional[Dict] = None):
        """Record authority validation in audit trail."""
        try:
            async with self._get_connection() as conn:
                await conn.execute("""
                    INSERT INTO authority_validations
                    (finding_id, validation_type, passed, validator, details_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    finding_id,
                    validation_type,
                    passed,
                    validator,
                    json.dumps(details) if details else None
                ))
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to record authority validation: {e}")
    
    async def get_finding_stats(self) -> Dict[str, int]:
        """Get finding distribution statistics."""
        try:
            async with self._get_connection() as conn:
                # Total findings
                cursor = await conn.execute("SELECT COUNT(*) FROM findings")
                total = (await cursor.fetchone())[0]
                
                # By status
                stats = {'total': total}
                for status in VerificationStatus:
                    cursor = await conn.execute(
                        "SELECT COUNT(*) FROM findings WHERE status = ?",
                        (status.value,)
                    )
                    count = (await cursor.fetchone())[0]
                    stats[status.value.lower()] = count
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get finding stats: {e}")
            return {'total': 0}
    
    def get_finding_count(self) -> int:
        """
        Get total count of findings.
        Synchronous wrapper for web API compatibility.
        """
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.database_path))
            cursor = conn.execute("SELECT COUNT(*) FROM findings")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Failed to get finding count: {e}")
            return 0
    
    def get_findings_by_status_sync(self) -> Dict[str, int]:
        """
        Get findings count by status.
        Synchronous wrapper for web API compatibility.
        """
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.database_path))
            
            result = {}
            for status in VerificationStatus:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM findings WHERE status = ?",
                    (status.value,)
                )
                result[status] = cursor.fetchone()[0]
            
            conn.close()
            return result
        except Exception as e:
            logger.error(f"Failed to get findings by status: {e}")
            return {}
    
    async def update_finding(self, finding: AssessmentResult) -> bool:
        """
        Update an existing finding in the database.
        
        Args:
            finding: AssessmentResult to update
            
        Returns:
            True if update was successful
        """
        try:
            # Use save_finding which does INSERT OR REPLACE
            return await self.save_finding(finding)
        except Exception as e:
            logger.error(f"Failed to update finding {finding.id}: {e}")
            return False
    
    async def record_health_metric(self, metric_name: str, value: float, 
                                   metadata: Optional[Dict] = None):
        """Record health metric for monitoring."""
        try:
            async with self._get_connection() as conn:
                await conn.execute("""
                    INSERT INTO health_metrics (metric_name, metric_value, metric_metadata_json)
                    VALUES (?, ?, ?)
                """, (
                    metric_name,
                    value,
                    json.dumps(metadata) if metadata else None
                ))
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to record health metric: {e}")
    
    async def log_system_event(self, event_type: str, category: str,
                               message: str, details: Optional[Dict] = None,
                               severity: str = "INFO"):
        """Log system event."""
        try:
            async with self._get_connection() as conn:
                await conn.execute("""
                    INSERT INTO system_events 
                    (event_type, event_category, message, details_json, severity)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    event_type,
                    category,
                    message,
                    json.dumps(details) if details else None,
                    severity
                ))
                await conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
    
    async def create_backup(self) -> Optional[Path]:
        """Create database backup."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"vortex_backup_{timestamp}.db"
            
            # Close connection for backup
            if self._connection:
                await self._connection.close()
                self._connection = None
            
            # Copy database file
            shutil.copy2(self.database_path, backup_path)
            
            self._last_backup = datetime.utcnow()
            logger.info(f"Database backup created: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    def _row_to_finding(self, row) -> AssessmentResult:
        """Convert database row to AssessmentResult."""
        # This is a simplified conversion - would need full implementation
        finding = AssessmentResult(
            id=row[0],  # id
            url=row[1],  # url
            finding_type=FindingType(row[2]) if row[2] else None,
            severity=FindingSeverity(row[3]) if row[3] else None,
            status=VerificationStatus(row[4]) if row[4] else None,
            heuristic_score=row[5],
            evidence=row[8],
            vulnerable_parameter=row[10],
            payload=row[11]
        )
        
        # Deserialize AI analysis
        if row[12]:  # ai_analysis_json
            try:
                ai_data = json.loads(row[12])
                finding.ai_analysis = AIAnalysisResult(**ai_data)
            except Exception as e:
                logger.error(f"Failed to deserialize AI analysis: {e}")
        
        # Deserialize verification result
        if row[13]:  # verification_result_json
            try:
                ver_data = json.loads(row[13])
                finding.verification_result = VerificationResult(**ver_data)
            except Exception as e:
                logger.error(f"Failed to deserialize verification result: {e}")
        
        return finding


# Global database manager instance
global_database_manager = DatabaseManager()


# Helper functions for web interface
async def init_database():
    """Initialize database (web interface helper)."""
    await global_database_manager.initialize()


def get_database() -> DatabaseManager:
    """Get global database manager instance (web interface helper)."""
    return global_database_manager