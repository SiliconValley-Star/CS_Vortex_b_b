"""
Vortex Settings Configuration
Centralized configuration management with environment variable support
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import yaml
import json

# Load environment variables
load_dotenv()

@dataclass
class AIConfig:
    """AI and OpenRouter configuration."""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "anthropic/claude-3-sonnet-20240229"
    
    # Model tiers (V17.0 ULTIMATE - per .clinerules/all.md)
    # Primary: Uncensored Hermes for honest security assessment
    # Secondary: Gemini 2.0 for fast validation
    primary_models: List[str] = field(default_factory=lambda: [
        "nousresearch/hermes-3-llama-3.1-405b:free",  # Uncensored analysis
        "google/gemini-2.0-flash-exp:free"            # Fast validation
    ])
    fallback_models: List[str] = field(default_factory=lambda: [
        "meta-llama/llama-3.2-90b-vision-instruct:free",
        "google/gemini-flash-1.5-8b:free"
    ])
    emergency_models: List[str] = field(default_factory=lambda: [
        "meta-llama/llama-3.2-3b-instruct:free"
    ])
    
    # Model weights for consensus (per .clinerules - Hermes priority)
    hermes_weight: float = 0.7  # Uncensored analysis priority
    gemini_weight: float = 0.3  # Validation support
    
    # Request settings
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: int = 30
    max_concurrent_requests: int = 5
    retry_attempts: int = 3
    retry_delay_seconds: int = 2
    
    # Quality thresholds
    min_confidence: float = 0.6
    quality_threshold: float = 0.7
    
    # Cost management
    daily_cost_limit: float = 50.0
    monthly_cost_limit: float = 1000.0
    
    # Features
    analysis_enabled: bool = True
    behavioral_analysis_enabled: bool = True
    response_cache_enabled: bool = True
    cache_ttl_hours: int = 24

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///output/database/vortex.db"
    evidence_db_url: str = "sqlite:///output/database/evidence_integrity.db"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    # Backup settings
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    backup_retention_days: int = 30
    backup_directory: str = "output/database/backup"

@dataclass
class SecurityConfig:
    """Security and encryption configuration."""
    secret_key: str = ""
    jwt_secret_key: str = ""
    encryption_key: str = ""
    
    # Evidence integrity
    evidence_integrity_enabled: bool = True
    cryptographic_integrity_required: bool = True
    evidence_backup_encryption: bool = True
    
    # Session security
    session_timeout_hours: int = 24
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True

@dataclass
class WebConfig:
    """Web server configuration."""
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    secret_key: str = ""
    
    # CORS settings
    allowed_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:8080"
    ])
    allow_credentials: bool = True

@dataclass
class NetworkConfig:
    """Network and HTTP configuration."""
    timeout: int = 30
    max_redirects: int = 5
    max_retries: int = 3
    backoff_factor: float = 1.0
    
    # Rate limiting
    requests_per_minute: int = 120
    burst_size: int = 20
    backoff_seconds: int = 60
    
    # Proxy settings
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: str = "localhost,127.0.0.1"

@dataclass
class ScanningConfig:
    """Scanning and vulnerability detection configuration."""
    max_concurrent_scans: int = 5
    max_concurrent_requests_per_domain: int = 10
    max_concurrent_total_requests: int = 100
    
    # Timeouts
    scan_timeout_hours: int = 24
    request_timeout_seconds: int = 30
    verification_timeout_seconds: int = 60
    
    # WAF evasion
    waf_evasion_enabled: bool = True
    waf_detection_threshold: int = 3
    waf_bypass_techniques: List[str] = field(default_factory=lambda: [
        "encoding", "case_variation", "comment_injection", "fragmentation"
    ])
    
    # User agent rotation
    user_agent_rotation: bool = True
    random_headers_enabled: bool = True
    random_headers_count: int = 3

@dataclass
class QualityConfig:
    """Quality assurance configuration."""
    min_evidence_quality: float = 0.7
    min_ai_confidence: float = 0.6
    min_system_verification_confidence: float = 0.75
    submission_quality_threshold: float = 0.8
    
    # False positive filtering
    false_positive_filter_enabled: bool = True
    cdn_detection_enabled: bool = True
    waf_response_detection_enabled: bool = True
    
    # Requirements
    require_behavioral_evidence: bool = True
    require_cryptographic_integrity: bool = True

@dataclass
class LegalConfig:
    """Legal compliance configuration."""
    authorized_domains: List[str] = field(default_factory=list)
    legal_contact_email: str = ""
    responsible_disclosure_policy_url: str = ""
    
    # Data retention
    data_retention_days: int = 90
    evidence_retention_days: int = 365
    log_retention_days: int = 30
    
    # Privacy
    pii_detection_enabled: bool = True
    pii_redaction_enabled: bool = True
    gdpr_compliance_mode: bool = False
    
    # Legal disclaimers
    legal_disclaimer_required: bool = True
    terms_of_service_url: str = ""
    privacy_policy_url: str = ""

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "output/logs/application.log"
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    
    # Security logging
    security_log_file: str = "output/logs/security.log"
    audit_log_file: str = "output/logs/audit.log"
    compliance_log_file: str = "output/logs/compliance.log"
    
    # Metrics
    metrics_port: int = 9090
    health_check_interval_seconds: int = 30
    
    # Alerting
    alert_email_enabled: bool = False
    alert_email_to: str = ""

@dataclass
class SystemConfig:
    """System resource configuration."""
    max_memory_mb: int = 6000
    memory_cleanup_threshold: float = 0.85
    memory_emergency_threshold: float = 0.95
    
    # Performance limits
    max_cpu_percent: int = 80
    max_memory_percent: int = 85
    max_disk_usage_percent: int = 90
    
    # Async configuration
    async_worker_count: int = 10
    async_queue_size: int = 1000
    async_timeout_seconds: int = 300

@dataclass
class Settings:
    """Main settings container."""
    version: str = "1.0.0"
    environment: str = "development"
    debug_mode: bool = False
    
    # Configuration sections
    ai: AIConfig = field(default_factory=AIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    web: WebConfig = field(default_factory=WebConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    scanning: ScanningConfig = field(default_factory=ScanningConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    legal: LegalConfig = field(default_factory=LegalConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

def load_from_env() -> Settings:
    """Load settings from environment variables."""
    settings = Settings()
    
    # AI Configuration
    settings.ai.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    settings.ai.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", settings.ai.openrouter_base_url)
    settings.ai.default_model = os.getenv("OPENROUTER_DEFAULT_MODEL", settings.ai.default_model)
    
    # Parse model lists
    if os.getenv("AI_PRIMARY_MODELS"):
        settings.ai.primary_models = os.getenv("AI_PRIMARY_MODELS", "").split(",")
    if os.getenv("AI_FALLBACK_MODELS"):
        settings.ai.fallback_models = os.getenv("AI_FALLBACK_MODELS", "").split(",")
    
    settings.ai.max_tokens = int(os.getenv("AI_MAX_TOKENS", settings.ai.max_tokens))
    settings.ai.temperature = float(os.getenv("AI_TEMPERATURE", settings.ai.temperature))
    settings.ai.timeout_seconds = int(os.getenv("AI_TIMEOUT_SECONDS", settings.ai.timeout_seconds))
    
    # Database Configuration
    settings.database.url = os.getenv("DATABASE_URL", settings.database.url)
    settings.database.evidence_db_url = os.getenv("EVIDENCE_DB_URL", settings.database.evidence_db_url)
    settings.database.pool_size = int(os.getenv("DATABASE_POOL_SIZE", settings.database.pool_size))
    
    # Security Configuration
    settings.security.secret_key = os.getenv("SECRET_KEY", settings.security.secret_key)
    settings.security.jwt_secret_key = os.getenv("JWT_SECRET_KEY", settings.security.jwt_secret_key)
    settings.security.encryption_key = os.getenv("ENCRYPTION_KEY", settings.security.encryption_key)
    
    # Web Configuration
    settings.web.host = os.getenv("WEB_HOST", settings.web.host)
    settings.web.port = int(os.getenv("WEB_PORT", settings.web.port))
    settings.web.debug = os.getenv("WEB_DEBUG", "false").lower() == "true"
    settings.web.secret_key = os.getenv("WEB_SECRET_KEY", settings.web.secret_key)
    
    # Parse CORS origins
    if os.getenv("CORS_ALLOWED_ORIGINS"):
        settings.web.allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    
    # Network Configuration
    settings.network.timeout = int(os.getenv("HTTP_TIMEOUT", settings.network.timeout))
    settings.network.max_retries = int(os.getenv("HTTP_MAX_RETRIES", settings.network.max_retries))
    settings.network.requests_per_minute = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", settings.network.requests_per_minute))
    
    settings.network.http_proxy = os.getenv("HTTP_PROXY")
    settings.network.https_proxy = os.getenv("HTTPS_PROXY")
    settings.network.no_proxy = os.getenv("NO_PROXY", settings.network.no_proxy)
    
    # Scanning Configuration
    settings.scanning.max_concurrent_scans = int(os.getenv("MAX_CONCURRENT_SCANS", settings.scanning.max_concurrent_scans))
    settings.scanning.max_concurrent_requests_per_domain = int(os.getenv("MAX_CONCURRENT_REQUESTS_PER_DOMAIN", settings.scanning.max_concurrent_requests_per_domain))
    settings.scanning.waf_evasion_enabled = os.getenv("WAF_EVASION_ENABLED", "true").lower() == "true"
    
    # Quality Configuration
    settings.quality.min_evidence_quality = float(os.getenv("MIN_EVIDENCE_QUALITY", settings.quality.min_evidence_quality))
    settings.quality.submission_quality_threshold = float(os.getenv("SUBMISSION_QUALITY_THRESHOLD", settings.quality.submission_quality_threshold))
    settings.quality.require_cryptographic_integrity = os.getenv("REQUIRE_CRYPTOGRAPHIC_INTEGRITY", "true").lower() == "true"
    
    # Legal Configuration
    if os.getenv("AUTHORIZED_DOMAINS"):
        settings.legal.authorized_domains = os.getenv("AUTHORIZED_DOMAINS", "").split(",")
    settings.legal.legal_contact_email = os.getenv("LEGAL_CONTACT_EMAIL", settings.legal.legal_contact_email)
    settings.legal.pii_detection_enabled = os.getenv("PII_DETECTION_ENABLED", "true").lower() == "true"
    
    # Monitoring Configuration
    settings.monitoring.log_level = os.getenv("LOG_LEVEL", settings.monitoring.log_level)
    settings.monitoring.log_file = os.getenv("LOG_FILE", settings.monitoring.log_file)
    settings.monitoring.enabled = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    
    # System Configuration
    settings.system.max_memory_mb = int(os.getenv("MAX_MEMORY_MB", settings.system.max_memory_mb))
    settings.system.memory_cleanup_threshold = float(os.getenv("MEMORY_CLEANUP_THRESHOLD", settings.system.memory_cleanup_threshold))
    
    # Global settings
    settings.environment = os.getenv("ENVIRONMENT", settings.environment)
    settings.version = os.getenv("VERSION", settings.version)
    settings.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    return settings

def load_from_file(config_path: str) -> Settings:
    """Load settings from configuration file (YAML or JSON)."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    # Convert nested dict to Settings object
    settings = Settings()
    
    # Update settings from config data
    for section_name, section_data in config_data.items():
        if hasattr(settings, section_name) and isinstance(section_data, dict):
            section_obj = getattr(settings, section_name)
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
    
    return settings

async def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load settings from environment variables and optional config file."""
    # Start with environment variables
    settings = load_from_env()
    
    # Override with config file if provided
    if config_path and Path(config_path).exists():
        try:
            file_settings = load_from_file(config_path)
            # Merge file settings with env settings (env takes precedence)
            settings = merge_settings(file_settings, settings)
        except Exception as e:
            print(f"Warning: Failed to load config file {config_path}: {e}")
    
    # Validate critical settings
    validate_settings(settings)
    
    return settings

def merge_settings(base: Settings, override: Settings) -> Settings:
    """Merge two settings objects, with override taking precedence."""
    # This is a simplified merge - in production, you might want more sophisticated merging
    merged = Settings()
    
    # Copy base settings
    for field_name in base.__dataclass_fields__:
        setattr(merged, field_name, getattr(base, field_name))
    
    # Override with non-default values from override
    for field_name in override.__dataclass_fields__:
        override_value = getattr(override, field_name)
        # Only override if the value is not the default
        if override_value != getattr(Settings(), field_name):
            setattr(merged, field_name, override_value)
    
    return merged

def validate_settings(settings: Settings) -> None:
    """Validate critical settings and raise errors for missing required values."""
    errors = []
    
    # Validate AI configuration
    if not settings.ai.openrouter_api_key:
        errors.append("OPENROUTER_API_KEY is required")
    
    if not settings.ai.primary_models:
        errors.append("At least one primary AI model must be configured")
    
    # Validate security configuration
    if not settings.security.secret_key:
        errors.append("SECRET_KEY is required for security")
    
    # Validate legal configuration
    if not settings.legal.authorized_domains:
        print("Warning: No authorized domains configured - scanning will be restricted")
    
    # Validate system resources
    if settings.system.max_memory_mb < 1000:
        errors.append("MAX_MEMORY_MB must be at least 1000 MB")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

def get_config_template() -> Dict[str, Any]:
    """Get a configuration template for creating config files."""
    return {
        "ai": {
            "openrouter_api_key": "your_api_key_here",
            "primary_models": ["anthropic/claude-3-sonnet-20240229"],
            "fallback_models": ["anthropic/claude-3-haiku-20240307"],
            "max_tokens": 4096,
            "temperature": 0.1
        },
        "database": {
            "url": "sqlite:///output/database/vortex.db",
            "backup_enabled": True
        },
        "security": {
            "secret_key": "your-secret-key-here",
            "evidence_integrity_enabled": True
        },
        "web": {
            "host": "127.0.0.1",
            "port": 8080,
            "debug": False
        },
        "legal": {
            "authorized_domains": ["example.com"],
            "pii_detection_enabled": True
        },
        "quality": {
            "min_evidence_quality": 0.7,
            "require_cryptographic_integrity": True
        }
    }

# Global settings instance (loaded lazily)
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        import asyncio
        try:
            # Try to get current event loop
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, just load synchronously
            _settings = load_from_env()
        except RuntimeError:
            # No event loop running, use asyncio.run()
            _settings = asyncio.run(load_settings())
    return _settings
