"""
VORTEX Scanning Configuration - V19.1
Scanner-specific configurations and tuning parameters

Defines per-scanner settings including:
- Payload limits and timeouts
- Detection thresholds
- Rate limiting per scanner type
- Technology-specific adjustments
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ScannerConfig:
    """Configuration for individual scanner."""
    enabled: bool = True
    max_payloads: int = 50
    timeout_seconds: float = 30.0
    max_retries: int = 2
    confidence_threshold: float = 0.65
    rate_limit_rps: float = 10.0
    follow_redirects: bool = True
    verify_ssl: bool = True


# Per-scanner configurations
SCANNER_CONFIGS: Dict[str, ScannerConfig] = {
    'sqli': ScannerConfig(
        max_payloads=80,
        timeout_seconds=45.0,
        confidence_threshold=0.70,
        rate_limit_rps=8.0,
    ),
    'xss': ScannerConfig(
        max_payloads=60,
        timeout_seconds=30.0,
        confidence_threshold=0.60,
        rate_limit_rps=12.0,
    ),
    'csrf': ScannerConfig(
        max_payloads=20,
        timeout_seconds=20.0,
        confidence_threshold=0.75,
        rate_limit_rps=15.0,
    ),
    'lfi': ScannerConfig(
        max_payloads=40,
        timeout_seconds=30.0,
        confidence_threshold=0.70,
        rate_limit_rps=10.0,
    ),
    'ssrf': ScannerConfig(
        max_payloads=30,
        timeout_seconds=60.0,
        confidence_threshold=0.75,
        rate_limit_rps=5.0,
    ),
    'ssti': ScannerConfig(
        max_payloads=35,
        timeout_seconds=30.0,
        confidence_threshold=0.80,
        rate_limit_rps=8.0,
    ),
    'xxe': ScannerConfig(
        max_payloads=25,
        timeout_seconds=45.0,
        confidence_threshold=0.75,
        rate_limit_rps=6.0,
    ),
    'file_upload': ScannerConfig(
        max_payloads=15,
        timeout_seconds=60.0,
        confidence_threshold=0.70,
        rate_limit_rps=3.0,
    ),
    'jwt': ScannerConfig(
        max_payloads=20,
        timeout_seconds=30.0,
        confidence_threshold=0.80,
        rate_limit_rps=10.0,
    ),
    'graphql': ScannerConfig(
        max_payloads=30,
        timeout_seconds=30.0,
        confidence_threshold=0.70,
        rate_limit_rps=8.0,
    ),
    'dom_xss': ScannerConfig(
        max_payloads=25,
        timeout_seconds=90.0,
        confidence_threshold=0.65,
        rate_limit_rps=2.0,
    ),
}


# Scan mode presets
SCAN_MODES: Dict[str, Dict[str, Any]] = {
    'passive': {
        'description': 'Minimal interaction, reconnaissance only',
        'max_payloads_multiplier': 0.2,
        'rate_limit_multiplier': 0.3,
        'enabled_scanners': ['csrf', 'jwt'],
    },
    'active': {
        'description': 'Standard scanning (default)',
        'max_payloads_multiplier': 1.0,
        'rate_limit_multiplier': 1.0,
        'enabled_scanners': ['sqli', 'xss', 'csrf', 'lfi', 'ssrf', 'jwt'],
    },
    'aggressive': {
        'description': 'Maximum coverage, all scanners',
        'max_payloads_multiplier': 2.0,
        'rate_limit_multiplier': 2.0,
        'enabled_scanners': list(SCANNER_CONFIGS.keys()),
    },
}


def get_scanner_config(scanner_type: str) -> ScannerConfig:
    """Get configuration for a specific scanner type."""
    return SCANNER_CONFIGS.get(scanner_type, ScannerConfig())


def get_mode_config(mode: str) -> Dict[str, Any]:
    """Get scan mode configuration."""
    return SCAN_MODES.get(mode, SCAN_MODES['active'])
