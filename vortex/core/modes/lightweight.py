"""
VORTEX Lightweight Mode - V21.0
Resource-optimized scanning mode for low-resource environments

FEATURES:
- Playwright optional (saves ~500MB RAM)
- Minimal scanner set (essential only)
- Aggressive resource limits
- Fast triage focus
- CPU/Memory optimized

USE CASES:
- Low-resource VPS
- Quick triage scans
- CI/CD integration
- Batch scanning
- Cost optimization

COMPARISON:
Standard Mode: 2-4GB RAM, Full features
Lightweight Mode: 500MB-1GB RAM, Essential features only
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class ScannerPriority(str, Enum):
    """Scanner priority levels."""
    ESSENTIAL = "essential"  # Must-have for basic security
    STANDARD = "standard"    # Good to have
    ADVANCED = "advanced"    # Resource-intensive
    OPTIONAL = "optional"    # Nice to have


@dataclass
class LightweightConfig:
    """Lightweight mode configuration."""
    
    # Feature flags
    enable_playwright: bool = False  # Disable Playwright by default (saves 500MB)
    enable_ai_analysis: bool = False  # Disable AI to save API costs
    enable_advanced_scanners: bool = False  # Only essential scanners
    
    # Resource limits (More aggressive than standard)
    max_memory_mb: int = 1024  # 1GB limit
    max_cpu_percent: int = 40  # 40% CPU max
    max_concurrent_requests: int = 5  # Only 5 concurrent
    request_delay_seconds: float = 2.0  # Slower but safer
    
    # Scanner selection
    essential_scanners: List[str] = None
    
    def __post_init__(self):
        """Initialize essential scanners list."""
        if self.essential_scanners is None:
            self.essential_scanners = [
                'sqli',      # SQL Injection (critical)
                'xss',       # XSS (critical)
                'lfi',       # LFI (high)
                'ssrf',      # SSRF (high)
                'csrf',      # CSRF (medium)
            ]


class LightweightMode:
    """
    Lightweight scanning mode for resource-constrained environments.
    
    Key differences from standard mode:
    - No Playwright (saves 500MB+ RAM)
    - No AI analysis (saves API costs)
    - Only essential scanners
    - Lower concurrency
    - Smaller memory footprint
    """
    
    # Scanner priority mapping
    SCANNER_PRIORITIES = {
        # Essential (Always run)
        'sqli': ScannerPriority.ESSENTIAL,
        'xss': ScannerPriority.ESSENTIAL,
        'lfi': ScannerPriority.ESSENTIAL,
        'ssrf': ScannerPriority.ESSENTIAL,
        'csrf': ScannerPriority.ESSENTIAL,
        
        # Standard (Run if resources allow)
        'ssti': ScannerPriority.STANDARD,
        'xxe': ScannerPriority.STANDARD,
        'file_upload': ScannerPriority.STANDARD,
        'jwt': ScannerPriority.STANDARD,
        
        # Advanced (Skip in lightweight mode)
        'dom_xss': ScannerPriority.ADVANCED,  # Requires Playwright
        'graphql': ScannerPriority.ADVANCED,  # Complex analysis
        
        # Optional (Skip in lightweight mode)
        'business_logic': ScannerPriority.OPTIONAL,
        'auth_matrix': ScannerPriority.OPTIONAL,
        'race_conditions': ScannerPriority.OPTIONAL,
    }
    
    def __init__(self, config: Optional[LightweightConfig] = None):
        """
        Initialize lightweight mode.
        
        Args:
            config: Lightweight mode configuration
        """
        self.config = config or LightweightConfig()
        self.active_scanners: Set[str] = set()
        
        # Determine active scanners based on config
        self._initialize_scanners()
        
        logger.info(
            f"Lightweight mode initialized: "
            f"{len(self.active_scanners)} scanners, "
            f"Playwright={'enabled' if self.config.enable_playwright else 'disabled'}, "
            f"AI={'enabled' if self.config.enable_ai_analysis else 'disabled'}"
        )
    
    def _initialize_scanners(self):
        """Initialize active scanners based on configuration."""
        # If essential_scanners explicitly defined, use only those
        if self.config.essential_scanners:
            self.active_scanners = set(self.config.essential_scanners)
            return
        
        # Otherwise, add essential scanners
        for scanner, priority in self.SCANNER_PRIORITIES.items():
            if priority == ScannerPriority.ESSENTIAL:
                self.active_scanners.add(scanner)
        
        # Add standard scanners if not ultra-lightweight
        if self.config.max_memory_mb >= 1536:  # 1.5GB+
            for scanner, priority in self.SCANNER_PRIORITIES.items():
                if priority == ScannerPriority.STANDARD:
                    self.active_scanners.add(scanner)
        
        # Add advanced scanners only if explicitly enabled
        if self.config.enable_advanced_scanners:
            for scanner, priority in self.SCANNER_PRIORITIES.items():
                if priority == ScannerPriority.ADVANCED:
                    # Check if Playwright required
                    if scanner == 'dom_xss' and not self.config.enable_playwright:
                        logger.warning("Skipping dom_xss: Playwright disabled")
                        continue
                    self.active_scanners.add(scanner)
    
    def get_active_scanners(self) -> List[str]:
        """Get list of active scanners for this mode."""
        return list(self.active_scanners)
    
    def is_scanner_enabled(self, scanner_name: str) -> bool:
        """Check if a scanner is enabled in lightweight mode."""
        return scanner_name in self.active_scanners
    
    def get_resource_limits(self) -> dict:
        """Get resource limits for lightweight mode."""
        return {
            'max_memory_mb': self.config.max_memory_mb,
            'max_cpu_percent': self.config.max_cpu_percent,
            'max_concurrent_requests': self.config.max_concurrent_requests,
            'request_delay_seconds': self.config.request_delay_seconds,
        }
    
    def should_use_ai(self) -> bool:
        """Check if AI analysis should be used."""
        return self.config.enable_ai_analysis
    
    def should_use_playwright(self) -> bool:
        """Check if Playwright should be used."""
        return self.config.enable_playwright
    
    def get_mode_summary(self) -> dict:
        """Get summary of lightweight mode configuration."""
        return {
            'mode': 'lightweight',
            'scanners': {
                'enabled': list(self.active_scanners),
                'count': len(self.active_scanners),
                'essential_only': not self.config.enable_advanced_scanners,
            },
            'features': {
                'playwright': self.config.enable_playwright,
                'ai_analysis': self.config.enable_ai_analysis,
                'advanced_scanners': self.config.enable_advanced_scanners,
            },
            'resources': {
                'max_memory_mb': self.config.max_memory_mb,
                'max_cpu_percent': self.config.max_cpu_percent,
                'max_concurrent': self.config.max_concurrent_requests,
                'request_delay': self.config.request_delay_seconds,
            },
            'estimated_memory': self._estimate_memory_usage(),
        }
    
    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage based on configuration."""
        base_memory = 300  # Base Vortex core
        
        # Add scanner memory
        scanner_memory = len(self.active_scanners) * 50
        
        # Add Playwright if enabled
        if self.config.enable_playwright:
            playwright_memory = 500
        else:
            playwright_memory = 0
        
        # Add AI if enabled
        if self.config.enable_ai_analysis:
            ai_memory = 200
        else:
            ai_memory = 0
        
        total = base_memory + scanner_memory + playwright_memory + ai_memory
        
        return f"{total}MB (base:{base_memory} + scanners:{scanner_memory} + playwright:{playwright_memory} + ai:{ai_memory})"


def create_lightweight_mode(
    max_memory_mb: int = 1024,
    enable_playwright: bool = False,
    enable_ai: bool = False,
    enable_advanced: bool = False
) -> LightweightMode:
    """
    Factory function to create lightweight mode.
    
    Args:
        max_memory_mb: Maximum memory limit in MB
        enable_playwright: Enable Playwright (adds ~500MB)
        enable_ai: Enable AI analysis (adds API costs)
        enable_advanced: Enable advanced scanners
        
    Returns:
        Configured LightweightMode instance
    """
    config = LightweightConfig(
        enable_playwright=enable_playwright,
        enable_ai_analysis=enable_ai,
        enable_advanced_scanners=enable_advanced,
        max_memory_mb=max_memory_mb,
    )
    
    return LightweightMode(config)


def create_ultra_lightweight_mode() -> LightweightMode:
    """
    Create ultra-lightweight mode (minimum resources).
    
    Configuration:
    - 512MB RAM limit
    - Only 3 essential scanners (SQLi, XSS, LFI)
    - No Playwright
    - No AI
    - Minimal concurrency
    """
    config = LightweightConfig(
        enable_playwright=False,
        enable_ai_analysis=False,
        enable_advanced_scanners=False,
        max_memory_mb=512,
        max_cpu_percent=30,
        max_concurrent_requests=3,
        request_delay_seconds=3.0,
        essential_scanners=['sqli', 'xss', 'lfi'],  # Absolute minimum
    )
    
    return LightweightMode(config)


def create_ci_cd_mode() -> LightweightMode:
    """
    Create CI/CD optimized mode.
    
    Configuration:
    - Fast execution
    - Essential scanners only
    - No interactive features
    - API-friendly
    """
    config = LightweightConfig(
        enable_playwright=False,  # CI/CD usually headless
        enable_ai_analysis=False,  # Deterministic results preferred
        enable_advanced_scanners=False,
        max_memory_mb=1024,
        max_cpu_percent=50,
        max_concurrent_requests=10,
        request_delay_seconds=1.0,
    )
    
    return LightweightMode(config)


# Global lightweight mode instance
_lightweight_mode: Optional[LightweightMode] = None


def get_lightweight_mode() -> LightweightMode:
    """Get global lightweight mode instance."""
    global _lightweight_mode
    if _lightweight_mode is None:
        _lightweight_mode = create_lightweight_mode()
    return _lightweight_mode


def set_lightweight_mode(mode: LightweightMode):
    """Set global lightweight mode instance."""
    global _lightweight_mode
    _lightweight_mode = mode


# Usage examples
if __name__ == "__main__":
    # Example 1: Standard lightweight mode
    mode1 = create_lightweight_mode()
    print("Standard Lightweight:")
    print(mode1.get_mode_summary())
    
    # Example 2: Ultra lightweight for VPS
    mode2 = create_ultra_lightweight_mode()
    print("\nUltra Lightweight:")
    print(mode2.get_mode_summary())
    
    # Example 3: CI/CD mode
    mode3 = create_ci_cd_mode()
    print("\nCI/CD Mode:")
    print(mode3.get_mode_summary())