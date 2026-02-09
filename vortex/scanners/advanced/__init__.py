"""
VORTEX Advanced Scanners Package
"""

# Lazy imports to avoid circular dependency
PLAYWRIGHT_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass


def get_dom_scanner():
    """Get DOM scanner instance (lazy import)."""
    from scanners.advanced.dom_scanner import get_dom_scanner as _get
    return _get()


async def scan_for_dom_xss(url, params=None):
    """Scan for DOM XSS (lazy import)."""
    from scanners.advanced.dom_scanner import scan_for_dom_xss as _scan
    return await _scan(url, params)
