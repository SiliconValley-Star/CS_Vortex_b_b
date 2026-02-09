"""
Test suite for DOM-based XSS Scanner (Playwright)
Tests browser-based XSS detection and DOM manipulation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from scanners.advanced.dom_scanner import PlaywrightDOMScanner, DOMScanResult, PLAYWRIGHT_AVAILABLE
from domain.enums import FindingType, FindingSeverity


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
class TestDOMScanner:
    """Test DOM Scanner functionality (requires Playwright)."""
    
    @pytest.fixture
    def scanner(self):
        return PlaywrightDOMScanner()
    
    def test_scanner_initialization(self, scanner):
        """Test scanner initializes correctly."""
        assert scanner.browser is None
        assert scanner.context is None
        assert scanner.timeout == 30000
        assert isinstance(scanner.results, list)
    
    def test_xss_payloads_defined(self, scanner):
        """Test XSS payloads are properly defined."""
        assert len(PlaywrightDOMScanner.XSS_PAYLOADS) > 0
        # Should have script tags
        assert any('<script>' in p for p in PlaywrightDOMScanner.XSS_PAYLOADS)
        # Should have event handlers
        assert any('onerror' in p for p in PlaywrightDOMScanner.XSS_PAYLOADS)
        # Should have VORTEX marker
        assert any('__VORTEX_XSS__' in p for p in PlaywrightDOMScanner.XSS_PAYLOADS)
    
    def test_dom_sinks_defined(self, scanner):
        """Test DOM sinks are defined."""
        assert len(PlaywrightDOMScanner.DOM_SINKS) > 0
        assert 'innerHTML' in PlaywrightDOMScanner.DOM_SINKS
        assert 'eval' in PlaywrightDOMScanner.DOM_SINKS
        assert 'document.write' in PlaywrightDOMScanner.DOM_SINKS
    
    @pytest.mark.asyncio
    async def test_initialize_browser(self, scanner):
        """Test browser initialization."""
        with patch('scanners.advanced.dom_scanner.async_playwright') as mock_pw:
            mock_playwright = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            
            mock_pw.return_value.start = AsyncMock(return_value=mock_playwright)
            mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            
            result = await scanner.initialize()
            
            assert result is True
            assert scanner.browser is not None
    
    @pytest.mark.asyncio
    async def test_close_browser(self, scanner):
        """Test browser cleanup."""
        scanner.browser = AsyncMock()
        scanner.context = AsyncMock()
        
        await scanner.close()
        
        scanner.context.close.assert_called_once()
        scanner.browser.close.assert_called_once()
    
    def test_dom_scan_result_creation(self):
        """Test DOMScanResult dataclass."""
        result = DOMScanResult(
            url='https://example.com',
            vulnerability_type='DOM_XSS',
            payload='<script>alert(1)</script>',
            injection_point='param',
            evidence='XSS detected',
            severity='HIGH',
            confirmed=True
        )
        
        assert result.url == 'https://example.com'
        assert result.confirmed is True
        assert result.severity == 'HIGH'
    
    @pytest.mark.asyncio
    async def test_scan_url_no_browser(self, scanner):
        """Test scan when browser not initialized."""
        with patch.object(scanner, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = False
            
            results = await scanner.scan_url('https://example.com')
            
            assert results == []
    
    def test_convert_to_findings(self, scanner):
        """Test conversion of scan results to findings."""
        scan_results = [
            DOMScanResult(
                url='https://example.com',
                vulnerability_type='DOM_XSS',
                payload='<script>alert(1)</script>',
                injection_point='q',
                evidence='XSS executed',
                severity='HIGH',
                confirmed=True
            ),
            DOMScanResult(
                url='https://example.com',
                vulnerability_type='REFLECTED_XSS',
                payload='<img src=x>',
                injection_point='search',
                evidence='Payload reflected',
                severity='MEDIUM',
                confirmed=False
            )
        ]
        
        findings = scanner.convert_to_findings(scan_results)
        
        assert len(findings) == 2
        assert findings[0].finding_type == FindingType.XSS_DOM
        assert findings[0].severity == FindingSeverity.HIGH
        assert findings[1].finding_type == FindingType.XSS_REFLECTED
        assert findings[1].severity == FindingSeverity.MEDIUM


@pytest.mark.skipif(PLAYWRIGHT_AVAILABLE, reason="Test only when Playwright not available")
class TestDOMScannerWithoutPlaywright:
    """Test DOM Scanner behavior when Playwright not installed."""
    
    def test_scanner_without_playwright(self):
        """Test that scanner can be imported even without Playwright."""
        scanner = PlaywrightDOMScanner()
        assert scanner is not None