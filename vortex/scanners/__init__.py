"""
VORTEX Vulnerability Scanners - V19.0 ULTIMATE
Heuristic detection with authority-compliant evidence

SCANNER TYPES:
- SQL Injection (SQLi)
- Cross-Site Scripting (XSS)
- Local File Inclusion (LFI)
- Server-Side Request Forgery (SSRF)
- Cross-Site Request Forgery (CSRF)
- Server-Side Template Injection (SSTI)
- XML External Entity (XXE)
- File Upload Vulnerabilities
- JWT Security Issues

IMPORTANT: All scanners produce HEURISTIC_ONLY evidence.
PoC replay requires AI-generated PoCs.
"""

from scanners.base import BaseScanner
from scanners.vulns.sqli import SQLInjectionScanner
from scanners.vulns.xss import XSSScanner
from scanners.vulns.lfi import LFIScanner
from scanners.vulns.ssrf import SSRFScanner
from scanners.vulns.csrf import CSRFScanner
from scanners.vulns.ssti import SSTIScanner
from scanners.vulns.xxe import XXEScanner
from scanners.vulns.file_upload import FileUploadScanner
from scanners.api.jwt_scanner import JWTScanner

__all__ = [
    'BaseScanner',
    'SQLInjectionScanner',
    'XSSScanner',
    'LFIScanner',
    'SSRFScanner',
    'CSRFScanner',
    'SSTIScanner',
    'XXEScanner',
    'FileUploadScanner',
    'JWTScanner',
]