"""
VORTEX Reconnaissance Module - V19.0
Deep discovery and asset inventory

FEATURES:
- Subdomain enumeration (crt.sh, naive DNS)
- Technology detection (Wappalyzer logic)
- Port scanning (via Nmap wrapper)
- Asset tree construction

ARCHITECTURE:
- ReconManager: Orchestrates discovery
- SubdomainScanner: Passive/Active enumeration
- TechDetector: Stack fingerprinting
"""

import asyncio
import logging
import socket
import re
import json
import aiohttp
from typing import List, Dict, Set, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import wrappers if needed
from core.tools.orchestrator import get_tool_orchestrator
from core.events import emit, EventType, log_scan_progress

logger = logging.getLogger(__name__)


@dataclass
class Asset:
    """Represents a discovered asset (subdomain)."""
    domain: str
    ip: Optional[str] = None
    technologies: List[str] = field(default_factory=list)
    ports: List[int] = field(default_factory=list)
    status_code: Optional[int] = None
    title: Optional[str] = None
    is_live: bool = False
    discovery_source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SubdomainScanner:
    """Subdomain enumeration engine."""
    
    def __init__(self):
        self.found_subdomains: Set[str] = set()
    
    async def scan(self, domain: str) -> List[str]:
        """Run all subdomain discovery methods."""
        results = set()
        
        # 1. crt.sh (Passive, very fast)
        crt_results = await self._query_crtsh(domain)
        results.update(crt_results)
        
        # 2. Tool Orchestrator fallback (e.g. Subfinder if installed)
        # TODO: Add tool integration if highly requested
        
        self.found_subdomains = results
        return list(results)
    
    async def _query_crtsh(self, domain: str) -> Set[str]:
        """Query crt.sh certificate transparency logs."""
        url = f"https://crt.sh/?q=%.{domain}&output=json"
        domains = set()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for entry in data:
                            name_value = entry.get('name_value', '')
                            # Split multiline names
                            for sub in name_value.split('\n'):
                                sub = sub.strip().lower()
                                # Clean asterisks
                                if sub.startswith('*.'):
                                    sub = sub[2:]
                                if sub.endswith(f".{domain}") or sub == domain:
                                    domains.add(sub)
        except Exception as e:
            logger.warning(f"crt.sh query failed: {e}")
        
        return domains


class TechDetector:
    """Identify technologies running on a URL."""
    
    # Simple signatures (expanded in prod)
    SIGNATURES = {
        'php': [r'PHPSESSID', r'\.php', r'X-Powered-By: PHP'],
        'node': [r'Express', r'connect.sid', r'socket.io'],
        'python': [r'csrftoken', r'sessionid', r'Werkzeug', r'Gunicorn'],
        'java': [r'JSESSIONID', r'Tomcat', r'Jetty', r'\.jsp'],
        'asp': [r'ASPSESSIONID', r'ASP.NET', r'\.aspx'],
        'nginx': [r'Server: nginx'],
        'apache': [r'Server: Apache'],
        'cloudflare': [r'Server: cloudflare', r'__cfduid']
    }
    
    async def detect(self, url: str) -> List[str]:
        """Detect technologies for a URL."""
        techs = set()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5, ssl=False) as resp:
                    headers = str(resp.headers)
                    text = await resp.text()
                    cookies = str(resp.cookies)
                    
                    combined = f"{headers} {text[:2000]} {cookies}"
                    
                    for tech, patterns in self.SIGNATURES.items():
                        for pattern in patterns:
                            if re.search(pattern, combined, re.IGNORECASE):
                                techs.add(tech)
                                break
        except Exception:
            pass
            
        return list(techs)


class ReconManager:
    """Orchestrator for Reconnaissance."""
    
    def __init__(self):
        self.subdomain_scanner = SubdomainScanner()
        self.tech_detector = TechDetector()
        self.assets: Dict[str, Asset] = {}
    
    async def fast_recon(self, domain: str) -> List[Asset]:
        """
        Perform fast reconnaissance on a domain.
        1. Find subdomains
        2. Probe for liveness
        3. Detect technologies
        """
        domain = domain.replace('http://', '').replace('https://', '').split('/')[0]
        logger.info(f"Starting recon for: {domain}")
        emit(EventType.SCAN_START, f"🔭 Recon başlatıldı: {domain}")
        
        # 1. Subdomains
        subdomains = await self.subdomain_scanner.scan(domain)
        logger.info(f"Found {len(subdomains)} subdomains")
        emit(EventType.SCAN_PROGRESS, f"Bulunan subdomain: {len(subdomains)}", {'count': len(subdomains)})
        
        # 2. Probe & Detect (Parallel)
        tasks = []
        for sub in subdomains:
            tasks.append(self._probe_asset(sub))
        
        # Limit concurrency
        results = []
        chunk_size = 10
        total = len(tasks)
        
        for i in range(0, total, chunk_size):
            chunk = tasks[i:i+chunk_size]
            results.extend(await asyncio.gather(*chunk))
            
            progress = int((i + len(chunk)) / total * 100)
            log_scan_progress(domain, progress, "Varlıklar analiz ediliyor...")
        
        # Filter live assets
        live_assets = [a for a in results if a.is_live]
        
        # Store results
        self.assets = {a.domain: a for a in live_assets}
        
        emit(EventType.SCAN_COMPLETE, f"Recon tamamlandı. {len(live_assets)} aktif varlık bulundu.", 
             {'assets': len(live_assets)})
        
        return live_assets
    
    async def _probe_asset(self, domain: str) -> Asset:
        """Probe a single asset."""
        asset = Asset(domain=domain)
        url = f"https://{domain}"
        
        try:
            # Resolve IP
            asset.ip = socket.gethostbyname(domain)
            
            # HTTP Probe
            asset.technologies = await self.tech_detector.detect(url)
            asset.is_live = True
            
            # Simple status check
            async with aiohttp.ClientSession() as session:
                async with session.head(url, timeout=3, ssl=False) as resp:
                    asset.status_code = resp.status
        
        except Exception:
            # Try HTTP if HTTPS fails
            try:
                url = f"http://{domain}"
                asset.technologies = await self.tech_detector.detect(url)
                asset.is_live = True
                async with aiohttp.ClientSession() as session:
                    async with session.head(url, timeout=3) as resp:
                        asset.status_code = resp.status
            except Exception:
                pass
        
        return asset


# Global instance
global_recon_manager = ReconManager()

def get_recon_manager() -> ReconManager:
    return global_recon_manager
