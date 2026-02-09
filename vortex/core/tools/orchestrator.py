"""
VORTEX Tool Orchestrator - V18.0 ULTIMATE
External security tool integration layer

Integrates industry-standard tools:
- SQLMap for advanced SQL injection
- Nmap for network reconnaissance
- Nuclei for template-based scanning

ARCHITECTURE:
- Asynchronous subprocess execution
- Output parsing and normalization
- Rate limiting and resource management
- Tool availability detection
"""

import asyncio
import json
import shutil
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from domain.models import AssessmentResult
from domain.enums import FindingType, FindingSeverity, VerificationStatus

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from external tool execution."""
    tool_name: str
    success: bool
    execution_time: float
    raw_output: str
    parsed_findings: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    exit_code: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BaseToolWrapper(ABC):
    """
    Abstract base class for external tool wrappers.
    
    All tool integrations must implement:
    - is_available(): Check if tool is installed
    - execute(): Run the tool with given parameters
    - parse_output(): Convert tool output to findings
    """
    
    def __init__(self):
        self.tool_name: str = "base"
        self.tool_path: Optional[str] = None
        self.timeout: int = 300  # 5 minutes default
        self.max_concurrent: int = 3
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if external tool is installed and accessible."""
        pass
    
    @abstractmethod
    async def execute(self, target: str, options: Dict[str, Any]) -> ToolResult:
        """Execute the tool against target with given options."""
        pass
    
    @abstractmethod
    def parse_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse tool output into structured findings."""
        pass
    
    async def _run_command(self, cmd: List[str], timeout: Optional[int] = None) -> tuple:
        """
        Run command asynchronously with timeout.
        
        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        timeout = timeout or self.timeout
        
        try:
            async with self._semaphore:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )
                    return (
                        stdout.decode('utf-8', errors='replace'),
                        stderr.decode('utf-8', errors='replace'),
                        process.returncode
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return ("", f"Command timed out after {timeout}s", -1)
                    
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return ("", str(e), -1)


class SQLMapWrapper(BaseToolWrapper):
    """
    SQLMap integration for advanced SQL injection testing.
    
    Supports:
    - Automatic database detection
    - Multiple injection techniques
    - WAF bypass
    - Data extraction (with limits)
    """
    
    def __init__(self):
        super().__init__()
        self.tool_name = "sqlmap"
        self.tool_path = shutil.which("sqlmap")
        self.timeout = 600  # 10 minutes for SQLMap
    
    def is_available(self) -> bool:
        """Check if sqlmap is installed."""
        if self.tool_path:
            return True
        # Try python sqlmap module
        try:
            result = subprocess.run(
                ["python", "-m", "sqlmap", "--version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def execute(self, target: str, options: Dict[str, Any]) -> ToolResult:
        """
        Execute SQLMap against target.
        
        Args:
            target: Target URL with parameter
            options: SQLMap options dict
                - level: Injection level (1-5)
                - risk: Risk level (1-3)
                - technique: Injection techniques (BEUSTQ)
                - tamper: Tamper scripts for WAF bypass
        """
        start_time = datetime.utcnow()
        
        if not self.is_available():
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                execution_time=0,
                raw_output="",
                error="SQLMap not installed"
            )
        
        # Build command
        cmd = self._build_command(target, options)
        
        logger.info(f"Executing SQLMap: {' '.join(cmd[:5])}...")
        
        stdout, stderr, exit_code = await self._run_command(cmd)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Parse findings
        parsed = self.parse_output(stdout)
        
        return ToolResult(
            tool_name=self.tool_name,
            success=exit_code == 0 and len(parsed) > 0,
            execution_time=execution_time,
            raw_output=stdout,
            parsed_findings=parsed,
            error=stderr if exit_code != 0 else None,
            exit_code=exit_code
        )
    
    def _build_command(self, target: str, options: Dict[str, Any]) -> List[str]:
        """Build SQLMap command with options."""
        cmd = ["sqlmap", "-u", target, "--batch", "--output-dir=/tmp/sqlmap"]
        
        # Add common options
        level = options.get("level", 1)
        risk = options.get("risk", 1)
        cmd.extend(["--level", str(level), "--risk", str(risk)])
        
        # Technique selection
        technique = options.get("technique", "BEUSTQ")
        cmd.extend(["--technique", technique])
        
        # WAF bypass tamper scripts
        tamper = options.get("tamper")
        if tamper:
            cmd.extend(["--tamper", tamper])
        
        # Output format
        cmd.append("--forms")
        
        # Safety limits
        cmd.extend([
            "--threads", "2",
            "--time-sec", "5",
            "--retries", "2"
        ])
        
        return cmd
    
    def parse_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse SQLMap output for vulnerabilities."""
        findings = []
        
        # Look for vulnerability indicators
        vuln_indicators = [
            "is vulnerable",
            "sqlmap identified the following injection",
            "Parameter:",
            "Type:",
            "Title:"
        ]
        
        lines = raw_output.split('\n')
        current_finding = {}
        
        for line in lines:
            line = line.strip()
            
            if "is vulnerable" in line.lower():
                if current_finding:
                    findings.append(current_finding)
                current_finding = {
                    "type": "sqli",
                    "confirmed": True,
                    "evidence": line,
                    "details": {}
                }
            
            elif "Parameter:" in line and current_finding:
                current_finding["parameter"] = line.split("Parameter:")[-1].strip()
            
            elif "Type:" in line and current_finding:
                current_finding["details"]["injection_type"] = line.split("Type:")[-1].strip()
            
            elif "Title:" in line and current_finding:
                current_finding["details"]["title"] = line.split("Title:")[-1].strip()
            
            elif "Payload:" in line and current_finding:
                current_finding["payload"] = line.split("Payload:")[-1].strip()
        
        if current_finding:
            findings.append(current_finding)
        
        return findings


class NmapWrapper(BaseToolWrapper):
    """
    Nmap integration for network reconnaissance.
    
    Supports:
    - Port scanning
    - Service detection
    - OS fingerprinting
    - Script scanning (NSE)
    """
    
    def __init__(self):
        super().__init__()
        self.tool_name = "nmap"
        self.tool_path = shutil.which("nmap")
        self.timeout = 300
    
    def is_available(self) -> bool:
        """Check if nmap is installed."""
        return self.tool_path is not None
    
    async def execute(self, target: str, options: Dict[str, Any]) -> ToolResult:
        """
        Execute Nmap scan against target.
        
        Args:
            target: Target IP/hostname
            options: Nmap options
                - ports: Port range (e.g., "1-1000")
                - scan_type: Scan type (sS, sT, sU, etc.)
                - scripts: NSE scripts to run
        """
        start_time = datetime.utcnow()
        
        if not self.is_available():
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                execution_time=0,
                raw_output="",
                error="Nmap not installed"
            )
        
        # Build command
        cmd = self._build_command(target, options)
        
        logger.info(f"Executing Nmap: {' '.join(cmd[:5])}...")
        
        stdout, stderr, exit_code = await self._run_command(cmd)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Parse findings
        parsed = self.parse_output(stdout)
        
        return ToolResult(
            tool_name=self.tool_name,
            success=exit_code == 0,
            execution_time=execution_time,
            raw_output=stdout,
            parsed_findings=parsed,
            error=stderr if exit_code != 0 else None,
            exit_code=exit_code
        )
    
    def _build_command(self, target: str, options: Dict[str, Any]) -> List[str]:
        """Build Nmap command with options."""
        cmd = ["nmap"]
        
        # Output format
        cmd.extend(["-oX", "-"])  # XML to stdout
        
        # Scan type
        scan_type = options.get("scan_type", "-sT")  # TCP connect (no root)
        cmd.append(scan_type)
        
        # Port range
        ports = options.get("ports", "1-1000")
        cmd.extend(["-p", ports])
        
        # Service detection
        if options.get("service_detection", True):
            cmd.append("-sV")
        
        # Scripts
        scripts = options.get("scripts")
        if scripts:
            cmd.extend(["--script", scripts])
        
        # Speed/timing
        timing = options.get("timing", "T3")
        cmd.append(f"-{timing}")
        
        # Target
        cmd.append(target)
        
        return cmd
    
    def parse_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse Nmap XML output."""
        findings = []
        
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(raw_output)
            
            for host in root.findall('.//host'):
                host_addr = host.find('.//address[@addrtype="ipv4"]')
                if host_addr is not None:
                    ip = host_addr.get('addr')
                else:
                    continue
                
                for port in host.findall('.//port'):
                    state = port.find('state')
                    if state is not None and state.get('state') == 'open':
                        service = port.find('service')
                        findings.append({
                            "type": "open_port",
                            "ip": ip,
                            "port": port.get('portid'),
                            "protocol": port.get('protocol'),
                            "service": service.get('name') if service is not None else 'unknown',
                            "version": service.get('version') if service is not None else None,
                            "product": service.get('product') if service is not None else None
                        })
        
        except Exception as e:
            logger.error(f"Failed to parse Nmap output: {e}")
            # Fallback to text parsing
            for line in raw_output.split('\n'):
                if '/tcp' in line or '/udp' in line:
                    parts = line.split()
                    if len(parts) >= 3 and 'open' in parts[1]:
                        findings.append({
                            "type": "open_port",
                            "port": parts[0].split('/')[0],
                            "protocol": parts[0].split('/')[1],
                            "state": parts[1],
                            "service": parts[2] if len(parts) > 2 else 'unknown'
                        })
        
        return findings


class NucleiWrapper(BaseToolWrapper):
    """
    Nuclei integration for template-based vulnerability scanning.
    
    Supports:
    - CVE detection
    - Misconfigurations
    - Exposed panels
    - Custom templates
    """
    
    def __init__(self):
        super().__init__()
        self.tool_name = "nuclei"
        self.tool_path = shutil.which("nuclei")
        self.timeout = 600  # 10 minutes
    
    def is_available(self) -> bool:
        """Check if nuclei is installed."""
        return self.tool_path is not None
    
    async def execute(self, target: str, options: Dict[str, Any]) -> ToolResult:
        """
        Execute Nuclei scan against target.
        
        Args:
            target: Target URL
            options: Nuclei options
                - templates: Template paths/tags
                - severity: Severity filter
                - rate_limit: Requests per second
        """
        start_time = datetime.utcnow()
        
        if not self.is_available():
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                execution_time=0,
                raw_output="",
                error="Nuclei not installed"
            )
        
        # Create temp file for JSON output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            cmd = self._build_command(target, options, output_file)
            
            logger.info(f"Executing Nuclei: {' '.join(cmd[:5])}...")
            
            stdout, stderr, exit_code = await self._run_command(cmd)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Read JSON output
            try:
                with open(output_file, 'r') as f:
                    raw_json = f.read()
                parsed = self.parse_output(raw_json)
            except Exception:
                parsed = []
            
            return ToolResult(
                tool_name=self.tool_name,
                success=exit_code == 0,
                execution_time=execution_time,
                raw_output=stdout + raw_json if 'raw_json' in dir() else stdout,
                parsed_findings=parsed,
                error=stderr if exit_code != 0 else None,
                exit_code=exit_code
            )
        finally:
            # Cleanup temp file
            Path(output_file).unlink(missing_ok=True)
    
    def _build_command(self, target: str, options: Dict[str, Any], output_file: str) -> List[str]:
        """Build Nuclei command with options."""
        cmd = ["nuclei", "-u", target, "-json", "-o", output_file]
        
        # Severity filter
        severity = options.get("severity", "critical,high,medium")
        cmd.extend(["-severity", severity])
        
        # Templates
        templates = options.get("templates")
        if templates:
            cmd.extend(["-t", templates])
        
        # Tags
        tags = options.get("tags")
        if tags:
            cmd.extend(["-tags", tags])
        
        # Rate limiting
        rate_limit = options.get("rate_limit", 100)
        cmd.extend(["-rate-limit", str(rate_limit)])
        
        # Concurrency
        cmd.extend(["-c", "10"])
        
        # Silent mode (less noise)
        cmd.append("-silent")
        
        return cmd
    
    def parse_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse Nuclei JSON output."""
        findings = []
        
        for line in raw_output.strip().split('\n'):
            if not line:
                continue
            
            try:
                data = json.loads(line)
                findings.append({
                    "type": "nuclei",
                    "template_id": data.get("template-id"),
                    "name": data.get("info", {}).get("name"),
                    "severity": data.get("info", {}).get("severity"),
                    "description": data.get("info", {}).get("description"),
                    "url": data.get("matched-at"),
                    "matcher_name": data.get("matcher-name"),
                    "extracted_results": data.get("extracted-results"),
                    "curl_command": data.get("curl-command"),
                    "tags": data.get("info", {}).get("tags")
                })
            except json.JSONDecodeError:
                continue
        
        return findings


class ToolOrchestrator:
    """
    Main orchestrator for external security tools.
    
    Manages tool execution and result aggregation.
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseToolWrapper] = {
            'sqlmap': SQLMapWrapper(),
            'nmap': NmapWrapper(),
            'nuclei': NucleiWrapper()
        }
        self.execution_history: List[ToolResult] = []
    
    def get_available_tools(self) -> Dict[str, bool]:
        """Get availability status of all tools."""
        return {
            name: tool.is_available() 
            for name, tool in self.tools.items()
        }
    
    async def run_tool(self, tool_name: str, target: str, 
                       options: Optional[Dict[str, Any]] = None) -> ToolResult:
        """
        Run a specific tool against target.
        
        Args:
            tool_name: Name of tool (sqlmap, nmap, nuclei)
            target: Target URL/IP
            options: Tool-specific options
        """
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                execution_time=0,
                raw_output="",
                error=f"Unknown tool: {tool_name}"
            )
        
        tool = self.tools[tool_name]
        options = options or {}
        
        result = await tool.execute(target, options)
        self.execution_history.append(result)
        
        return result
    
    async def run_comprehensive_scan(self, target: str, 
                                     tool_list: Optional[List[str]] = None) -> Dict[str, ToolResult]:
        """
        Run comprehensive scan using multiple tools.
        
        Args:
            target: Target URL/IP
            tool_list: List of tools to use (default: all available)
        """
        results = {}
        
        if tool_list is None:
            tool_list = [name for name, avail in self.get_available_tools().items() if avail]
        
        for tool_name in tool_list:
            results[tool_name] = await self.run_tool(tool_name, target)
        
        return results
    
    def convert_to_findings(self, results: Dict[str, ToolResult]) -> List[AssessmentResult]:
        """
        Convert tool results to standard AssessmentResult objects.
        """
        findings = []
        
        for tool_name, result in results.items():
            if not result.success:
                continue
            
            for parsed in result.parsed_findings:
                finding = self._create_finding_from_parsed(tool_name, parsed)
                if finding:
                    findings.append(finding)
        
        return findings
    
    def _create_finding_from_parsed(self, tool_name: str, 
                                    parsed: Dict[str, Any]) -> Optional[AssessmentResult]:
        """Create AssessmentResult from parsed tool output."""
        import uuid
        
        try:
            # Map tool findings to severity
            severity_map = {
                "critical": FindingSeverity.CRITICAL,
                "high": FindingSeverity.HIGH,
                "medium": FindingSeverity.MEDIUM,
                "low": FindingSeverity.LOW,
                "info": FindingSeverity.INFO
            }
            
            # Map tool findings to type
            type_map = {
                "sqli": FindingType.SQLI,
                "xss": FindingType.XSS_REFLECTED,
                "nuclei": FindingType.OTHER,
                "open_port": FindingType.INFO_DISCLOSURE
            }
            
            parsed_type = parsed.get("type", "other")
            finding_type = type_map.get(parsed_type, FindingType.OTHER)
            
            parsed_severity = str(parsed.get("severity", "medium")).lower()
            severity = severity_map.get(parsed_severity, FindingSeverity.MEDIUM)
            
            finding = AssessmentResult(
                id=uuid.uuid4(),
                url=parsed.get("url", ""),
                finding_type=finding_type,
                severity=severity,
                status=VerificationStatus.SYSTEM_VERIFIED,
                heuristic_score=0.9,  # High confidence from external tools
                evidence=json.dumps(parsed),
                payload=parsed.get("payload"),
                vulnerable_parameter=parsed.get("parameter")
            )
            
            return finding
            
        except Exception as e:
            logger.error(f"Failed to create finding from {tool_name}: {e}")
            return None


# Global orchestrator instance
global_tool_orchestrator = ToolOrchestrator()


def get_tool_orchestrator() -> ToolOrchestrator:
    """Get global tool orchestrator instance."""
    return global_tool_orchestrator
