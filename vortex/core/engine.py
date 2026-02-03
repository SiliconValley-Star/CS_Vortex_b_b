"""
VORTEX Scan Engine - V17.0 ULTIMATE
Main orchestration engine integrating all systems

RESPONSIBILITIES:
- Scan workflow coordination
- Component integration
- Error recovery
- Progress tracking
"""

import asyncio
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from domain.models import AssessmentResult
from domain.enums import VerificationStatus, FindingSeverity
from core.exceptions import VortexException
from core.database import global_database_manager
from core.network import global_network_client
from core.workflow.orchestrator import WorkflowOrchestrator
from core.health.monitor import global_health_monitor
from core.queue_manager import global_queue_manager
from core.ai.fallbacks import global_fallback_manager

# V21.0 - Performance profiling and metrics (NON-BREAKING)
try:
    from utils.profiling import profile_async, global_profiler
    from core.metrics import global_metrics
    PROFILING_ENABLED = True
except ImportError:
    PROFILING_ENABLED = False
    logger = logging.getLogger(__name__)
    logger.debug("Performance profiling not available")

# Import vulnerability scanners
from scanners.vulns.sqli import SQLInjectionScanner
from scanners.vulns.xss import XSSScanner
from scanners.vulns.lfi import LFIScanner
from scanners.vulns.ssrf import SSRFScanner

# V19.0 - New module integrations
try:
    from core.events import get_event_emitter, EventType, log_scan_start, log_scan_progress, log_finding
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False

# V20.0 - Attack Chain Orchestrator Integration
try:
    from core.intelligence.attack_chains import get_chain_orchestrator, detect_and_execute_chains
    ATTACK_CHAINS_AVAILABLE = True
except ImportError:
    ATTACK_CHAINS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Attack Chain Orchestrator not available")

try:
    from core.recon import get_recon_manager
    RECON_AVAILABLE = True
except ImportError:
    RECON_AVAILABLE = False
    get_recon_manager = None

try:
    from core.stealth import get_stealth_client
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

logger = logging.getLogger(__name__)


class VortexScanEngine:
    """
    Main VORTEX scan engine.
    
    Coordinates all system components for complete vulnerability scanning workflow.
    """
    
    def __init__(self):
        self.database = global_database_manager
        self.network_client = global_network_client
        self.workflow_orchestrator = WorkflowOrchestrator()
        self.health_monitor = global_health_monitor
        self.queue_manager = global_queue_manager

        # Queue processing task
        self.queue_processor_task = None
        self._running = False
        
        # Initialize vulnerability scanners (lazy loaded)
        self.scanners = {}
        self._load_scanners()
        
        self.scan_active = False
        self.stop_requested = False
        self.active_scans: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            'scans_completed': 0,
            'findings_detected': 0,
            'findings_verified': 0,
            'findings_submitted': 0
        }
        
        # V19.0 - Event emitter for real-time updates
        self.event_emitter = get_event_emitter() if EVENTS_AVAILABLE else None
        
        # Recon manager for subdomain discovery
        self.recon_manager = get_recon_manager() if RECON_AVAILABLE else None
        
        # V20.0 - Attack Chain Orchestrator for multi-step attacks
        self.chain_orchestrator = get_chain_orchestrator() if ATTACK_CHAINS_AVAILABLE else None
        
        # V21.0 - Performance metrics (NON-BREAKING)
        self.metrics = global_metrics if PROFILING_ENABLED else None
    
    def _load_scanners(self):
        """Lazy load vulnerability scanners to avoid circular imports."""
        try:
            from scanners.vulns.sqli import SQLInjectionScanner
            from scanners.vulns.xss import XSSScanner
            from scanners.vulns.lfi import LFIScanner
            from scanners.vulns.ssrf import SSRFScanner
            from scanners.vulns.csrf import CSRFScanner
            from scanners.vulns.ssti import SSTIScanner
            from scanners.vulns.xxe import XXEScanner
            from scanners.vulns.file_upload import FileUploadScanner
            from scanners.api.jwt_scanner import JWTScanner
            
            self.scanners = {
                'sqli': SQLInjectionScanner(),
                'xss': XSSScanner(),
                'lfi': LFIScanner(),
                'ssrf': SSRFScanner(),
                'csrf': CSRFScanner(),
                'ssti': SSTIScanner(),
                'xxe': XXEScanner(),
                'file_upload': FileUploadScanner(),
                'jwt': JWTScanner()
            }
            
            # V20.0 - Load advanced scanners (DOM XSS, GraphQL)
            try:
                from scanners.advanced.dom_scanner import PlaywrightDOMScanner
                self.scanners['dom_xss'] = PlaywrightDOMScanner()
                logger.info("DOM XSS Scanner loaded (Playwright)")
            except ImportError as e:
                logger.warning(f"DOM XSS Scanner not available: {e}")
            
            try:
                from scanners.api.graphql_scanner import GraphQLScanner
                self.scanners['graphql'] = GraphQLScanner()
                logger.info("GraphQL Security Scanner loaded")
            except ImportError as e:
                logger.warning(f"GraphQL Scanner not available: {e}")
            
            logger.info(f"Loaded {len(self.scanners)} vulnerability scanners")
        except ImportError as e:
            logger.error(f"Failed to load scanners: {e}")
            # Fallback to basic scanners only
            from scanners.vulns.sqli import SQLInjectionScanner
            from scanners.vulns.xss import XSSScanner
            from scanners.vulns.lfi import LFIScanner
            from scanners.vulns.ssrf import SSRFScanner
            
            self.scanners = {
                'sqli': SQLInjectionScanner(),
                'xss': XSSScanner(),
                'lfi': LFIScanner(),
                'ssrf': SSRFScanner()
            }
            logger.warning(f"Loaded {len(self.scanners)} basic scanners only")
    
    async def initialize(self):
        """Initialize all engine components."""
        logger.info("Initializing VORTEX Scan Engine...")
        
        try:
            # Initialize database
            await self.database.initialize()
            
            # Initialize network client
            await self.network_client.initialize()
            
            # Workflow orchestrator doesn't need initialization
            # It's ready to use after instantiation
            
            # Start health monitoring
            await self.health_monitor.start_monitoring(interval_seconds=300)

            # Initialize and start queue manager
            await self.queue_manager.initialize()
            
            # Start queue processor
            self._running = True
            self.queue_processor_task = asyncio.create_task(self._process_scan_queue())
            
            logger.info("VORTEX Scan Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}", exc_info=True)
            raise
    
    async def shutdown(self):
        """Shutdown all engine components."""
        logger.info("Shutting down VORTEX Scan Engine...")
        
        try:
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()
            
            # Stop queue processor
            self._running = False
            if self.queue_processor_task:
                self.queue_processor_task.cancel()
                try:
                    await self.queue_processor_task
                except asyncio.CancelledError:
                    pass

            await self.queue_manager.shutdown()
            
            # Close network client
            await self.network_client.close()
            
            # Close database
            await self.database.close()
            
            logger.info("VORTEX Scan Engine shut down successfully")
            
        except Exception as e:
            logger.error(f"Engine shutdown error: {e}", exc_info=True)
    
    async def _process_scan_queue(self):
        """Process scan tasks from the queue."""
        logger.info("Starting scan queue processor")
        
        while self._running:
            try:
                # Get next scan task (wait for item)
                task_item = await self.queue_manager.dequeue("scan_tasks")
                
                if task_item:
                    try:
                        scan_id = task_item.data['scan_id']
                        targets = task_item.data['targets']
                        
                        logger.info(f"Processing scan task from queue: {scan_id}")
                        
                        # Execute the scan (using internal logic, bypassing queue check to avoid loop)
                        # We extract the logic from start_scan into _execute_scan_internal
                        await self._execute_scan_internal(scan_id, targets, task_item.data.get('config', {}))
                        
                    except Exception as e:
                        logger.error(f"Error processing queued task: {e}")
                
                await asyncio.sleep(0.1) # Small delay
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1)

    async def _execute_scan_internal(self, scan_id: str, targets: List[str], config: Dict[str, Any]):
        """Internal execution of a scan, bypassing the queue."""
        scan_types = config.get('include_vulns', ['sqli', 'xss', 'lfi', 'ssrf'])
        
        # V19.0 - Emit scan start event
        if self.event_emitter:
            self.event_emitter.scan_start(str(targets), 'full')
        
        all_findings = []
        scan_start = datetime.utcnow()
        
        try:
            total_targets = len(targets)
            shutdown_event = None # No event passed in queued mode typically, or managed via scan_active
            
            for idx, target_url in enumerate(targets):
                if self.stop_requested: # Check flag
                    logger.info("Scan stop requested")
                    break
                
                # Scan this target
                result = await self.scan_target(target_url, scan_types)
                
                if result and result.get('findings'):
                    all_findings.extend(result['findings'])
                
                # V19.0 - Emit progress event
                progress = int((idx + 1) / total_targets * 100)
                if self.event_emitter:
                    self.event_emitter.scan_progress(target_url, progress, f"Target {idx + 1}/{total_targets}")
            
        except Exception as e:
            logger.error(f"Internal scan execution failed: {e}")

    async def start_scan(self,
                        config: Dict[str, Any],
                        progress_callback: Optional[callable] = None,
                        shutdown_event: Optional[asyncio.Event] = None) -> Dict[str, Any]:
        """
        Queue a complete scan based on configuration.
        
        Args:
            config: Scan configuration dictionary
            progress_callback: Optional callback for progress updates
            shutdown_event: Optional event to signal shutdown
            
        Returns:
            Scan ID and status (Queued)
        """
        scan_id = f"scan_{datetime.utcnow().timestamp()}"
        targets = config.get('targets', [])
        
        logger.info(f"Queuing scan {scan_id} with {len(targets)} targets")
        
        # Add to priority queue
        await self.queue_manager.enqueue_finding(
            queue_name="scan_tasks",
            item={
                'type': 'scan_task',
                'scan_id': scan_id,
                'targets': targets,
                'config': config
            },
            priority=1
        )
        
        return {
            'scan_id': scan_id,
            'status': 'QUEUED',
            'message': 'Scan added to processing queue',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def scan_target(self,
                         target_url: str,
                         scan_types: Optional[List[str]] = None,
                         enable_recon: bool = False,
                         enable_chains: bool = False) -> Dict[str, Any]:
        """
        Scan target URL for vulnerabilities.
        
        Args:
            target_url: Target URL to scan
            scan_types: List of vulnerability types to scan for
            enable_recon: If True, perform subdomain reconnaissance first
            enable_chains: If True, detect and execute multi-step attack chains (V20.0)
            
        Returns:
            Scan results summary
        """
        scan_id = f"scan_{datetime.utcnow().timestamp()}"
        logger.info(f"Starting scan {scan_id} for {target_url} (recon: {enable_recon})")
        
        self.scan_active = True
        scan_start = datetime.utcnow()
        
        findings = []
        targets_to_scan = [target_url]
        
        try:
            # Recon phase: Discover subdomains if enabled
            if enable_recon and self.recon_manager:
                logger.info(f"Starting subdomain reconnaissance for {target_url}")
                
                # Extract domain from URL
                domain = target_url.replace('http://', '').replace('https://', '').split('/')[0]
                
                # Discover assets
                assets = await self.recon_manager.fast_recon(domain)
                logger.info(f"Discovered {len(assets)} live subdomains")
                
                # Add discovered subdomains to scan targets
                if assets:
                    targets_to_scan = [f"https://{asset.domain}" for asset in assets if asset.is_live]
                    logger.info(f"Will scan {len(targets_to_scan)} targets (including subdomains)")
            
            # Default scan types if not specified
            if not scan_types:
                scan_types = ['sqli', 'xss', 'lfi', 'ssrf']
            
            # Scan each target
            for target in targets_to_scan:
                logger.info(f"Scanning target: {target}")
                
                # Scan for each vulnerability type
                for scan_type in scan_types:
                    try:
                        scan_findings = await self._scan_vulnerability_type(
                            target,
                            scan_type
                        )
                        
                        if scan_findings:
                            findings.extend(scan_findings)
                            self.stats['findings_detected'] += len(scan_findings)
                            
                    except Exception as e:
                        logger.error(f"Scan type {scan_type} failed for {target}: {e}")
            
            # V20.0 - Attack Chain Detection and Execution
            executed_chains = []
            if enable_chains and self.chain_orchestrator and findings:
                logger.info(f"Analyzing {len(findings)} findings for attack chains...")
                
                try:
                    # Detect potential attack chains
                    chains = self.chain_orchestrator.detect_chains(findings)
                    
                    if chains:
                        logger.info(f"Detected {len(chains)} potential attack chains")
                        
                        # Execute chains
                        for chain in chains:
                            logger.info(f"Executing attack chain: {chain.name}")
                            success = await self.chain_orchestrator.execute_chain(chain)
                            
                            if success:
                                executed_chains.append(chain)
                                logger.info(
                                    f"Attack chain completed: {chain.name}",
                                    severity=chain.severity.value
                                )
                    else:
                        logger.info("No attack chains detected from current findings")
                        
                except Exception as e:
                    logger.error(f"Attack chain analysis failed: {e}")
            
            # Calculate scan duration
            scan_duration = (datetime.utcnow() - scan_start).total_seconds()
            
            self.stats['scans_completed'] += 1
            
            # V21.0 - Record scan metrics (NON-BREAKING)
            if self.metrics:
                self.metrics.record_scan(
                    scanner_type='full_scan',
                    duration=scan_duration,
                    findings_count=len(findings)
                )
            
            # Return summary with chain information
            result = {
                'scan_id': scan_id,
                'target_url': target_url,
                'targets_scanned': len(targets_to_scan),
                'subdomains_discovered': len(targets_to_scan) - 1 if enable_recon else 0,
                'scan_types': scan_types,
                'findings_count': len(findings),
                'findings': [self._finding_to_summary(f) for f in findings],
                'duration_seconds': scan_duration,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add chain results if enabled
            if enable_chains and executed_chains:
                result['attack_chains'] = {
                    'detected': len(chains) if 'chains' in locals() else 0,
                    'executed': len(executed_chains),
                    'chains': [
                        self.chain_orchestrator.get_chain_report(chain)
                        for chain in executed_chains
                    ]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Scan {scan_id} failed: {e}", exc_info=True)
            raise
        
        finally:
            self.scan_active = False
    
    async def _scan_vulnerability_type(self,
                                       target_url: str,
                                       scan_type: str) -> Optional[List[AssessmentResult]]:
        """
        Scan for specific vulnerability type using appropriate scanner.
        NOW WITH FULL WORKFLOW INTEGRATION:
        - Heuristic detection (scanner)
        - AI advisory analysis
        - System verification
        - Evidence validation
        - Authority-based determination
        
        Args:
            target_url: Target URL to scan
            scan_type: Type of vulnerability to scan for (sqli, xss, lfi, ssrf)
            
        Returns:
            List of workflow-processed findings
        """
        logger.debug(f"Scanning {target_url} for {scan_type}")
        
        # Get appropriate scanner
        scanner = self.scanners.get(scan_type.lower())
        if not scanner:
            logger.warning(f"No scanner available for type: {scan_type}")
            return []
        
        try:
            # Phase 1: Run heuristic scanner (detection)
            raw_findings = await scanner.scan(target_url)
            
            if not raw_findings:
                return []
            
            logger.info(
                f"Scanner detected {len(raw_findings)} potential findings for {scan_type}",
                target=target_url
            )
            
            # V21.0 - Record scanner metrics (NON-BREAKING)
            if self.metrics:
                scan_duration = (datetime.utcnow() - datetime.utcnow()).total_seconds()  # Will be updated
                self.metrics.record_scan(
                    scanner_type=scan_type,
                    duration=scan_duration,
                    findings_count=len(raw_findings)
                )
            
            # Phase 2-8: Process each finding through complete workflow
            processed_findings = []
            
            for finding in raw_findings:
                try:
                    logger.info(
                        f"Processing finding through workflow",
                        finding_id=str(finding.id),
                        type=scan_type
                    )
                    
                    # Run complete workflow: AI → Verification → Evidence → Authority
                    processed_finding = await self.workflow_orchestrator.process_finding_workflow(
                        finding_data={},  # Empty dict, we pass the finding object directly
                        finding=finding    # Pass existing finding from scanner
                    )
                    
                    processed_findings.append(processed_finding)
                    
                    logger.info(
                        f"Workflow complete for finding",
                        finding_id=str(processed_finding.id),
                        final_status=processed_finding.status.value,
                        ai_verdict=processed_finding.ai_analysis.verdict.value if processed_finding.ai_analysis else 'N/A'
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Workflow processing failed for finding",
                        finding_id=str(finding.id) if finding else 'unknown',
                        error=str(e)
                    )
                    
                    # V19.2 - Enhanced Fallback Mechanism
                    try:
                        # Create fallback result attached to finding
                        logger.warning(f"Applying heuristic fallback for finding {finding.id}")
                        fallback_result = global_fallback_manager.create_heuristic_fallback_result(
                            finding, 
                            reason=f"Workflow exception: {str(e)}"
                        )
                        
                        # Attach fallback analysis
                        finding.ai_analysis = fallback_result
                        finding.status = VerificationStatus.NEEDS_MANUAL
                        finding.authority_level = AuthorityLevel.HEURISTIC
                        
                        processed_findings.append(finding)
                        
                    except Exception as fallback_error:
                        logger.critical(f"Fallback mechanism failed: {fallback_error}")
                        # Last resort: Keep original finding
                        processed_findings.append(finding)
            
            return processed_findings
            
        except Exception as e:
            logger.error(f"Scanner {scan_type} failed: {e}", exc_info=True)
            return []
    
    def _finding_to_summary(self, finding: AssessmentResult) -> Dict[str, Any]:
        """Convert finding to summary dict."""
        return {
            'id': str(finding.id),
            'url': finding.url,
            'type': finding.finding_type.value if finding.finding_type else 'unknown',
            'severity': finding.severity.value if finding.severity else 'unknown',
            'status': finding.status.value if finding.status else 'unknown',
            'confidence': finding.heuristic_score
        }
    
    async def start_scan_async(self, config: Dict[str, Any]) -> str:
        """
        Start an async scan with the given configuration.
        Called by web_server.py for async scan operations.
        
        Args:
            config: Scan configuration with keys:
                - url: Target URL (required)
                - scan_types: List of scan types (optional, defaults to all)
                - options: Additional options (optional)
        
        Returns:
            scan_id: Unique identifier for tracking the scan
        """
        target_url = config.get('url')
        if not target_url:
            raise ValueError("Target URL is required")
        
        scan_types = config.get('scan_types', ['sqli', 'xss', 'lfi', 'ssrf', 'csrf', 'ssti', 'xxe', 'file_upload', 'jwt'])
        
        # Generate scan ID
        scan_id = str(uuid.uuid4())[:8]
        
        # Track active scan
        self.active_scans[scan_id] = {
            'target_url': target_url,
            'scan_types': scan_types,
            'status': 'running',
            'progress': 0,
            'findings_count': 0,
            'start_time': datetime.utcnow(),
            'findings': []
        }
        
        logger.info(f"Starting async scan {scan_id} for {target_url}")
        
        # Start scan in background
        asyncio.create_task(self._run_scan_background(scan_id, target_url, scan_types))
        
        return scan_id
    
    async def _run_scan_background(self, scan_id: str, target_url: str, scan_types: List[str]):
        """Run scan in background and update status."""
        try:
            # Run the actual scan
            result = await self.run_scan(target_url, scan_types)
            
            # Update scan status
            if scan_id in self.active_scans:
                self.active_scans[scan_id]['status'] = 'completed'
                self.active_scans[scan_id]['progress'] = 100
                self.active_scans[scan_id]['findings_count'] = result.get('findings_count', 0)
                self.active_scans[scan_id]['findings'] = result.get('findings', [])
                self.active_scans[scan_id]['end_time'] = datetime.utcnow()
                
        except Exception as e:
            logger.error(f"Background scan {scan_id} failed: {e}")
            if scan_id in self.active_scans:
                self.active_scans[scan_id]['status'] = 'failed'
                self.active_scans[scan_id]['error'] = str(e)
    
    def stop_scan(self, scan_id: str) -> bool:
        """
        Stop a running scan.
        
        Args:
            scan_id: ID of the scan to stop
            
        Returns:
            True if scan was stopped, False if not found
        """
        if scan_id in self.active_scans:
            self.active_scans[scan_id]['status'] = 'stopped'
            self.active_scans[scan_id]['end_time'] = datetime.utcnow()
            self.stop_requested = True
            logger.info(f"Scan {scan_id} stop requested")
            return True
        return False
    
    def get_scan_status(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a running or completed scan.
        
        Args:
            scan_id: ID of the scan
            
        Returns:
            Dict with scan status or None if not found
        """
        if scan_id not in self.active_scans:
            return None
        
        scan_info = self.active_scans[scan_id]
        
        # Calculate elapsed time
        start_time = scan_info.get('start_time', datetime.utcnow())
        end_time = scan_info.get('end_time', datetime.utcnow())
        elapsed = (end_time - start_time).total_seconds()
        
        return {
            'scan_id': scan_id,
            'target_url': scan_info.get('target_url'),
            'status': scan_info.get('status', 'unknown'),
            'progress': scan_info.get('progress', 0),
            'findings_count': scan_info.get('findings_count', 0),
            'elapsed_seconds': elapsed,
            'error': scan_info.get('error')
        }
    
    async def run_scan(self, target_url: str, scan_types: List[str]) -> Dict[str, Any]:
        """
        Run scan on target (wrapper for scan_target).
        Alias method for backward compatibility with _run_scan_background.
        
        Args:
            target_url: Target URL to scan
            scan_types: List of vulnerability types to scan for
            
        Returns:
            Scan results summary
        """
        return await self.scan_target(target_url, scan_types)
    
    async def get_findings(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get findings from database with optional filters.
        
        Args:
            severity: Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)
            status: Filter by status (SUBMIT_READY, NEEDS_MANUAL, etc.)
            limit: Maximum number of findings to return
            offset: Number of findings to skip
            
        Returns:
            List of finding dictionaries
        """
        try:
            # Get findings from database
            findings = await self.database.get_recent_findings(limit=limit)
            
            # Apply filters
            if severity:
                findings = [f for f in findings if f.severity and f.severity.value == severity]
            
            if status:
                findings = [f for f in findings if f.status and f.status.value == status]
            
            # Apply offset
            findings = findings[offset:]
            
            # Convert to dict format
            return [self._finding_to_summary(f) for f in findings]
            
        except Exception as e:
            logger.error(f"Failed to get findings: {e}")
            return []
    
    async def get_finding_by_id(self, finding_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific finding by ID.
        
        Args:
            finding_id: Unique identifier of the finding
            
        Returns:
            Finding dict or None if not found
        """
        try:
            finding = await self.database.get_finding(finding_id)
            
            if finding:
                return {
                    'id': str(finding.id),
                    'url': finding.url,
                    'type': finding.finding_type.value if finding.finding_type else 'unknown',
                    'severity': finding.severity.value if finding.severity else 'unknown',
                    'status': finding.status.value if finding.status else 'unknown',
                    'confidence': finding.heuristic_score,
                    'evidence': finding.evidence,
                    'payload': finding.payload,
                    'parameter': finding.vulnerable_parameter,
                    'ai_analysis': {
                        'verdict': finding.ai_analysis.verdict.value if finding.ai_analysis else None,
                        'confidence': finding.ai_analysis.confidence if finding.ai_analysis else None,
                        'reasoning': finding.ai_analysis.reasoning if finding.ai_analysis else None
                    } if finding.ai_analysis else None,
                    'verification': {
                        'success': finding.verification_result.success if finding.verification_result else None,
                        'match_type': finding.verification_result.match_type if finding.verification_result else None,
                        'confidence': finding.verification_result.confidence if finding.verification_result else None
                    } if finding.verification_result else None
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get finding {finding_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get engine statistics."""
        return self.stats.copy()
    
    async def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information and loaded components."""
        return {
            'version': 'V17.0 ULTIMATE',
            'scanners_loaded': list(self.scanners.keys()),
            'scanner_count': len(self.scanners),
            'recon_available': RECON_AVAILABLE,
            'stealth_available': STEALTH_AVAILABLE,
            'attack_chains_available': ATTACK_CHAINS_AVAILABLE,
            'profiling_enabled': PROFILING_ENABLED,
            'events_available': EVENTS_AVAILABLE,
            'stats': self.get_stats()
        }


# Global scan engine instance
global_scan_engine = VortexScanEngine()