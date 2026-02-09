#!/usr/bin/env python3
"""
Vortex - Enterprise-grade Bug Bounty Automation Framework
Main CLI interface with streaming output and legal compliance integration
"""

import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings, load_settings
from core.engine import VortexScanEngine
from core.legal_compliance import LegalGuardian
from core.streaming_memory import SemanticMemoryManager
from core.memory_manager import DynamicMemoryManager  # V20.0 - Advanced Memory Manager
from domain.enums import ScanMode, LogLevel
from utils.monitoring import SystemMonitor

# Initialize console and logger
console = Console()
logger = structlog.get_logger()

class VortexCLI:
    """Main CLI interface for Vortex security scanner."""
    
    def __init__(self):
        self.settings: Optional[Settings] = None
        self.engine: Optional[VortexScanEngine] = None
        self.legal_guardian: Optional[LegalGuardian] = None
        self.memory_manager: Optional[SemanticMemoryManager] = None
        self.dynamic_memory: Optional[DynamicMemoryManager] = None  # V20.0 - Advanced Memory Manager
        self.system_monitor: Optional[SystemMonitor] = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self, config_path: Optional[str] = None):
        """Initialize Vortex components with legal compliance validation."""
        try:
            # Load configuration
            self.settings = await load_settings(config_path)
            
            # Initialize legal guardian first (critical for compliance)
            self.legal_guardian = LegalGuardian(self.settings.legal)
            
            # Initialize memory manager (V20.0 - Using both managers)
            self.memory_manager = SemanticMemoryManager(
                max_memory_mb=self.settings.system.max_memory_mb
            )
            
            # V20.0 - Initialize advanced dynamic memory manager
            self.dynamic_memory = DynamicMemoryManager()
            
            # Initialize system monitor
            self.system_monitor = SystemMonitor(self.settings.monitoring)
            
            # Initialize main engine
            self.engine = VortexScanEngine()
            
            # CRITICAL: Initialize engine to start queue processor and all components
            await self.engine.initialize()
            
            # Start background monitoring
            await self.system_monitor.start()
            await self.memory_manager.start_monitoring()
            
            # V20.0 - Start dynamic memory monitoring (zone-based)
            # Note: Dynamic memory manager monitors automatically during operations
            logger.info(f"Dynamic Memory Manager: Zone={self.dynamic_memory.current_zone.value}")
            
            logger.info("Vortex initialized successfully",
                       version=self.settings.version,
                       legal_compliance=True,
                       memory_management="advanced")
            
        except Exception as e:
            logger.error("Failed to initialize Vortex", error=str(e))
            raise
    
    async def shutdown(self):
        """Graceful shutdown with cleanup."""
        logger.info("Initiating graceful shutdown...")
        
        if self.engine:
            await self.engine.shutdown()
        
        if self.system_monitor:
            await self.system_monitor.stop()
            
        if self.memory_manager:
            await self.memory_manager.stop_monitoring()
        
        logger.info("Vortex shutdown complete")

# Async wrapper for Click commands
def async_command(f):
    """Decorator to wrap async click commands."""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper

# CLI Commands
@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, verbose, debug):
    """Vortex - Enterprise Bug Bounty Automation Framework"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug
    
    # Configure logging level
    if debug:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))
    elif verbose:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

@cli.command()
@click.argument('targets', nargs=-1, required=True)
@click.option('--mode', '-m', type=click.Choice(['passive', 'active', 'aggressive']),
              default='active', help='Scan mode')
@click.option('--lightweight', is_flag=True, default=False, help='Enable lightweight mode (minimal resources, essential scanners only)')
@click.option('--ultra-lightweight', is_flag=True, default=False, help='Enable ultra-lightweight mode (512MB RAM, 3 scanners only)')
@click.option('--ci-cd-mode', is_flag=True, default=False, help='Enable CI/CD optimized mode (fast, deterministic, no AI)')
@click.option('--output', '-o', help='Output directory')
@click.option('--threads', '-t', type=int, default=10, help='Number of concurrent threads')
@click.option('--delay', '-d', type=float, default=1.0, help='Delay between requests (seconds)')
@click.option('--timeout', type=int, default=30, help='Request timeout (seconds)')
@click.option('--user-agent', help='Custom User-Agent string')
@click.option('--proxy', help='Single proxy (http://host:port or socks5://host:port)')
@click.option('--proxy-list', help='File containing proxy list (FREE - one per line)')
@click.option('--use-tor', is_flag=True, default=False, help='Use Tor SOCKS5 proxy (FREE - requires Tor running on localhost:9050)')
@click.option('--auth', help='Authentication (username:password)')
@click.option('--headers', multiple=True, help='Custom headers (key:value)')
@click.option('--scope-file', help='File containing authorized targets')
@click.option('--exclude', multiple=True, help='Exclude patterns')
@click.option('--include-vulns', multiple=True, help='Include specific vulnerability types')
@click.option('--exclude-vulns', multiple=True, help='Exclude specific vulnerability types')
@click.option('--ai-model', help='AI model to use for analysis')
@click.option('--quality-threshold', type=float, default=0.7, help='Minimum quality threshold')
@click.option('--legal-check', is_flag=True, default=False, help='Enable legal compliance checking (disabled by default)')
@click.option('--enable-recon', is_flag=True, default=False, help='Enable subdomain reconnaissance (discovers and scans all subdomains)')
@click.option('--enable-dom', is_flag=True, default=False, help='Enable DOM-based XSS scanning with Playwright (requires playwright)')
@click.option('--enable-graphql', is_flag=True, default=False, help='Enable GraphQL API security scanning')
@click.option('--enable-chains', is_flag=True, default=False, help='Enable multi-step attack chain detection and execution (V20.0)')
@click.option('--enable-mutations', is_flag=True, default=False, help='Enable intelligent payload mutations for WAF bypass (V20.0)')
@click.pass_context
@async_command
async def scan(ctx, targets, mode, lightweight, ultra_lightweight, ci_cd_mode, output, threads, delay, timeout, user_agent,
               proxy, proxy_list, use_tor, auth, headers, scope_file, exclude, include_vulns, exclude_vulns,
               ai_model, quality_threshold, legal_check, enable_recon, enable_dom, enable_graphql,
               enable_chains, enable_mutations):
    """Start a security scan on specified targets."""
    
    vortex = VortexCLI()
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        console.print("\n[yellow]Received shutdown signal. Stopping scan...[/yellow]")
        vortex.shutdown_event.set()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # V21.0 - Check for lightweight mode and configure memory manager
        lightweight_mode_obj = None
        is_lightweight = ultra_lightweight or lightweight or ci_cd_mode
        
        if ultra_lightweight:
            from core.modes import create_ultra_lightweight_mode, set_lightweight_mode
            lightweight_mode_obj = create_ultra_lightweight_mode()
            set_lightweight_mode(lightweight_mode_obj)
            console.print("[yellow]⚡ Ultra-Lightweight Mode Enabled[/yellow]")
            console.print(f"[dim]Memory: {lightweight_mode_obj.get_mode_summary()['estimated_memory']}[/dim]")
        elif ci_cd_mode:
            from core.modes import create_ci_cd_mode, set_lightweight_mode
            lightweight_mode_obj = create_ci_cd_mode()
            set_lightweight_mode(lightweight_mode_obj)
            console.print("[cyan]🔧 CI/CD Mode Enabled[/cyan]")
        elif lightweight:
            from core.modes import create_lightweight_mode, set_lightweight_mode
            lightweight_mode_obj = create_lightweight_mode()
            set_lightweight_mode(lightweight_mode_obj)
            console.print("[green]💡 Lightweight Mode Enabled[/green]")
            console.print(f"[dim]Memory: {lightweight_mode_obj.get_mode_summary()['estimated_memory']}[/dim]")
        
        # V21.0 - Configure memory manager for lightweight mode
        if is_lightweight:
            from core.memory_manager import global_memory_manager
            global_memory_manager.lightweight_mode = True
            global_memory_manager.MEMORY_LIMIT_MB = 1024.0
            global_memory_manager.GREEN_THRESHOLD = 0.50
            global_memory_manager.YELLOW_THRESHOLD = 0.65
            global_memory_manager.RED_THRESHOLD = 0.80
            global_memory_manager.EMERGENCY_THRESHOLD = 0.90
            console.print("[dim]Memory manager configured for lightweight mode (1GB limit)[/dim]")
        
        # Initialize Vortex
        await vortex.initialize(ctx.obj['config'])
        
        # V22.0 - Configure proxies (FREE)
        if proxy or proxy_list or use_tor:
            from core.stealth.evasion import ProxyManager
            
            # Get proxy manager from network client if available
            if hasattr(vortex.engine, 'network_client'):
                proxy_mgr = vortex.engine.network_client.proxy_manager
                if proxy_mgr:
                    # Add Tor proxy
                    if use_tor:
                        proxy_mgr.add_tor_proxy()
                        console.print("[green]✓ Tor proxy enabled (FREE)[/green]")
                        console.print("[dim]Make sure Tor is running: brew install tor && tor[/dim]")
                    
                    # Load proxy list
                    if proxy_list:
                        proxy_mgr.load_proxy_list(proxy_list)
                        stats = proxy_mgr.get_stats()
                        console.print(f"[green]✓ Loaded {stats['total_proxies']} proxies from file (FREE)[/green]")
                    
                    # Add single proxy
                    if proxy:
                        # Parse proxy URL
                        if '://' in proxy:
                            protocol, rest = proxy.split('://', 1)
                            if ':' in rest:
                                host, port = rest.rsplit(':', 1)
                                proxy_mgr.add_proxy(protocol, host, int(port))
                                console.print(f"[green]✓ Single proxy added: {protocol}://{host}:{port} (FREE)[/green]")
                        else:
                            console.print("[yellow]⚠ Invalid proxy format. Use: http://host:port or socks5://host:port[/yellow]")
                    
                    # Enable proxy usage
                    vortex.engine.network_client.use_proxies = True
                    console.print("[cyan]🔒 Proxy rotation enabled[/cyan]")
        
        # Legal compliance check (DISABLED - only check if explicitly enabled)
        if legal_check:
            console.print("[yellow]⚠ Legal compliance check is enabled[/yellow]")
            console.print("[blue]Performing legal compliance validation...[/blue]")
            try:
                for target in targets:
                    if not await vortex.legal_guardian.validate_target_authorization(target):
                        console.print(f"[red]ERROR: Target {target} not authorized for scanning[/red]")
                        return
                console.print("[green]✓ All targets authorized for scanning[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Legal check failed: {e}. Continuing anyway...[/yellow]")
        
        # V21.0 - Lightweight mode: Filter scanners
        if lightweight_mode_obj:
            # Use lightweight mode scanner selection
            default_scan_types = lightweight_mode_obj.get_active_scanners()
            console.print(f"[dim]Active scanners: {', '.join(default_scan_types)}[/dim]")
            
            # Override flags if conflicting with lightweight mode
            if not lightweight_mode_obj.should_use_playwright() and enable_dom:
                console.print("[yellow]⚠ DOM XSS disabled (Playwright not available in lightweight mode)[/yellow]")
                enable_dom = False
            
            if not lightweight_mode_obj.should_use_ai():
                console.print("[yellow]⚠ AI analysis disabled in lightweight mode[/yellow]")
        else:
            # V20.0 - Add DOM and GraphQL to scan types if enabled
            default_scan_types = ['sqli', 'xss', 'lfi', 'ssrf', 'csrf', 'ssti', 'xxe', 'file_upload', 'jwt']
            if enable_dom:
                default_scan_types.append('dom_xss')
                console.print("[cyan]🌐 DOM XSS Scanner enabled (Playwright)[/cyan]")
            if enable_graphql:
                default_scan_types.append('graphql')
                console.print("[cyan]📊 GraphQL Security Scanner enabled[/cyan]")
        
        # V20.0 - Display attack chain and mutation status
        if enable_chains:
            console.print("[cyan]🔗 Multi-Step Attack Chain Detection enabled[/cyan]")
        if enable_mutations:
            console.print("[cyan]🧬 Intelligent Payload Mutations enabled[/cyan]")
        
        # Configure scan parameters
        scan_config = {
            'targets': list(targets),
            'mode': ScanMode(mode),
            'output_dir': output,
            'max_concurrent': threads,
            'request_delay': delay,
            'timeout': timeout,
            'user_agent': user_agent,
            'proxy': proxy,
            'auth': auth,
            'custom_headers': dict(h.split(':', 1) for h in headers if ':' in h),
            'scope_file': scope_file,
            'exclude_patterns': list(exclude),
            'include_vulns': list(include_vulns) if include_vulns else default_scan_types,
            'exclude_vulns': list(exclude_vulns) if exclude_vulns else None,
            'ai_model': ai_model,
            'quality_threshold': quality_threshold
        }
        
        # Display scan configuration
        config_table = Table(title="Scan Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Targets", ", ".join(targets))
        config_table.add_row("Mode", mode.upper())
        config_table.add_row("Threads", str(threads))
        config_table.add_row("Delay", f"{delay}s")
        config_table.add_row("Quality Threshold", f"{quality_threshold}")
        
        console.print(config_table)
        
        # Start scan with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            scan_task = progress.add_task("Scanning targets...", total=len(targets))
            
            # Display recon status if enabled
            if enable_recon:
                console.print("[cyan]🔭 Subdomain reconnaissance enabled - discovering subdomains...[/cyan]")
            
            # V20.0 - Show memory status
            mem_status = vortex.dynamic_memory.get_status()
            console.print(f"[dim]Memory Zone: {mem_status['zone']} ({mem_status['memory_percent']:.1f}%)[/dim]")
            
            # Scan each target directly (CLI should not use queue system)
            all_findings = []
            total_subdomains_found = 0
            
            for idx, target in enumerate(targets):
                if vortex.shutdown_event.is_set():
                    break
                
                # V20.0 - Check memory and apply backpressure if needed
                if await vortex.dynamic_memory.should_apply_backpressure():
                    console.print("[yellow]⚠ Memory pressure detected - slowing down...[/yellow]")
                    await asyncio.sleep(2)
                    await vortex.dynamic_memory.auto_manage_memory()
                
                # Scan this target directly (with optional recon, advanced scanners, and chains)
                target_results = await vortex.engine.scan_target(
                    target_url=target,
                    scan_types=list(include_vulns) if include_vulns else default_scan_types,
                    enable_recon=enable_recon,
                    enable_chains=enable_chains
                )
                
                if target_results:
                    if target_results.get('findings'):
                        all_findings.extend(target_results['findings'])
                    
                    # Track subdomains discovered
                    if enable_recon:
                        subdomains = target_results.get('subdomains_discovered', 0)
                        total_subdomains_found += subdomains
                        if subdomains > 0:
                            console.print(f"[green]✓ Found {subdomains} subdomains for {target}[/green]")
                
                # Update progress
                progress.update(scan_task, completed=idx + 1)
            
            # Create results summary
            results = {
                'findings': all_findings,
                'total_targets': len(targets),
                'subdomains_discovered': total_subdomains_found if enable_recon else 0,
                'duration': 'N/A'
            }
            
            # Display results summary
            await display_scan_results(results, console)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Scan interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Scan failed: {e}[/red]")
        logger.error("Scan failed", error=str(e))
    finally:
        await vortex.shutdown()

@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'html', 'markdown', 'pdf']), default='markdown')
@click.option('--output', '-o', help='Output file path')
@click.option('--include-poc', is_flag=True, default=True, help='Include PoC for findings')
@click.argument('scan_id', required=False)
@click.pass_context
@async_command
async def report(ctx, scan_id, format, output, include_poc):
    """Generate a detailed security report."""
    from pathlib import Path
    from utils.poc_generator import poc_generator
    import json
    from datetime import datetime
    
    vortex = VortexCLI()
    
    try:
        await vortex.initialize(ctx.obj['config'])
        
        console.print(f"[blue]Generating {format.upper()} report...[/blue]\n")
        
        # Get all findings (or specific scan)
        findings = await vortex.engine.get_findings(limit=1000)
        
        if not findings:
            console.print("[yellow]No findings to report[/yellow]")
            return
        
        # Group by severity
        severity_groups = {}
        for finding in findings:
            sev = finding.severity.value if finding.severity else 'UNKNOWN'
            if sev not in severity_groups:
                severity_groups[sev] = []
            severity_groups[sev].append(finding)
        
        # Generate report based on format
        if format == 'json':
            report_data = {
                'generated_at': datetime.utcnow().isoformat(),
                'total_findings': len(findings),
                'severity_breakdown': {k: len(v) for k, v in severity_groups.items()},
                'findings': [f.to_dict() for f in findings]
            }
            
            if output:
                with open(output, 'w') as f:
                    json.dump(report_data, f, indent=2)
                console.print(f"[green]✓ JSON report saved to: {output}[/green]")
            else:
                console.print(json.dumps(report_data, indent=2))
        
        elif format == 'markdown':
            report_md = generate_markdown_report(findings, include_poc, poc_generator)
            
            if output:
                with open(output, 'w') as f:
                    f.write(report_md)
                console.print(f"[green]✓ Markdown report saved to: {output}[/green]")
            else:
                console.print(report_md)
        
        elif format == 'html':
            report_html = generate_html_report(findings, include_poc, poc_generator)
            
            if output:
                with open(output, 'w') as f:
                    f.write(report_html)
                console.print(f"[green]✓ HTML report saved to: {output}[/green]")
            else:
                # Save to default location
                output_dir = Path('output/reports')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
                with open(output_file, 'w') as f:
                    f.write(report_html)
                console.print(f"[green]✓ HTML report saved to: {output_file}[/green]")
        
        # Display summary
        console.print(f"\n[bold]Report Summary:[/bold]")
        console.print(f"  Total Findings: {len(findings)}")
        for severity, count in sorted(severity_groups.items(), key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}.get(x[0], 4)):
            color = {'CRITICAL': 'bright_red', 'HIGH': 'red', 'MEDIUM': 'yellow', 'LOW': 'green'}.get(severity, 'white')
            console.print(f"  [{color}]{severity}[/{color}]: {count}")
        
    except Exception as e:
        console.print(f"[red]Report generation failed: {e}[/red]")
    finally:
        await vortex.shutdown()


def generate_markdown_report(findings, include_poc, poc_gen):
    """Generate Markdown format report."""
    from datetime import datetime
    
    report = f"""# VORTEX Security Assessment Report

**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
**Total Findings**: {len(findings)}

## Executive Summary

This report contains {len(findings)} security findings identified by the VORTEX automated security scanner.

### Severity Breakdown

"""
    
    severity_counts = {}
    for finding in findings:
        sev = finding.severity.value if finding.severity else 'UNKNOWN'
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        count = severity_counts.get(severity, 0)
        if count > 0:
            report += f"- **{severity}**: {count}\n"
    
    report += "\n## Findings\n\n"
    
    # Sort by severity
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(f.severity.value if f.severity else 'INFO', 5))
    
    for idx, finding in enumerate(sorted_findings, 1):
        report += f"### {idx}. {finding.finding_type.value if finding.finding_type else 'Unknown'}\n\n"
        report += f"- **Severity**: {finding.severity.value if finding.severity else 'N/A'}\n"
        report += f"- **Status**: {finding.status.value if finding.status else 'N/A'}\n"
        report += f"- **URL**: `{finding.url}`\n"
        report += f"- **Confidence**: {finding.heuristic_score:.1%}\n"
        
        if finding.vulnerable_parameter:
            report += f"- **Parameter**: `{finding.vulnerable_parameter}`\n"
        
        if finding.payload:
            report += f"- **Payload**: `{finding.payload}`\n"
        
        report += "\n"
        
        if finding.evidence:
            report += f"**Evidence**:\n```\n{finding.evidence[:500]}\n```\n\n"
        
        # Add PoC if requested
        if include_poc and finding.status.value in ['SUBMIT_READY', 'SYSTEM_VERIFIED']:
            try:
                poc_data = poc_gen.generate_poc(finding.to_dict() if hasattr(finding, 'to_dict') else {})
                if poc_data.get('markdown'):
                    report += f"\n{poc_data['markdown']}\n\n"
            except:
                pass
        
        report += "---\n\n"
    
    report += f"\n\n*Report generated by VORTEX Security Scanner v1.0*\n"
    
    return report


def generate_html_report(findings, include_poc, poc_gen):
    """Generate HTML format report."""
    from datetime import datetime
    
    severity_counts = {}
    for finding in findings:
        sev = finding.severity.value if finding.severity else 'UNKNOWN'
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VORTEX Security Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        .summary {{ background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .severity-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: 600; margin-right: 10px; }}
        .critical {{ background: #d32f2f; }}
        .high {{ background: #ef6c00; }}
        .medium {{ background: #fbc02d; color: #000; }}
        .low {{ background: #388e3c; }}
        .finding {{ border: 1px solid #e0e0e0; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .finding-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .evidence {{ background: #f5f5f5; padding: 15px; border-radius: 4px; font-family: monospace; overflow-x: auto; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 VORTEX Security Assessment Report</h1>
        
        <div class="summary">
            <p><strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            <p><strong>Total Findings:</strong> {len(findings)}</p>
            <p><strong>Severity Breakdown:</strong></p>
"""
    
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_counts.get(severity, 0)
        if count > 0:
            css_class = severity.lower()
            html += f'            <span class="severity-badge {css_class}">{severity}: {count}</span>\n'
    
    html += """        </div>
        
        <h2>Findings</h2>
"""
    
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(f.severity.value if f.severity else 'INFO', 5))
    
    for idx, finding in enumerate(sorted_findings, 1):
        sev = finding.severity.value if finding.severity else 'UNKNOWN'
        sev_class = sev.lower()
        
        html += f"""        <div class="finding">
            <div class="finding-header">
                <h3>{idx}. {finding.finding_type.value if finding.finding_type else 'Unknown'}</h3>
                <span class="severity-badge {sev_class}">{sev}</span>
            </div>
            <p><strong>URL:</strong> <code>{finding.url}</code></p>
            <p><strong>Status:</strong> {finding.status.value if finding.status else 'N/A'}</p>
            <p><strong>Confidence:</strong> {finding.heuristic_score:.1%}</p>
"""
        
        if finding.vulnerable_parameter:
            html += f'            <p><strong>Parameter:</strong> <code>{finding.vulnerable_parameter}</code></p>\n'
        
        if finding.payload:
            html += f'            <p><strong>Payload:</strong> <code>{finding.payload}</code></p>\n'
        
        if finding.evidence:
            evidence = finding.evidence[:500].replace('<', '&lt;').replace('>', '&gt;')
            html += f'            <div class="evidence">{evidence}</div>\n'
        
        html += "        </div>\n"
    
    html += """    </div>
</body>
</html>"""
    
    return html

@cli.command()
@async_command
async def status():
    """Show system status and health metrics."""
    vortex = VortexCLI()
    
    try:
        await vortex.initialize()
        
        # Get system status
        status_data = await vortex.system_monitor.get_system_status()
        
        # Display status table
        status_table = Table(title="Vortex System Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details")
        
        for component, info in status_data.items():
            status_color = "green" if info['healthy'] else "red"
            status_table.add_row(
                component.title(),
                f"[{status_color}]{info['status']}[/{status_color}]",
                info.get('details', '')
            )
        
        console.print(status_table)
        
    except Exception as e:
        console.print(f"[red]Failed to get system status: {e}[/red]")
    finally:
        await vortex.shutdown()

async def display_scan_results(results, console):
    """Display scan results in a formatted table."""
    if not results or not results.get('findings'):
        console.print("[yellow]No vulnerabilities found[/yellow]")
        return
    
    findings_table = Table(title="Scan Results")
    findings_table.add_column("ID", style="cyan")
    findings_table.add_column("Type", style="magenta")
    findings_table.add_column("Severity", style="red")
    findings_table.add_column("URL", style="blue")
    findings_table.add_column("Status", style="green")
    findings_table.add_column("Confidence", style="yellow")
    
    for finding in results['findings']:
        # Finding is a dict
        severity = finding.get('severity', 'UNKNOWN')
        severity_color = {
            'CRITICAL': 'bright_red',
            'HIGH': 'red',
            'MEDIUM': 'yellow',
            'LOW': 'green',
            'INFO': 'blue'
        }.get(severity, 'white')
        
        url = finding.get('url', '')
        findings_table.add_row(
            finding.get('id', '')[:8],
            finding.get('type', 'unknown'),
            f"[{severity_color}]{severity}[/{severity_color}]",
            url[:50] + "..." if len(url) > 50 else url,
            finding.get('status', 'unknown'),
            f"{finding.get('confidence', 0.0):.2f}"
        )
    
    console.print(findings_table)
    
    # Summary statistics
    findings_list = results['findings']
    subdomains_info = ""
    if results.get('subdomains_discovered', 0) > 0:
        subdomains_info = f"Subdomains Discovered: {results.get('subdomains_discovered')}\n"
    
    summary_panel = Panel(
        f"Total Findings: {len(findings_list)}\n"
        f"{subdomains_info}"
        f"Critical: {sum(1 for f in findings_list if f.get('severity') == 'CRITICAL')}\n"
        f"High: {sum(1 for f in findings_list if f.get('severity') == 'HIGH')}\n"
        f"Medium: {sum(1 for f in findings_list if f.get('severity') == 'MEDIUM')}\n"
        f"Low: {sum(1 for f in findings_list if f.get('severity') == 'LOW')}\n"
        f"Scan Duration: {results.get('duration', 'Unknown')}",
        title="Scan Summary",
        border_style="green"
    )
    console.print(summary_panel)

if __name__ == "__main__":
    # Handle async CLI
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    cli()
