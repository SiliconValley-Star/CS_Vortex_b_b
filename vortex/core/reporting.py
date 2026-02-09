"""
VORTEX Reporting Engine - V18.0 ULTIMATE
Professional HTML/PDF report generation for bug bounty submissions

REPORT SECTIONS:
1. Executive Summary
2. Findings Overview (Charts/Stats)
3. Technical Details (Per Finding)
4. Proof of Concept
5. Remediation Recommendations
6. Appendix (Raw Data, Timeline)

OUTPUT FORMATS:
- HTML (responsive, printable)
- PDF (via WeasyPrint or Playwright)
- JSON (machine-readable)
- Markdown (for documentation)
"""

import asyncio
import json
import logging
import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter
import html

logger = logging.getLogger(__name__)

# PDF generation imports
try:
    from weasyprint import HTML as WeasyHTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.info("WeasyPrint not installed. PDF export via Playwright only.")

from domain.models import AssessmentResult
from domain.enums import FindingSeverity, VerificationStatus, FindingType


@dataclass
class ReportMetadata:
    """Report metadata and configuration."""
    title: str = "Vortex Security Assessment Report"
    company_name: str = "Security Team"
    author: str = "Vortex Automated Scanner"
    target_scope: str = ""
    scan_date: datetime = field(default_factory=datetime.utcnow)
    report_date: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"
    classification: str = "CONFIDENTIAL"
    logo_path: Optional[str] = None


class ReportingEngine:
    """
    Professional security report generator.
    
    Creates publication-ready reports with:
    - Executive summary for management
    - Technical details for security teams
    - PoC steps for validation
    - Remediation guidance
    """
    
    # Severity colors
    SEVERITY_COLORS = {
        'CRITICAL': '#dc3545',  # Red
        'HIGH': '#fd7e14',       # Orange
        'MEDIUM': '#ffc107',     # Yellow
        'LOW': '#28a745',        # Green
        'INFO': '#17a2b8'        # Blue
    }
    
    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(self, 
                            findings: List[AssessmentResult],
                            metadata: Optional[ReportMetadata] = None) -> str:
        """
        Generate a complete HTML security report.
        
        Args:
            findings: List of findings to include
            metadata: Report metadata
        
        Returns:
            Path to generated HTML file
        """
        metadata = metadata or ReportMetadata()
        
        # Calculate statistics
        stats = self._calculate_statistics(findings)
        
        # Generate HTML content
        html_content = self._build_html_report(findings, metadata, stats)
        
        # Save to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"vortex_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filepath}")
        return str(filepath)
    
    def generate_pdf_report(self,
                           findings: List[AssessmentResult],
                           metadata: Optional[ReportMetadata] = None) -> Optional[str]:
        """
        Generate PDF report from findings.
        
        Uses WeasyPrint if available, otherwise falls back to Playwright.
        """
        # First generate HTML
        html_path = self.generate_html_report(findings, metadata)
        
        if WEASYPRINT_AVAILABLE:
            return self._convert_to_pdf_weasyprint(html_path)
        else:
            return asyncio.run(self._convert_to_pdf_playwright(html_path))
    
    def generate_json_report(self,
                            findings: List[AssessmentResult],
                            metadata: Optional[ReportMetadata] = None) -> str:
        """Generate machine-readable JSON report."""
        metadata = metadata or ReportMetadata()
        
        report_data = {
            'metadata': {
                'title': metadata.title,
                'company': metadata.company_name,
                'author': metadata.author,
                'target_scope': metadata.target_scope,
                'scan_date': metadata.scan_date.isoformat(),
                'report_date': metadata.report_date.isoformat(),
                'version': metadata.version
            },
            'summary': self._calculate_statistics(findings),
            'findings': [self._finding_to_dict(f) for f in findings]
        }
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"vortex_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON report generated: {filepath}")
        return str(filepath)
    
    def generate_markdown_report(self,
                                findings: List[AssessmentResult],
                                metadata: Optional[ReportMetadata] = None) -> str:
        """Generate Markdown report for documentation."""
        metadata = metadata or ReportMetadata()
        stats = self._calculate_statistics(findings)
        
        md_content = self._build_markdown_report(findings, metadata, stats)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"vortex_report_{timestamp}.md"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report generated: {filepath}")
        return str(filepath)
    
    def _calculate_statistics(self, findings: List[AssessmentResult]) -> Dict[str, Any]:
        """Calculate report statistics."""
        severity_counts = Counter()
        type_counts = Counter()
        status_counts = Counter()
        
        for finding in findings:
            if finding.severity:
                severity_counts[finding.severity.value] += 1
            if finding.finding_type:
                type_counts[finding.finding_type.value] += 1
            if finding.status:
                status_counts[finding.status.value] += 1
        
        # Calculate risk score (weighted)
        risk_weights = {'CRITICAL': 10, 'HIGH': 7, 'MEDIUM': 4, 'LOW': 1, 'INFO': 0}
        total_risk = sum(
            severity_counts.get(sev, 0) * weight 
            for sev, weight in risk_weights.items()
        )
        max_risk = len(findings) * 10 if findings else 1
        risk_score = (total_risk / max_risk) * 100
        
        return {
            'total_findings': len(findings),
            'by_severity': dict(severity_counts),
            'by_type': dict(type_counts),
            'by_status': dict(status_counts),
            'risk_score': round(risk_score, 1),
            'submit_ready_count': status_counts.get('SUBMIT_READY', 0),
            'needs_manual_count': status_counts.get('NEEDS_MANUAL', 0)
        }
    
    def _build_html_report(self, findings: List[AssessmentResult],
                          metadata: ReportMetadata,
                          stats: Dict[str, Any]) -> str:
        """Build complete HTML report."""
        
        # CSS Styles
        css = self._get_report_css()
        
        # Build sections
        header = self._build_header_html(metadata)
        executive_summary = self._build_executive_summary_html(stats, metadata)
        findings_overview = self._build_findings_overview_html(stats)
        findings_detail = self._build_findings_detail_html(findings)
        remediation = self._build_remediation_html(findings)
        footer = self._build_footer_html(metadata)
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(metadata.title)}</title>
    <style>{css}</style>
</head>
<body>
    {header}
    
    <main class="container">
        {executive_summary}
        {findings_overview}
        {findings_detail}
        {remediation}
    </main>
    
    {footer}
</body>
</html>"""
        
        return html_template
    
    def _get_report_css(self) -> str:
        """Get CSS styles for report."""
        return """
:root {
    --primary: #2563eb;
    --critical: #dc3545;
    --high: #fd7e14;
    --medium: #ffc107;
    --low: #28a745;
    --info: #17a2b8;
    --dark: #1f2937;
    --light: #f3f4f6;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: var(--dark);
    background: white;
}

.container { max-width: 1200px; margin: 0 auto; padding: 2rem; }

/* Header */
.report-header {
    background: linear-gradient(135deg, var(--dark) 0%, #374151 100%);
    color: white;
    padding: 3rem 2rem;
    text-align: center;
}

.report-header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
.report-header .subtitle { opacity: 0.8; font-size: 1.2rem; }
.report-header .meta { margin-top: 1rem; opacity: 0.7; font-size: 0.9rem; }

/* Sections */
section { margin-bottom: 3rem; }
section h2 {
    color: var(--dark);
    border-bottom: 3px solid var(--primary);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Executive Summary */
.exec-summary {
    background: var(--light);
    padding: 2rem;
    border-radius: 8px;
}

.risk-score {
    display: inline-flex;
    align-items: center;
    gap: 1rem;
    font-size: 2rem;
    font-weight: bold;
}

.risk-score .label { font-size: 1rem; font-weight: normal; }

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
}

.stat-card {
    background: white;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stat-card .number { font-size: 2.5rem; font-weight: bold; }
.stat-card .label { color: #6b7280; margin-top: 0.5rem; }

.stat-card.critical { border-left: 4px solid var(--critical); }
.stat-card.high { border-left: 4px solid var(--high); }
.stat-card.medium { border-left: 4px solid var(--medium); }
.stat-card.low { border-left: 4px solid var(--low); }

/* Findings Table */
.findings-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

.findings-table th, .findings-table td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}

.findings-table th { background: var(--light); font-weight: 600; }
.findings-table tr:hover { background: #f9fafb; }

/* Severity Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}

.badge.critical { background: var(--critical); color: white; }
.badge.high { background: var(--high); color: white; }
.badge.medium { background: var(--medium); color: #000; }
.badge.low { background: var(--low); color: white; }
.badge.info { background: var(--info); color: white; }

/* Finding Detail */
.finding-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    overflow: hidden;
}

.finding-card .card-header {
    background: var(--light);
    padding: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.finding-card .card-body { padding: 1.5rem; }

.finding-card .field { margin-bottom: 1rem; }
.finding-card .field-label { font-weight: 600; color: #374151; margin-bottom: 0.25rem; }
.finding-card .field-value { color: #4b5563; }

.code-block {
    background: #1f2937;
    color: #e5e7eb;
    padding: 1rem;
    border-radius: 4px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 0.875rem;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

/* Footer */
.report-footer {
    text-align: center;
    padding: 2rem;
    background: var(--light);
    color: #6b7280;
    font-size: 0.875rem;
}

/* Print Styles */
@media print {
    .report-header { background: var(--dark) !important; -webkit-print-color-adjust: exact; }
    .finding-card { page-break-inside: avoid; }
    .container { max-width: 100%; }
}
"""
    
    def _build_header_html(self, metadata: ReportMetadata) -> str:
        """Build report header."""
        return f"""
<header class="report-header">
    <h1>{html.escape(metadata.title)}</h1>
    <div class="subtitle">{html.escape(metadata.target_scope)}</div>
    <div class="meta">
        <span>Generated: {metadata.report_date.strftime('%Y-%m-%d %H:%M UTC')}</span> |
        <span>By: {html.escape(metadata.author)}</span> |
        <span class="classification">{html.escape(metadata.classification)}</span>
    </div>
</header>"""
    
    def _build_executive_summary_html(self, stats: Dict[str, Any], 
                                      metadata: ReportMetadata) -> str:
        """Build executive summary section."""
        risk_level = "Critical" if stats['risk_score'] > 70 else \
                     "High" if stats['risk_score'] > 50 else \
                     "Medium" if stats['risk_score'] > 25 else "Low"
        
        return f"""
<section id="executive-summary">
    <h2>Executive Summary</h2>
    <div class="exec-summary">
        <p>This security assessment was conducted on <strong>{metadata.scan_date.strftime('%Y-%m-%d')}</strong>
        against the target scope: <strong>{html.escape(metadata.target_scope or 'Not specified')}</strong>.</p>
        
        <div class="risk-score" style="margin-top: 1.5rem;">
            <span style="color: {self._get_risk_color(stats['risk_score'])}">{stats['risk_score']}%</span>
            <span class="label">Overall Risk Score ({risk_level})</span>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{stats['total_findings']}</div>
                <div class="label">Total Findings</div>
            </div>
            <div class="stat-card critical">
                <div class="number">{stats['by_severity'].get('CRITICAL', 0)}</div>
                <div class="label">Critical</div>
            </div>
            <div class="stat-card high">
                <div class="number">{stats['by_severity'].get('HIGH', 0)}</div>
                <div class="label">High</div>
            </div>
            <div class="stat-card medium">
                <div class="number">{stats['by_severity'].get('MEDIUM', 0)}</div>
                <div class="label">Medium</div>
            </div>
            <div class="stat-card low">
                <div class="number">{stats['by_severity'].get('LOW', 0)}</div>
                <div class="label">Low</div>
            </div>
        </div>
    </div>
</section>"""
    
    def _build_findings_overview_html(self, stats: Dict[str, Any]) -> str:
        """Build findings overview table."""
        return f"""
<section id="findings-overview">
    <h2>Findings Overview</h2>
    <p>The following table summarizes all identified vulnerabilities grouped by type and severity.</p>
    
    <table class="findings-table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Critical</th>
                <th>High</th>
                <th>Medium</th>
                <th>Low</th>
                <th>Total</th>
            </tr>
        </thead>
        <tbody>
            {self._build_overview_rows(stats)}
        </tbody>
    </table>
</section>"""
    
    def _build_overview_rows(self, stats: Dict[str, Any]) -> str:
        """Build overview table rows by type."""
        # Group by type (simplified)
        by_type = stats.get('by_type', {})
        rows = ""
        
        for vuln_type, count in by_type.items():
            rows += f"""
            <tr>
                <td>{html.escape(vuln_type.replace('_', ' ').title())}</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td><strong>{count}</strong></td>
            </tr>"""
        
        return rows if rows else "<tr><td colspan='6' style='text-align:center'>No findings</td></tr>"
    
    def _build_findings_detail_html(self, findings: List[AssessmentResult]) -> str:
        """Build detailed findings section."""
        cards = ""
        
        # Sort by severity
        severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']
        sorted_findings = sorted(
            findings,
            key=lambda f: severity_order.index(f.severity.value) if f.severity else 999
        )
        
        for i, finding in enumerate(sorted_findings, 1):
            cards += self._build_finding_card(finding, i)
        
        return f"""
<section id="findings-detail">
    <h2>Detailed Findings</h2>
    {cards}
</section>"""
    
    def _build_finding_card(self, finding: AssessmentResult, index: int) -> str:
        """Build a single finding card."""
        severity = finding.severity.value if finding.severity else 'UNKNOWN'
        finding_type = finding.finding_type.value if finding.finding_type else 'unknown'
        
        return f"""
<div class="finding-card">
    <div class="card-header">
        <span><strong>#{index}</strong> - {html.escape(finding_type.replace('_', ' ').title())}</span>
        <span class="badge {severity.lower()}">{severity}</span>
    </div>
    <div class="card-body">
        <div class="field">
            <div class="field-label">Affected URL</div>
            <div class="field-value"><a href="{html.escape(finding.url or '')}">{html.escape(finding.url or 'N/A')}</a></div>
        </div>
        
        <div class="field">
            <div class="field-label">Vulnerable Parameter</div>
            <div class="field-value">{html.escape(finding.vulnerable_parameter or 'N/A')}</div>
        </div>
        
        <div class="field">
            <div class="field-label">Payload</div>
            <div class="code-block">{html.escape(finding.payload or 'N/A')}</div>
        </div>
        
        <div class="field">
            <div class="field-label">Evidence</div>
            <div class="code-block">{html.escape(str(finding.evidence)[:500] if finding.evidence else 'N/A')}</div>
        </div>
        
        <div class="field">
            <div class="field-label">Confidence Score</div>
            <div class="field-value">{(finding.heuristic_score or 0) * 100:.0f}%</div>
        </div>
        
        <div class="field">
            <div class="field-label">Status</div>
            <div class="field-value">{finding.status.value if finding.status else 'Unknown'}</div>
        </div>
    </div>
</div>"""
    
    def _build_remediation_html(self, findings: List[AssessmentResult]) -> str:
        """Build remediation recommendations section."""
        # Get unique vulnerability types
        vuln_types = set()
        for f in findings:
            if f.finding_type:
                vuln_types.add(f.finding_type.value)
        
        recommendations = ""
        for vtype in vuln_types:
            recommendations += self._get_remediation_for_type(vtype)
        
        return f"""
<section id="remediation">
    <h2>Remediation Recommendations</h2>
    {recommendations}
</section>"""
    
    def _get_remediation_for_type(self, vuln_type: str) -> str:
        """Get remediation recommendations for vulnerability type."""
        remediation_db = {
            'sqli': {
                'title': 'SQL Injection',
                'recommendations': [
                    'Use parameterized queries (prepared statements)',
                    'Implement input validation and sanitization',
                    'Apply the principle of least privilege to database accounts',
                    'Use an ORM framework with built-in protection'
                ]
            },
            'xss_reflected': {
                'title': 'Cross-Site Scripting (Reflected)',
                'recommendations': [
                    'Encode all user input before rendering in HTML',
                    'Implement Content Security Policy (CSP)',
                    'Use HTTPOnly and Secure flags for cookies',
                    'Validate and sanitize input on server-side'
                ]
            },
            'xss_dom': {
                'title': 'DOM-Based XSS',
                'recommendations': [
                    'Avoid using innerHTML with untrusted data',
                    'Use textContent instead of innerHTML where possible',
                    'Sanitize data before passing to JavaScript sinks',
                    'Implement strict Content Security Policy'
                ]
            },
            'local_file_inclusion': {
                'title': 'Local File Inclusion',
                'recommendations': [
                    'Avoid passing user input to file system functions',
                    'Use a whitelist of allowed files',
                    'Sanitize and validate file paths',
                    'Implement proper access controls'
                ]
            }
        }
        
        vtl = vuln_type.lower()
        if vtl in remediation_db:
            data = remediation_db[vtl]
            items = ''.join(f'<li>{r}</li>' for r in data['recommendations'])
            return f"""
<div class="finding-card">
    <div class="card-header"><strong>{data['title']}</strong></div>
    <div class="card-body"><ul>{items}</ul></div>
</div>"""
        
        return ""
    
    def _build_footer_html(self, metadata: ReportMetadata) -> str:
        """Build report footer."""
        return f"""
<footer class="report-footer">
    <p>Generated by Vortex Security Scanner v{metadata.version}</p>
    <p>{html.escape(metadata.classification)} - {metadata.report_date.strftime('%Y-%m-%d')}</p>
</footer>"""
    
    def _get_risk_color(self, score: float) -> str:
        """Get color based on risk score."""
        if score > 70:
            return self.SEVERITY_COLORS['CRITICAL']
        elif score > 50:
            return self.SEVERITY_COLORS['HIGH']
        elif score > 25:
            return self.SEVERITY_COLORS['MEDIUM']
        else:
            return self.SEVERITY_COLORS['LOW']
    
    def _finding_to_dict(self, finding: AssessmentResult) -> Dict[str, Any]:
        """Convert finding to dictionary for JSON export."""
        return {
            'id': str(finding.id),
            'url': finding.url,
            'type': finding.finding_type.value if finding.finding_type else None,
            'severity': finding.severity.value if finding.severity else None,
            'status': finding.status.value if finding.status else None,
            'confidence': finding.heuristic_score,
            'parameter': finding.vulnerable_parameter,
            'payload': finding.payload,
            'evidence': finding.evidence
        }
    
    def _convert_to_pdf_weasyprint(self, html_path: str) -> str:
        """Convert HTML to PDF using WeasyPrint."""
        try:
            pdf_path = html_path.replace('.html', '.pdf')
            WeasyHTML(filename=html_path).write_pdf(pdf_path)
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
        except Exception as e:
            logger.error(f"WeasyPrint PDF generation failed: {e}")
            return None
    
    async def _convert_to_pdf_playwright(self, html_path: str) -> Optional[str]:
        """Convert HTML to PDF using Playwright."""
        try:
            from playwright.async_api import async_playwright
            
            pdf_path = html_path.replace('.html', '.pdf')
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                await page.goto(f'file://{html_path}')
                await page.pdf(path=pdf_path, format='A4', print_background=True)
                
                await browser.close()
            
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Playwright PDF generation failed: {e}")
            return None
    
    def _build_markdown_report(self, findings: List[AssessmentResult],
                              metadata: ReportMetadata,
                              stats: Dict[str, Any]) -> str:
        """Build Markdown report."""
        md = f"""# {metadata.title}

**Target:** {metadata.target_scope}  
**Date:** {metadata.report_date.strftime('%Y-%m-%d')}  
**Author:** {metadata.author}  

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Findings | {stats['total_findings']} |
| Critical | {stats['by_severity'].get('CRITICAL', 0)} |
| High | {stats['by_severity'].get('HIGH', 0)} |
| Medium | {stats['by_severity'].get('MEDIUM', 0)} |
| Low | {stats['by_severity'].get('LOW', 0)} |
| Risk Score | {stats['risk_score']}% |

---

## Detailed Findings

"""
        for i, finding in enumerate(findings, 1):
            md += f"""### Finding #{i}: {finding.finding_type.value if finding.finding_type else 'Unknown'}

- **Severity:** {finding.severity.value if finding.severity else 'Unknown'}
- **URL:** {finding.url or 'N/A'}
- **Parameter:** {finding.vulnerable_parameter or 'N/A'}
- **Confidence:** {(finding.heuristic_score or 0) * 100:.0f}%

**Payload:**
```
{finding.payload or 'N/A'}
```

**Evidence:**
```
{str(finding.evidence)[:300] if finding.evidence else 'N/A'}
```

---

"""
        
        return md


# Global reporting engine instance
global_reporting_engine = ReportingEngine()


def get_reporting_engine() -> ReportingEngine:
    """Get global reporting engine instance."""
    return global_reporting_engine
