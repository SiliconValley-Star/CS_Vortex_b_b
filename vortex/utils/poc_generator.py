"""
VORTEX PoC Generator - V17.0
Professional Proof of Concept generator for verified findings
"""

from typing import Dict, Any, Optional
from datetime import datetime
from domain.enums import FindingType, FindingSeverity


class PoCGenerator:
    """
    Generate professional Proof of Concept documentation and code
    for verified security findings.
    """
    
    def generate_poc(self, finding: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate complete PoC package for a finding.
        
        Args:
            finding: Finding data dictionary
            
        Returns:
            Dictionary with PoC components (markdown, curl, python, etc.)
        """
        finding_type = finding.get('type', 'UNKNOWN')
        
        generators = {
            'SQL_INJECTION': self._generate_sqli_poc,
            'XSS': self._generate_xss_poc,
            'LFI': self._generate_lfi_poc,
            'SSRF': self._generate_ssrf_poc
        }
        
        generator = generators.get(finding_type, self._generate_generic_poc)
        return generator(finding)
    
    def _generate_sqli_poc(self, finding: Dict[str, Any]) -> Dict[str, str]:
        """Generate SQL Injection PoC."""
        url = finding.get('url', '')
        param = finding.get('parameter', '')
        payload = finding.get('payload', '')
        evidence = finding.get('evidence', '')
        
        markdown = f"""# SQL Injection Vulnerability - Proof of Concept

## Overview
- **Vulnerability Type**: SQL Injection
- **Severity**: {finding.get('severity', 'N/A')}
- **Confidence**: {finding.get('confidence', 0):.1%}
- **URL**: `{url}`
- **Vulnerable Parameter**: `{param}`

## Vulnerability Description
SQL Injection vulnerability allows an attacker to inject malicious SQL code into database queries,
potentially leading to:
- Unauthorized data access
- Data manipulation/deletion
- Authentication bypass
- Remote code execution (database-dependent)

## Proof of Concept

### Vulnerable Request
```
GET {url}?{param}={payload}
```

### Evidence
```
{evidence}
```

### Reproduction Steps
1. Navigate to: `{url}`
2. Inject payload in `{param}` parameter: `{payload}`
3. Observe SQL error or anomalous database behavior
4. Verify successful injection through response differences

## Impact
- **Confidentiality**: HIGH - Database contents may be exposed
- **Integrity**: HIGH - Data may be modified or deleted
- **Availability**: MEDIUM - DoS through expensive queries

## Remediation
1. Use parameterized queries/prepared statements
2. Implement input validation and sanitization
3. Apply principle of least privilege to database accounts
4. Use ORM frameworks with built-in protections
5. Implement WAF rules for common SQL injection patterns

## References
- OWASP SQL Injection: https://owasp.org/www-community/attacks/SQL_Injection
- CWE-89: https://cwe.mitre.org/data/definitions/89.html

---
**Generated**: {datetime.utcnow().isoformat()}Z
**VORTEX Security Scanner**
"""
        
        curl = f"""# SQL Injection PoC - cURL Command
curl -X GET "{url}?{param}={payload}" \\
  -H "User-Agent: VORTEX-Security-Scanner/1.0" \\
  -v
"""
        
        python = f"""#!/usr/bin/env python3
\"\"\"
SQL Injection PoC - Python Script
Target: {url}
Parameter: {param}
\"\"\"

import requests

url = "{url}"
params = {{
    "{param}": "{payload}"
}}

response = requests.get(url, params=params)
print(f"Status: {{response.status_code}}")
print(f"Response Length: {{len(response.text)}}")
print("\\nResponse Body:")
print(response.text[:500])
"""
        
        return {
            'markdown': markdown,
            'curl': curl,
            'python': python
        }
    
    def _generate_xss_poc(self, finding: Dict[str, Any]) -> Dict[str, str]:
        """Generate XSS PoC."""
        url = finding.get('url', '')
        param = finding.get('parameter', '')
        payload = finding.get('payload', '')
        
        markdown = f"""# Cross-Site Scripting (XSS) - Proof of Concept

## Overview
- **Vulnerability Type**: Cross-Site Scripting (XSS)
- **Severity**: {finding.get('severity', 'N/A')}
- **Confidence**: {finding.get('confidence', 0):.1%}
- **URL**: `{url}`
- **Vulnerable Parameter**: `{param}`

## Vulnerability Description
XSS allows attackers to inject malicious scripts that execute in victims' browsers, enabling:
- Session hijacking
- Credential theft
- Malware distribution
- Website defacement

## Proof of Concept

### Payload
```javascript
{payload}
```

### Vulnerable URL
```
{url}?{param}={payload}
```

### Reproduction Steps
1. Navigate to: `{url}`
2. Inject payload: `{payload}` in parameter `{param}`
3. Observe JavaScript execution in browser
4. Check Developer Console for alerts/execution

## Impact
- **User Impact**: HIGH - Account compromise
- **Data Exposure**: HIGH - Session tokens, cookies
- **Reputation**: HIGH - User trust damage

## Remediation
1. Implement context-aware output encoding
2. Use Content Security Policy (CSP)
3. Apply HTTPOnly and Secure flags on cookies
4. Sanitize user input on server-side
5. Use modern frameworks with auto-escaping

## References
- OWASP XSS Guide: https://owasp.org/www-community/attacks/xss/
- CWE-79: https://cwe.mitre.org/data/definitions/79.html

---
**Generated**: {datetime.utcnow().isoformat()}Z
"""
        
        curl = f"""# XSS PoC - cURL Command
curl -X GET "{url}?{param}={payload}" \\
  -H "User-Agent: Mozilla/5.0" \\
  -v
"""
        
        return {
            'markdown': markdown,
            'curl': curl
        }
    
    def _generate_lfi_poc(self, finding: Dict[str, Any]) -> Dict[str, str]:
        """Generate LFI PoC."""
        url = finding.get('url', '')
        param = finding.get('parameter', '')
        payload = finding.get('payload', '')
        
        markdown = f"""# Local File Inclusion (LFI) - Proof of Concept

## Overview
- **Vulnerability Type**: Local File Inclusion
- **Severity**: {finding.get('severity', 'N/A')}
- **URL**: `{url}`
- **Vulnerable Parameter**: `{param}`

## Vulnerability Description
LFI allows reading arbitrary files from the server filesystem, potentially exposing:
- Configuration files
- Source code
- Sensitive credentials
- System information

## Proof of Concept

### Payload
```
{payload}
```

### Vulnerable Request
```
GET {url}?{param}={payload}
```

### Reproduction Steps
1. Send request with traversal payload
2. Observe file contents in response
3. Verify sensitive file access

## Impact
- **Confidentiality**: CRITICAL - System files exposed
- **Privilege Escalation**: Potential credential access
- **RCE Risk**: May chain with other vulns

## Remediation
1. Whitelist allowed file paths
2. Remove user input from file operations
3. Use file access APIs with path validation
4. Implement chroot/jail environments
5. Apply strict file permissions

## References
- OWASP LFI: https://owasp.org/www-community/attacks/Path_Traversal
- CWE-22: https://cwe.mitre.org/data/definitions/22.html

---
**Generated**: {datetime.utcnow().isoformat()}Z
"""
        
        curl = f"""# LFI PoC - cURL Command
curl -X GET "{url}?{param}={payload}" \\
  -v
"""
        
        return {
            'markdown': markdown,
            'curl': curl
        }
    
    def _generate_ssrf_poc(self, finding: Dict[str, Any]) -> Dict[str, str]:
        """Generate SSRF PoC."""
        url = finding.get('url', '')
        param = finding.get('parameter', '')
        payload = finding.get('payload', '')
        
        markdown = f"""# Server-Side Request Forgery (SSRF) - Proof of Concept

## Overview
- **Vulnerability Type**: Server-Side Request Forgery
- **Severity**: {finding.get('severity', 'N/A')}
- **URL**: `{url}`
- **Vulnerable Parameter**: `{param}`

## Vulnerability Description
SSRF allows attackers to make the server perform requests to arbitrary URLs, enabling:
- Internal network scanning
- Cloud metadata access
- Bypassing access controls
- Port scanning

## Proof of Concept

### Payload
```
{payload}
```

### Test Request
```
GET {url}?{param}={payload}
```

### Reproduction Steps
1. Send request with internal/metadata URL
2. Observe server-side request execution
3. Check response for internal data

## Impact
- **Internal Network Access**: Can reach internal services
- **Cloud Metadata**: May expose AWS/GCP credentials
- **Data Exfiltration**: Potential sensitive data leak

## Remediation
1. Whitelist allowed destination hosts/IPs
2. Block private IP ranges
3. Validate and sanitize URLs
4. Use network segmentation
5. Implement egress filtering

## References
- OWASP SSRF: https://owasp.org/www-community/attacks/Server_Side_Request_Forgery
- CWE-918: https://cwe.mitre.org/data/definitions/918.html

---
**Generated**: {datetime.utcnow().isoformat()}Z
"""
        
        curl = f"""# SSRF PoC - cURL Command
curl -X GET "{url}?{param}={payload}" \\
  -v
"""
        
        return {
            'markdown': markdown,
            'curl': curl
        }
    
    def _generate_generic_poc(self, finding: Dict[str, Any]) -> Dict[str, str]:
        """Generate generic PoC for unknown types."""
        url = finding.get('url', '')
        
        markdown = f"""# Security Vulnerability - Proof of Concept

## Overview
- **Vulnerability Type**: {finding.get('type', 'Unknown')}
- **Severity**: {finding.get('severity', 'N/A')}
- **URL**: `{url}`

## Evidence
```
{finding.get('evidence', 'No evidence provided')}
```

## Remediation
Please consult security best practices for this vulnerability type.

---
**Generated**: {datetime.utcnow().isoformat()}Z
"""
        
        return {
            'markdown': markdown
        }


# Global instance
poc_generator = PoCGenerator()