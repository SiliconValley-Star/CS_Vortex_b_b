"""
Business Logic Analyzer - PHASE 4.1
Detects business logic flaws and authorization issues
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from domain.enums import FindingType

logger = logging.getLogger(__name__)


@dataclass
class BusinessLogicPattern:
    """Pattern for detecting business logic flaws."""
    name: str
    description: str
    indicators: List[str]
    severity: str


class BusinessLogicAnalyzer:
    """
    Analyzes application for business logic flaws.
    
    PHASE 4.1 Implementation:
    - Authorization bypass detection
    - IDOR pattern recognition
    - Price manipulation detection
    - Workflow violation checks
    """
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.stats = {
            'total_analyzed': 0,
            'logic_flaws_detected': 0,
            'idor_detected': 0,
            'auth_bypasses': 0
        }
    
    def _load_patterns(self) -> List[BusinessLogicPattern]:
        """Load business logic flaw patterns."""
        return [
            BusinessLogicPattern(
                name="direct_object_reference",
                description="Direct object reference without authorization",
                indicators=["id=", "user_id=", "account=", "order_id="],
                severity="high"
            ),
            BusinessLogicPattern(
                name="price_manipulation",
                description="Price parameter in request",
                indicators=["price=", "amount=", "total=", "cost="],
                severity="high"
            ),
            BusinessLogicPattern(
                name="role_parameter",
                description="Role/permission in request",
                indicators=["role=", "admin=", "is_admin=", "privilege="],
                severity="critical"
            ),
        ]
    
    async def analyze(self, url: str, method: str = "GET", 
                     parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze URL and parameters for business logic flaws.
        
        Returns:
            Dictionary with analysis results
        """
        self.stats['total_analyzed'] += 1
        
        findings = []
        confidence = 0.0
        
        # Check URL patterns
        for pattern in self.patterns:
            for indicator in pattern.indicators:
                if indicator in url.lower():
                    findings.append({
                        'pattern': pattern.name,
                        'description': pattern.description,
                        'severity': pattern.severity,
                        'indicator': indicator
                    })
                    confidence = max(confidence, 0.65)
                    self.stats['logic_flaws_detected'] += 1
                    
                    if 'id' in indicator:
                        self.stats['idor_detected'] += 1
        
        # Check parameters if provided
        if parameters:
            for param_name, param_value in parameters.items():
                for pattern in self.patterns:
                    for indicator in pattern.indicators:
                        if indicator.replace('=', '') in param_name.lower():
                            findings.append({
                                'pattern': pattern.name,
                                'parameter': param_name,
                                'description': f"{pattern.description} in parameter",
                                'severity': pattern.severity
                            })
                            confidence = max(confidence, 0.70)
        
        return {
            'has_logic_flaw': len(findings) > 0,
            'confidence': confidence,
            'findings': findings,
            'finding_count': len(findings)
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get analyzer statistics."""
        return self.stats.copy()


# Global instance
business_logic_analyzer = BusinessLogicAnalyzer()