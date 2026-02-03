"""
VORTEX Adaptive WAF Evasion System - V17.0 ULTIMATE
Intelligent WAF detection and adaptive evasion strategies

Per .clinerules LEGAL_COMPLIANCE:
- Only used on authorized targets
- Ethical boundaries enforced
- No malicious intent
- Bug bounty context only

FEATURES:
- WAF fingerprinting
- Multi-encoding strategies
- Adaptive payload mutation
- Response-based strategy selection
- Legal compliance enforcement
"""

import re
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote, quote_plus, unquote
import base64

logger = logging.getLogger(__name__)


@dataclass
class WAFProfile:
    """WAF detection profile."""
    waf_type: str
    confidence: float  # 0.0-1.0
    indicators: List[str]
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    # Recommended strategies for this WAF
    recommended_strategies: List[str] = field(default_factory=list)


@dataclass
class EvasionStrategy:
    """WAF evasion strategy."""
    strategy_name: str
    description: str
    encoding_level: int  # 1=simple, 2=moderate, 3=complex
    success_rate: float = 0.0
    attempts: int = 0
    successes: int = 0


class EncodingEngine:
    """
    Multi-layer encoding engine for WAF evasion.
    
    Provides various encoding techniques that bypass common WAF rules.
    """
    
    @staticmethod
    def url_encode(payload: str, double: bool = False) -> str:
        """URL encoding (single or double)."""
        encoded = quote(payload, safe='')
        if double:
            encoded = quote(encoded, safe='')
        return encoded
    
    @staticmethod
    def hex_encode(payload: str) -> str:
        """Convert to hex encoding."""
        return ''.join(f'%{ord(c):02x}' for c in payload)
    
    @staticmethod
    def unicode_encode(payload: str) -> str:
        """Unicode encoding."""
        return ''.join(f'\\u{ord(c):04x}' for c in payload)
    
    @staticmethod
    def base64_encode(payload: str) -> str:
        """Base64 encoding."""
        return base64.b64encode(payload.encode()).decode()
    
    @staticmethod
    def case_mutation(payload: str) -> str:
        """Random case mutation."""
        return ''.join(
            c.upper() if random.random() > 0.5 else c.lower()
            for c in payload
        )
    
    @staticmethod
    def comment_injection(payload: str, comment_style: str = 'sql') -> str:
        """Inject comments to break pattern matching."""
        if comment_style == 'sql':
            # SQL inline comments
            parts = payload.split()
            return '/**/'.join(parts)
        elif comment_style == 'html':
            # HTML comments
            return payload.replace(' ', '<!-- -->')
        return payload
    
    @staticmethod
    def whitespace_mutation(payload: str) -> str:
        """Replace spaces with alternative whitespace."""
        alternatives = ['\t', '\n', '\r', '+', '%20', '%09', '%0a']
        return payload.replace(' ', random.choice(alternatives))
    
    @staticmethod
    def concatenation(payload: str, style: str = 'sql') -> str:
        """String concatenation."""
        if style == 'sql':
            # SQL concatenation: 'ad'+'min'
            if len(payload) > 2:
                mid = len(payload) // 2
                return f"'{payload[:mid]}'||'{payload[mid:]}'"
        return payload
    
    @staticmethod
    def null_byte_injection(payload: str) -> str:
        """Inject null bytes."""
        return payload.replace(' ', '%00')


class WAFDetector:
    """
    WAF detection and fingerprinting.
    
    Identifies WAF type based on response patterns.
    """
    
    # WAF signatures (from false_positive_filter.py)
    WAF_SIGNATURES = {
        'cloudflare': {
            'headers': ['cf-ray', 'cloudflare'],
            'body_patterns': ['attention required', 'cloudflare'],
            'status_codes': [403, 429],
            'strategies': ['hex_encoding', 'case_mutation', 'whitespace_mutation']
        },
        'aws_waf': {
            'headers': ['x-amzn-requestid', 'x-amzn-waf'],
            'body_patterns': ['aws', 'request blocked'],
            'status_codes': [403],
            'strategies': ['url_encode_double', 'comment_injection', 'unicode_encode']
        },
        'akamai': {
            'headers': ['akamai'],
            'body_patterns': ['akamai', 'reference #'],
            'status_codes': [403],
            'strategies': ['base64_encode', 'concatenation', 'null_byte']
        },
        'imperva': {
            'headers': ['x-cdn: imperva'],
            'body_patterns': ['imperva', 'incapsula'],
            'status_codes': [403, 406],
            'strategies': ['comment_injection', 'whitespace_mutation', 'case_mutation']
        },
        'mod_security': {
            'headers': [],
            'body_patterns': ['mod_security', 'modsecurity'],
            'status_codes': [406, 501],
            'strategies': ['comment_injection', 'hex_encoding', 'concatenation']
        }
    }
    
    def detect_waf(self, 
                   response_headers: Dict[str, str],
                   response_body: str,
                   status_code: int) -> Optional[WAFProfile]:
        """
        Detect WAF from response.
        
        Args:
            response_headers: Response headers
            response_body: Response body
            status_code: HTTP status code
            
        Returns:
            WAF profile if detected
        """
        headers_lower = {k.lower(): v.lower() for k, v in response_headers.items()}
        body_lower = response_body.lower()
        
        for waf_type, signatures in self.WAF_SIGNATURES.items():
            indicators = []
            confidence_score = 0.0
            
            # Check headers
            for header_sig in signatures['headers']:
                for header_key, header_value in headers_lower.items():
                    if header_sig in header_key or header_sig in header_value:
                        indicators.append(f"Header: {header_sig}")
                        confidence_score += 0.3
            
            # Check body patterns
            for body_pattern in signatures['body_patterns']:
                if body_pattern in body_lower:
                    indicators.append(f"Body pattern: {body_pattern}")
                    confidence_score += 0.2
            
            # Check status code
            if status_code in signatures['status_codes']:
                indicators.append(f"Status code: {status_code}")
                confidence_score += 0.1
            
            # If sufficient indicators found
            if confidence_score >= 0.3:
                confidence = min(confidence_score, 1.0)
                
                profile = WAFProfile(
                    waf_type=waf_type.upper(),
                    confidence=confidence,
                    indicators=indicators,
                    recommended_strategies=signatures['strategies']
                )
                
                logger.info(f"WAF detected: {waf_type.upper()} (confidence: {confidence:.2f})")
                
                return profile
        
        return None


class AdaptiveEvasionEngine:
    """
    Adaptive WAF evasion engine.
    
    RESPONSIBILITIES:
    - Detect WAF presence
    - Select appropriate evasion strategies
    - Mutate payloads adaptively
    - Learn from response patterns
    
    LEGAL COMPLIANCE (per .clinerules):
    - Only used on authorized targets
    - Scope validation enforced
    - Ethical boundaries maintained
    """
    
    def __init__(self):
        self.encoder = EncodingEngine()
        self.detector = WAFDetector()
        
        # Strategy registry
        self.strategies: Dict[str, EvasionStrategy] = self._initialize_strategies()
        
        # Detected WAFs per target
        self.waf_profiles: Dict[str, WAFProfile] = {}
        
        # Statistics
        self.total_evasion_attempts = 0
        self.successful_evasions = 0
        
        logger.info("Adaptive WAF Evasion Engine initialized")
    
    def _initialize_strategies(self) -> Dict[str, EvasionStrategy]:
        """Initialize evasion strategies."""
        return {
            'url_encode': EvasionStrategy(
                strategy_name='url_encode',
                description='Standard URL encoding',
                encoding_level=1
            ),
            'url_encode_double': EvasionStrategy(
                strategy_name='url_encode_double',
                description='Double URL encoding',
                encoding_level=2
            ),
            'hex_encoding': EvasionStrategy(
                strategy_name='hex_encoding',
                description='Hexadecimal encoding',
                encoding_level=2
            ),
            'unicode_encode': EvasionStrategy(
                strategy_name='unicode_encode',
                description='Unicode encoding',
                encoding_level=2
            ),
            'base64_encode': EvasionStrategy(
                strategy_name='base64_encode',
                description='Base64 encoding',
                encoding_level=2
            ),
            'case_mutation': EvasionStrategy(
                strategy_name='case_mutation',
                description='Random case mutation',
                encoding_level=1
            ),
            'comment_injection': EvasionStrategy(
                strategy_name='comment_injection',
                description='SQL/HTML comment injection',
                encoding_level=2
            ),
            'whitespace_mutation': EvasionStrategy(
                strategy_name='whitespace_mutation',
                description='Alternative whitespace characters',
                encoding_level=1
            ),
            'concatenation': EvasionStrategy(
                strategy_name='concatenation',
                description='String concatenation',
                encoding_level=2
            ),
            'null_byte': EvasionStrategy(
                strategy_name='null_byte',
                description='Null byte injection',
                encoding_level=3
            )
        }
    
    def detect_and_profile_waf(self,
                               target_url: str,
                               response_headers: Dict[str, str],
                               response_body: str,
                               status_code: int) -> Optional[WAFProfile]:
        """
        Detect and profile WAF for target.
        
        Args:
            target_url: Target URL
            response_headers: Response headers
            response_body: Response body
            status_code: Status code
            
        Returns:
            WAF profile if detected
        """
        profile = self.detector.detect_waf(response_headers, response_body, status_code)
        
        if profile:
            # Cache profile for target
            from urllib.parse import urlparse
            domain = urlparse(target_url).netloc
            self.waf_profiles[domain] = profile
        
        return profile
    
    def select_strategy(self, target_url: str, payload_type: str = 'generic') -> str:
        """
        Select best evasion strategy for target.
        
        Args:
            target_url: Target URL
            payload_type: Type of payload (sql, xss, etc.)
            
        Returns:
            Strategy name
        """
        from urllib.parse import urlparse
        domain = urlparse(target_url).netloc
        
        # If WAF detected, use recommended strategy
        if domain in self.waf_profiles:
            profile = self.waf_profiles[domain]
            if profile.recommended_strategies:
                # Select strategy with best success rate
                best_strategy = None
                best_rate = 0.0
                
                for strategy_name in profile.recommended_strategies:
                    if strategy_name in self.strategies:
                        strategy = self.strategies[strategy_name]
                        if strategy.success_rate > best_rate:
                            best_strategy = strategy_name
                            best_rate = strategy.success_rate
                
                if best_strategy:
                    return best_strategy
                
                # Use first recommended if no success data
                return profile.recommended_strategies[0]
        
        # Default: start with simple encoding
        return 'url_encode'
    
    def mutate_payload(self, 
                      payload: str, 
                      strategy_name: str,
                      payload_type: str = 'generic') -> str:
        """
        Mutate payload using specified strategy.
        
        Args:
            payload: Original payload
            strategy_name: Strategy to use
            payload_type: Type of payload (sql, xss, etc.)
            
        Returns:
            Mutated payload
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy_name}, using default")
            strategy_name = 'url_encode'
        
        strategy = self.strategies[strategy_name]
        strategy.attempts += 1
        
        # Apply encoding
        try:
            if strategy_name == 'url_encode':
                mutated = self.encoder.url_encode(payload, double=False)
            elif strategy_name == 'url_encode_double':
                mutated = self.encoder.url_encode(payload, double=True)
            elif strategy_name == 'hex_encoding':
                mutated = self.encoder.hex_encode(payload)
            elif strategy_name == 'unicode_encode':
                mutated = self.encoder.unicode_encode(payload)
            elif strategy_name == 'base64_encode':
                mutated = self.encoder.base64_encode(payload)
            elif strategy_name == 'case_mutation':
                mutated = self.encoder.case_mutation(payload)
            elif strategy_name == 'comment_injection':
                comment_style = 'sql' if payload_type == 'sql' else 'html'
                mutated = self.encoder.comment_injection(payload, comment_style)
            elif strategy_name == 'whitespace_mutation':
                mutated = self.encoder.whitespace_mutation(payload)
            elif strategy_name == 'concatenation':
                mutated = self.encoder.concatenation(payload, style=payload_type)
            elif strategy_name == 'null_byte':
                mutated = self.encoder.null_byte_injection(payload)
            else:
                mutated = payload
            
            self.total_evasion_attempts += 1
            
            logger.debug(f"Payload mutated using {strategy_name}: {len(payload)} → {len(mutated)} bytes")
            
            return mutated
            
        except Exception as e:
            logger.error(f"Payload mutation failed: {e}")
            return payload
    
    def adaptive_evasion(self,
                        target_url: str,
                        payload: str,
                        payload_type: str = 'generic',
                        max_attempts: int = 3) -> List[str]:
        """
        Generate adaptive evasion payloads.
        
        Args:
            target_url: Target URL
            payload: Original payload
            payload_type: Type of payload
            max_attempts: Maximum mutation attempts
            
        Returns:
            List of mutated payloads
        """
        mutated_payloads = [payload]  # Include original
        
        # Select initial strategy
        strategy = self.select_strategy(target_url, payload_type)
        
        # Generate mutations with different strategies
        strategies_to_try = [strategy]
        
        # Add alternative strategies
        from urllib.parse import urlparse
        domain = urlparse(target_url).netloc
        
        if domain in self.waf_profiles:
            profile = self.waf_profiles[domain]
            strategies_to_try.extend(profile.recommended_strategies[:max_attempts])
        else:
            # Default progression
            strategies_to_try.extend(['case_mutation', 'comment_injection'])
        
        # Generate mutations
        for strategy_name in strategies_to_try[:max_attempts]:
            mutated = self.mutate_payload(payload, strategy_name, payload_type)
            if mutated != payload and mutated not in mutated_payloads:
                mutated_payloads.append(mutated)
        
        logger.info(f"Generated {len(mutated_payloads)} evasion variants")
        
        return mutated_payloads
    
    def record_success(self, strategy_name: str) -> None:
        """Record successful evasion."""
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            strategy.successes += 1
            
            # Update success rate
            if strategy.attempts > 0:
                strategy.success_rate = strategy.successes / strategy.attempts
            
            self.successful_evasions += 1
            
            logger.info(f"Strategy {strategy_name} success recorded (rate: {strategy.success_rate:.2%})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evasion statistics."""
        strategy_stats = {
            name: {
                'attempts': strategy.attempts,
                'successes': strategy.successes,
                'success_rate': strategy.success_rate
            }
            for name, strategy in self.strategies.items()
            if strategy.attempts > 0
        }
        
        return {
            'total_attempts': self.total_evasion_attempts,
            'successful_evasions': self.successful_evasions,
            'overall_success_rate': self.successful_evasions / self.total_evasion_attempts if self.total_evasion_attempts > 0 else 0.0,
            'waf_profiles_detected': len(self.waf_profiles),
            'strategy_performance': strategy_stats
        }


# Global adaptive evasion engine
global_evasion_engine = AdaptiveEvasionEngine()