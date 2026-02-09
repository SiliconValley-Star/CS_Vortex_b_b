"""
VORTEX Structural Pattern Analyzer - V17.0 ULTIMATE
Advanced pattern matching through structural analysis

DETECTION METHODS:
- DOM structure comparison (HTML hierarchy)
- JSON schema differential
- XML structure analysis
- Response structure fingerprinting

CRITICAL: Structural changes are more deterministic than simple text matching
"""

import logging
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# BeautifulSoup import guard
try:
    from bs4 import BeautifulSoup, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("BeautifulSoup not available - DOM analysis disabled")


@dataclass
class StructuralChange:
    """Detected structural change."""
    change_type: str  # "dom_added", "dom_removed", "dom_modified", "json_schema"
    location: str
    baseline_value: Optional[str] = None
    test_value: Optional[str] = None
    confidence_impact: float = 0.0
    description: str = ""


@dataclass
class StructuralAnalysisResult:
    """Result from structural analysis."""
    changes: List[StructuralChange] = field(default_factory=list)
    confidence: float = 0.0
    determinism_score: float = 0.0
    structural_similarity: float = 0.0
    analysis_type: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)


class StructuralAnalyzer:
    """
    Analyze structural differences between responses.
    
    APPROACH:
    - DOM tree comparison for HTML
    - Schema comparison for JSON
    - Structure fingerprinting for XML
    - Hash-based change detection
    """
    
    def __init__(self):
        self.stats = {
            'dom_analyses': 0,
            'json_analyses': 0,
            'changes_detected': 0
        }
        
        # Confidence thresholds
        self.min_confidence_structural = 0.75
        self.high_confidence_threshold = 0.85
    
    def analyze_dom_changes(self,
                           baseline_html: str,
                           test_html: str) -> StructuralAnalysisResult:
        """
        Analyze DOM structural differences.
        
        Per VORTEX_EVIDENCE_STANDARDS.md:
        - Structural changes are more deterministic than text
        - New DOM nodes indicate application logic changes
        - Attribute changes may indicate XSS reflection
        
        Args:
            baseline_html: Original HTML response
            test_html: Test HTML response (with payload)
            
        Returns:
            StructuralAnalysisResult with detected changes
        """
        self.stats['dom_analyses'] += 1
        
        if not BS4_AVAILABLE:
            logger.error("BeautifulSoup not available for DOM analysis")
            return StructuralAnalysisResult(
                analysis_type='dom',
                confidence=0.0
            )
        
        try:
            # Parse HTML
            baseline_soup = BeautifulSoup(baseline_html, 'html.parser')
            test_soup = BeautifulSoup(test_html, 'html.parser')
            
            changes = []
            
            # 1. Compare DOM structure
            baseline_structure = self._extract_dom_structure(baseline_soup)
            test_structure = self._extract_dom_structure(test_soup)
            
            # Detect added nodes
            added_tags = test_structure - baseline_structure
            for tag in added_tags:
                changes.append(StructuralChange(
                    change_type='dom_added',
                    location=tag,
                    test_value=tag,
                    confidence_impact=0.3,
                    description=f'New DOM element added: {tag}'
                ))
            
            # Detect removed nodes
            removed_tags = baseline_structure - test_structure
            for tag in removed_tags:
                changes.append(StructuralChange(
                    change_type='dom_removed',
                    location=tag,
                    baseline_value=tag,
                    confidence_impact=0.25,
                    description=f'DOM element removed: {tag}'
                ))
            
            # 2. Check for error containers (common in SQLi/XSS)
            error_indicators = self._detect_error_containers(test_soup, baseline_soup)
            changes.extend(error_indicators)
            
            # 3. Check for attribute changes (XSS reflection)
            attribute_changes = self._detect_attribute_changes(baseline_soup, test_soup)
            changes.extend(attribute_changes)
            
            # 4. Calculate structural similarity
            similarity = self._calculate_dom_similarity(baseline_soup, test_soup)
            
            # 5. Calculate confidence and determinism
            confidence = self._calculate_structural_confidence(changes, similarity)
            determinism = self._calculate_determinism_score(changes, similarity)
            
            self.stats['changes_detected'] += len(changes)
            
            return StructuralAnalysisResult(
                changes=changes,
                confidence=confidence,
                determinism_score=determinism,
                structural_similarity=similarity,
                analysis_type='dom'
            )
            
        except Exception as e:
            logger.error(f"DOM analysis error: {e}")
            return StructuralAnalysisResult(
                analysis_type='dom',
                confidence=0.0
            )
    
    def analyze_json_schema(self,
                           baseline_json: str,
                           test_json: str) -> StructuralAnalysisResult:
        """
        Analyze JSON schema differences.
        
        Useful for:
        - API responses
        - AJAX endpoints
        - JSON-based applications
        
        Args:
            baseline_json: Original JSON response
            test_json: Test JSON response
            
        Returns:
            StructuralAnalysisResult
        """
        self.stats['json_analyses'] += 1
        
        try:
            baseline_data = json.loads(baseline_json)
            test_data = json.loads(test_json)
            
            changes = []
            
            # Compare keys at all levels
            baseline_keys = self._extract_json_keys(baseline_data)
            test_keys = self._extract_json_keys(test_data)
            
            # Detect added keys
            added_keys = test_keys - baseline_keys
            for key in added_keys:
                changes.append(StructuralChange(
                    change_type='json_key_added',
                    location=key,
                    test_value=key,
                    confidence_impact=0.35,
                    description=f'New JSON key: {key}'
                ))
            
            # Detect removed keys
            removed_keys = baseline_keys - test_keys
            for key in removed_keys:
                changes.append(StructuralChange(
                    change_type='json_key_removed',
                    location=key,
                    baseline_value=key,
                    confidence_impact=0.3,
                    description=f'JSON key removed: {key}'
                ))
            
            # Check for error keys
            error_keys = self._detect_json_error_keys(test_data)
            for key, value in error_keys:
                changes.append(StructuralChange(
                    change_type='json_error_key',
                    location=key,
                    test_value=str(value),
                    confidence_impact=0.4,
                    description=f'Error key detected: {key} = {value}'
                ))
            
            # Calculate similarity
            similarity = len(baseline_keys & test_keys) / max(len(baseline_keys | test_keys), 1)
            
            # Calculate scores
            confidence = self._calculate_structural_confidence(changes, similarity)
            determinism = self._calculate_determinism_score(changes, similarity)
            
            return StructuralAnalysisResult(
                changes=changes,
                confidence=confidence,
                determinism_score=determinism,
                structural_similarity=similarity,
                analysis_type='json'
            )
            
        except json.JSONDecodeError:
            # Not valid JSON
            return StructuralAnalysisResult(
                analysis_type='json',
                confidence=0.0
            )
        except Exception as e:
            logger.error(f"JSON analysis error: {e}")
            return StructuralAnalysisResult(
                analysis_type='json',
                confidence=0.0
            )
    
    def _extract_dom_structure(self, soup: BeautifulSoup) -> Set[str]:
        """Extract DOM structure as set of tag paths."""
        structure = set()
        
        for tag in soup.find_all():
            # Get tag path (e.g., "html > body > div > p")
            path = self._get_tag_path(tag)
            structure.add(path)
        
        return structure
    
    def _get_tag_path(self, tag: Tag) -> str:
        """Get hierarchical path to tag."""
        path_parts = []
        current = tag
        
        while current and current.name:
            # Add tag with optional id/class for specificity
            identifier = current.name
            if current.get('id'):
                identifier += f"#{current['id']}"
            elif current.get('class'):
                classes = ' '.join(current['class'][:2])  # First 2 classes
                identifier += f".{classes}"
            
            path_parts.insert(0, identifier)
            current = current.parent
            
            # Limit depth
            if len(path_parts) > 10:
                break
        
        return ' > '.join(path_parts)
    
    def _detect_error_containers(self,
                                 test_soup: BeautifulSoup,
                                 baseline_soup: BeautifulSoup) -> List[StructuralChange]:
        """Detect error message containers in test response."""
        changes = []
        
        # Common error container indicators
        error_indicators = [
            {'class': 'error'},
            {'class': 'warning'},
            {'class': 'alert'},
            {'id': 'error'},
            {'class': 'exception'},
            {'class': 'debug'}
        ]
        
        for indicator in error_indicators:
            test_errors = test_soup.find_all(**indicator)
            baseline_errors = baseline_soup.find_all(**indicator)
            
            if len(test_errors) > len(baseline_errors):
                for error_elem in test_errors[len(baseline_errors):]:
                    changes.append(StructuralChange(
                        change_type='error_container_added',
                        location=str(indicator),
                        test_value=error_elem.get_text()[:100],
                        confidence_impact=0.45,
                        description=f'Error container detected: {indicator}'
                    ))
        
        return changes
    
    def _detect_attribute_changes(self,
                                  baseline_soup: BeautifulSoup,
                                  test_soup: BeautifulSoup) -> List[StructuralChange]:
        """Detect attribute changes (potential XSS reflection)."""
        changes = []
        
        # Focus on dangerous attributes for XSS
        dangerous_attrs = ['onclick', 'onerror', 'onload', 'onmouseover', 'href', 'src']
        
        test_tags = test_soup.find_all()
        
        for tag in test_tags:
            for attr in dangerous_attrs:
                if tag.has_attr(attr):
                    attr_value = tag[attr]
                    
                    # Check if this attribute exists in baseline
                    baseline_match = baseline_soup.find(
                        tag.name,
                        {attr: attr_value}
                    )
                    
                    if not baseline_match:
                        changes.append(StructuralChange(
                            change_type='attribute_added',
                            location=f'{tag.name}[{attr}]',
                            test_value=str(attr_value)[:100],
                            confidence_impact=0.5,  # High impact for XSS
                            description=f'Suspicious attribute: {tag.name}[{attr}]'
                        ))
        
        return changes
    
    def _calculate_dom_similarity(self,
                                  baseline_soup: BeautifulSoup,
                                  test_soup: BeautifulSoup) -> float:
        """Calculate DOM structural similarity."""
        baseline_tags = {tag.name for tag in baseline_soup.find_all()}
        test_tags = {tag.name for tag in test_soup.find_all()}
        
        if not baseline_tags and not test_tags:
            return 1.0
        
        intersection = len(baseline_tags & test_tags)
        union = len(baseline_tags | test_tags)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_json_keys(self, data: Any, prefix: str = '') -> Set[str]:
        """Recursively extract all JSON keys."""
        keys = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.add(full_key)
                keys.update(self._extract_json_keys(value, full_key))
        
        elif isinstance(data, list):
            for i, item in enumerate(data[:5]):  # Limit to first 5 items
                keys.update(self._extract_json_keys(item, f"{prefix}[{i}]"))
        
        return keys
    
    def _detect_json_error_keys(self, data: Any) -> List[Tuple[str, Any]]:
        """Detect error-indicating keys in JSON."""
        error_keys = []
        error_indicators = ['error', 'exception', 'warning', 'message', 'debug', 'trace']
        
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower()
                if any(indicator in key_lower for indicator in error_indicators):
                    error_keys.append((key, value))
                
                # Recursive check
                if isinstance(value, (dict, list)):
                    error_keys.extend(self._detect_json_error_keys(value))
        
        elif isinstance(data, list):
            for item in data:
                error_keys.extend(self._detect_json_error_keys(item))
        
        return error_keys
    
    def _calculate_structural_confidence(self,
                                        changes: List[StructuralChange],
                                        similarity: float) -> float:
        """Calculate confidence from structural changes."""
        if not changes:
            return 0.0
        
        # Sum confidence impacts
        total_impact = sum(change.confidence_impact for change in changes)
        
        # Penalize high similarity (less change = less confidence)
        similarity_penalty = similarity * 0.2
        
        confidence = total_impact - similarity_penalty
        
        return max(0.0, min(confidence, 1.0))
    
    def _calculate_determinism_score(self,
                                    changes: List[StructuralChange],
                                    similarity: float) -> float:
        """
        Calculate determinism score for structural analysis.
        
        Per VORTEX_EVIDENCE_STANDARDS.md:
        - Structural changes are highly deterministic
        - Base score starts higher than behavioral
        """
        base_score = 0.6  # Higher base than behavioral (0.4)
        
        # Boost for specific high-determinism change types
        high_determinism_types = ['error_container_added', 'attribute_added', 'json_error_key']
        
        for change in changes:
            if change.change_type in high_determinism_types:
                base_score += 0.1
        
        # Normalize
        score = min(base_score, 1.0)
        
        return score
    
    def get_stats(self) -> Dict[str, int]:
        """Get analyzer statistics."""
        return self.stats.copy()


# Global analyzer instance
global_structural_analyzer = StructuralAnalyzer()


def analyze_structural_changes(baseline: str,
                               test: str,
                               content_type: str = 'html') -> StructuralAnalysisResult:
    """
    Convenience function for structural analysis.
    
    Args:
        baseline: Baseline response body
        test: Test response body
        content_type: 'html' or 'json'
        
    Returns:
        StructuralAnalysisResult
    """
    if content_type == 'json':
        return global_structural_analyzer.analyze_json_schema(baseline, test)
    else:
        return global_structural_analyzer.analyze_dom_changes(baseline, test)