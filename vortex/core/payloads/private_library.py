"""
Private Payload Library - PHASE 2.4
User's personal payload collection management

Allows researchers to:
- Add their own custom payloads
- Organize by categories and tags
- Track success rates and metadata
- Import/export payload collections
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PayloadFormat(str, Enum):
    """Supported payload file formats"""
    JSON = "json"
    YAML = "yaml"
    TXT = "txt"


@dataclass
class PrivatePayload:
    """User's custom payload"""
    payload: str
    category: str  # xss, sqli, lfi, custom, etc.
    description: str = ""
    tags: List[str] = None
    success_rate: float = 0.0
    notes: str = ""
    source: str = ""  # Where did you find this? (URL, researcher name, etc.)
    discovered_date: str = ""
    cvss_score: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PrivatePayload':
        """Create from dictionary"""
        return cls(**data)


class PrivatePayloadLibrary:
    """
    Manages user's private payload collection
    
    Directory structure:
    vortex/payloads/private/
    ├── README.md
    ├── my_xss_payloads.json
    ├── my_sqli_payloads.json
    └── custom_techniques.yaml
    """
    
    def __init__(self, library_dir: Optional[Path] = None):
        """
        Initialize private payload library
        
        Args:
            library_dir: Path to private payloads directory
        """
        if library_dir is None:
            # Default: vortex/payloads/private/
            library_dir = Path(__file__).parent.parent.parent / "payloads" / "private"
        
        self.library_dir = Path(library_dir)
        self.payloads: List[PrivatePayload] = []
        
        # Create directory if it doesn't exist
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all payloads
        self._load_all_payloads()
    
    def _load_all_payloads(self):
        """Load all payload files from library directory"""
        if not self.library_dir.exists():
            logger.warning(f"Private payload directory not found: {self.library_dir}")
            return
        
        loaded_count = 0
        
        # Load JSON files
        for json_file in self.library_dir.glob("*.json"):
            try:
                loaded = self._load_json_file(json_file)
                loaded_count += loaded
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        # Load YAML files
        for yaml_file in self.library_dir.glob("*.yaml"):
            try:
                loaded = self._load_yaml_file(yaml_file)
                loaded_count += loaded
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
        
        # Load TXT files (simple line-by-line)
        for txt_file in self.library_dir.glob("*.txt"):
            try:
                loaded = self._load_txt_file(txt_file)
                loaded_count += loaded
            except Exception as e:
                logger.error(f"Failed to load {txt_file}: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} private payloads from {self.library_dir}")
    
    def _load_json_file(self, filepath: Path) -> int:
        """Load payloads from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        if isinstance(data, list):
            for item in data:
                try:
                    payload = PrivatePayload.from_dict(item)
                    self.payloads.append(payload)
                    count += 1
                except Exception as e:
                    logger.warning(f"Skipping invalid payload in {filepath}: {e}")
        elif isinstance(data, dict):
            # Single payload
            try:
                payload = PrivatePayload.from_dict(data)
                self.payloads.append(payload)
                count += 1
            except Exception as e:
                logger.warning(f"Invalid payload in {filepath}: {e}")
        
        return count
    
    def _load_yaml_file(self, filepath: Path) -> int:
        """Load payloads from YAML file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        count = 0
        if isinstance(data, list):
            for item in data:
                try:
                    payload = PrivatePayload.from_dict(item)
                    self.payloads.append(payload)
                    count += 1
                except Exception as e:
                    logger.warning(f"Skipping invalid payload in {filepath}: {e}")
        elif isinstance(data, dict):
            try:
                payload = PrivatePayload.from_dict(data)
                self.payloads.append(payload)
                count += 1
            except Exception as e:
                logger.warning(f"Invalid payload in {filepath}: {e}")
        
        return count
    
    def _load_txt_file(self, filepath: Path) -> int:
        """Load payloads from TXT file (line by line)"""
        count = 0
        category = filepath.stem  # Use filename as category
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    payload = PrivatePayload(
                        payload=line,
                        category=category,
                        description=f"Loaded from {filepath.name}"
                    )
                    self.payloads.append(payload)
                    count += 1
        
        return count
    
    def get_payloads(self,
                     category: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     min_success_rate: float = 0.0) -> List[PrivatePayload]:
        """
        Get payloads with filtering
        
        Args:
            category: Filter by category (xss, sqli, etc.)
            tags: Filter by tags (must have ALL specified tags)
            min_success_rate: Minimum success rate
            
        Returns:
            List of matching payloads
        """
        result = []
        
        for payload in self.payloads:
            # Category filter
            if category and payload.category.lower() != category.lower():
                continue
            
            # Tags filter (must have all specified tags)
            if tags:
                payload_tags_lower = [t.lower() for t in payload.tags]
                if not all(tag.lower() in payload_tags_lower for tag in tags):
                    continue
            
            # Success rate filter
            if payload.success_rate < min_success_rate:
                continue
            
            result.append(payload)
        
        # Sort by success rate (descending)
        result.sort(key=lambda p: p.success_rate, reverse=True)
        
        return result
    
    def get_payload_strings(self,
                           category: Optional[str] = None,
                           tags: Optional[List[str]] = None,
                           min_success_rate: float = 0.0) -> List[str]:
        """
        Get payload strings (for easy integration with PayloadManager)
        
        Args:
            category: Filter by category
            tags: Filter by tags
            min_success_rate: Minimum success rate
            
        Returns:
            List of payload strings
        """
        payloads = self.get_payloads(category, tags, min_success_rate)
        return [p.payload for p in payloads]
    
    def add_payload(self, payload: PrivatePayload, save_to_file: bool = True):
        """
        Add new payload to library
        
        Args:
            payload: Payload to add
            save_to_file: Save to file immediately
        """
        self.payloads.append(payload)
        
        if save_to_file:
            self._save_payload_to_file(payload)
    
    def _save_payload_to_file(self, payload: PrivatePayload):
        """Save payload to category file"""
        filename = f"{payload.category}_payloads.json"
        filepath = self.library_dir / filename
        
        # Load existing file or create new
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        else:
            data = []
        
        # Add new payload
        data.append(payload.to_dict())
        
        # Save
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved payload to {filepath}")
    
    def export_to_file(self,
                       filepath: Path,
                       category: Optional[str] = None,
                       format: PayloadFormat = PayloadFormat.JSON):
        """
        Export payloads to file
        
        Args:
            filepath: Output file path
            category: Export only specific category
            format: Output format (json, yaml, txt)
        """
        payloads = self.get_payloads(category=category)
        
        if format == PayloadFormat.JSON:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([p.to_dict() for p in payloads], f, indent=2, ensure_ascii=False)
        
        elif format == PayloadFormat.YAML:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump([p.to_dict() for p in payloads], f, allow_unicode=True)
        
        elif format == PayloadFormat.TXT:
            with open(filepath, 'w', encoding='utf-8') as f:
                for payload in payloads:
                    f.write(f"{payload.payload}\n")
        
        logger.info(f"Exported {len(payloads)} payloads to {filepath}")
    
    def import_from_file(self, filepath: Path):
        """
        Import payloads from file
        
        Args:
            filepath: Input file path
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            loaded = self._load_json_file(filepath)
        elif filepath.suffix in ['.yaml', '.yml']:
            loaded = self._load_yaml_file(filepath)
        elif filepath.suffix == '.txt':
            loaded = self._load_txt_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Imported {loaded} payloads from {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get library statistics"""
        stats = {
            'total_payloads': len(self.payloads),
            'by_category': {},
            'by_tags': {},
            'avg_success_rate': 0.0,
            'with_success_rate': 0
        }
        
        success_rates = []
        
        for payload in self.payloads:
            # Category stats
            cat = payload.category
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            
            # Tag stats
            for tag in payload.tags:
                stats['by_tags'][tag] = stats['by_tags'].get(tag, 0) + 1
            
            # Success rate
            if payload.success_rate > 0:
                success_rates.append(payload.success_rate)
                stats['with_success_rate'] += 1
        
        if success_rates:
            stats['avg_success_rate'] = sum(success_rates) / len(success_rates)
        
        return stats
    
    def create_example_file(self):
        """Create example payload file for users"""
        example_file = self.library_dir / "examples.json"
        
        examples = [
            {
                "payload": "<script>alert(document.domain)</script>",
                "category": "xss",
                "description": "Basic DOM XSS - shows current domain",
                "tags": ["xss", "dom", "basic"],
                "success_rate": 0.65,
                "notes": "Works on most unfiltered inputs",
                "source": "Own research",
                "discovered_date": "2024-01-15",
                "cvss_score": 6.1
            },
            {
                "payload": "' OR '1'='1' -- ",
                "category": "sqli",
                "description": "Classic SQL injection auth bypass",
                "tags": ["sqli", "auth-bypass", "classic"],
                "success_rate": 0.72,
                "notes": "Still works on legacy systems",
                "source": "OWASP",
                "discovered_date": "2020-01-01",
                "cvss_score": 9.8
            }
        ]
        
        with open(example_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created example file: {example_file}")


# Global instance
_private_library = None


def get_private_library(library_dir: Optional[Path] = None) -> PrivatePayloadLibrary:
    """Get or create global private library instance"""
    global _private_library
    if _private_library is None:
        _private_library = PrivatePayloadLibrary(library_dir)
    return _private_library