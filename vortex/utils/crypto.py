"""
VORTEX Cryptographic Utilities - V17.0 ULTIMATE
Cryptographic functions for evidence integrity and security

Per .clinerules:
- SHA-256 hashing for evidence chains
- Secure random generation
- Signature verification
- Data integrity validation

FEATURES:
- Cryptographic hashing (SHA-256, SHA-512)
- HMAC generation and verification
- Secure random generation
- Data signing and verification
"""

import hashlib
import hmac
import secrets
import base64
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CryptoUtils:
    """
    Cryptographic utility functions.
    
    Provides secure hashing, signing, and verification for evidence integrity.
    """
    
    @staticmethod
    def hash_sha256(data: bytes) -> str:
        """
        Compute SHA-256 hash.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex-encoded hash
        """
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def hash_sha512(data: bytes) -> str:
        """
        Compute SHA-512 hash.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex-encoded hash
        """
        return hashlib.sha512(data).hexdigest()
    
    @staticmethod
    def hash_data(data: str, algorithm: str = 'sha256') -> str:
        """
        Hash string data with specified algorithm.
        
        Args:
            data: String data to hash
            algorithm: Hash algorithm ('sha256' or 'sha512')
            
        Returns:
            Hex-encoded hash
        """
        data_bytes = data.encode('utf-8')
        
        if algorithm == 'sha256':
            return CryptoUtils.hash_sha256(data_bytes)
        elif algorithm == 'sha512':
            return CryptoUtils.hash_sha512(data_bytes)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def generate_hmac(data: bytes, key: bytes, algorithm: str = 'sha256') -> str:
        """
        Generate HMAC.
        
        Args:
            data: Data to sign
            key: Secret key
            algorithm: Hash algorithm
            
        Returns:
            Hex-encoded HMAC
        """
        if algorithm == 'sha256':
            return hmac.new(key, data, hashlib.sha256).hexdigest()
        elif algorithm == 'sha512':
            return hmac.new(key, data, hashlib.sha512).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def verify_hmac(data: bytes, key: bytes, expected_hmac: str, algorithm: str = 'sha256') -> bool:
        """
        Verify HMAC.
        
        Args:
            data: Data to verify
            key: Secret key
            expected_hmac: Expected HMAC value
            algorithm: Hash algorithm
            
        Returns:
            True if HMAC matches
        """
        computed_hmac = CryptoUtils.generate_hmac(data, key, algorithm)
        return hmac.compare_digest(computed_hmac, expected_hmac)
    
    @staticmethod
    def generate_random_bytes(length: int = 32) -> bytes:
        """
        Generate cryptographically secure random bytes.
        
        Args:
            length: Number of bytes to generate
            
        Returns:
            Random bytes
        """
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_random_hex(length: int = 32) -> str:
        """
        Generate cryptographically secure random hex string.
        
        Args:
            length: Number of bytes (hex string will be 2x this length)
            
        Returns:
            Random hex string
        """
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_random_url_safe(length: int = 32) -> str:
        """
        Generate cryptographically secure URL-safe random string.
        
        Args:
            length: Approximate string length
            
        Returns:
            URL-safe random string
        """
        return secrets.token_urlsafe(length)


class EvidenceHasher:
    """
    Evidence hashing for integrity verification.
    
    Per .clinerules evidence_integrity.py integration.
    """
    
    def __init__(self, algorithm: str = 'sha256'):
        """
        Initialize evidence hasher.
        
        Args:
            algorithm: Hash algorithm to use
        """
        self.algorithm = algorithm
        self.crypto = CryptoUtils()
    
    def hash_evidence(self, evidence_data: Dict[str, Any]) -> str:
        """
        Hash evidence data deterministically.
        
        Args:
            evidence_data: Evidence dictionary
            
        Returns:
            Evidence hash
        """
        import json
        
        # Create deterministic representation
        canonical = json.dumps(evidence_data, sort_keys=True)
        
        return self.crypto.hash_data(canonical, self.algorithm)
    
    def create_evidence_signature(self,
                                  evidence_data: Dict[str, Any],
                                  secret_key: bytes) -> str:
        """
        Create cryptographic signature for evidence.
        
        Args:
            evidence_data: Evidence dictionary
            secret_key: Secret signing key
            
        Returns:
            Evidence signature
        """
        import json
        
        # Create deterministic representation
        canonical = json.dumps(evidence_data, sort_keys=True)
        data_bytes = canonical.encode('utf-8')
        
        return self.crypto.generate_hmac(data_bytes, secret_key, self.algorithm)
    
    def verify_evidence_signature(self,
                                  evidence_data: Dict[str, Any],
                                  secret_key: bytes,
                                  signature: str) -> bool:
        """
        Verify evidence signature.
        
        Args:
            evidence_data: Evidence dictionary
            secret_key: Secret signing key
            signature: Expected signature
            
        Returns:
            True if signature is valid
        """
        import json
        
        canonical = json.dumps(evidence_data, sort_keys=True)
        data_bytes = canonical.encode('utf-8')
        
        return self.crypto.verify_hmac(data_bytes, secret_key, signature, self.algorithm)


class SecureTokenGenerator:
    """
    Secure token generation for various purposes.
    
    Used for session tokens, API keys, etc.
    """
    
    def __init__(self):
        self.crypto = CryptoUtils()
    
    def generate_session_token(self, length: int = 32) -> str:
        """Generate secure session token."""
        return self.crypto.generate_random_url_safe(length)
    
    def generate_api_key(self, prefix: str = "vortex") -> str:
        """
        Generate API key with prefix.
        
        Args:
            prefix: Key prefix for identification
            
        Returns:
            API key
        """
        random_part = self.crypto.generate_random_hex(32)
        return f"{prefix}_{random_part}"
    
    def generate_finding_id(self) -> str:
        """Generate unique finding identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_part = self.crypto.generate_random_hex(8)
        return f"finding_{timestamp}_{random_part}"
    
    def generate_chain_id(self) -> str:
        """Generate evidence chain identifier."""
        random_part = self.crypto.generate_random_hex(16)
        return f"chain_{random_part}"


class DataIntegrityChecker:
    """
    Data integrity verification using checksums.
    
    Validates data hasn't been tampered with during storage or transmission.
    """
    
    def __init__(self):
        self.crypto = CryptoUtils()
    
    def calculate_checksum(self, data: bytes, algorithm: str = 'sha256') -> str:
        """
        Calculate data checksum.
        
        Args:
            data: Data bytes
            algorithm: Hash algorithm
            
        Returns:
            Checksum
        """
        if algorithm == 'sha256':
            return self.crypto.hash_sha256(data)
        elif algorithm == 'sha512':
            return self.crypto.hash_sha512(data)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def verify_checksum(self, 
                       data: bytes, 
                       expected_checksum: str,
                       algorithm: str = 'sha256') -> bool:
        """
        Verify data checksum.
        
        Args:
            data: Data bytes
            expected_checksum: Expected checksum value
            algorithm: Hash algorithm
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.calculate_checksum(data, algorithm)
        return actual_checksum == expected_checksum
    
    def create_integrity_metadata(self, 
                                  data: bytes,
                                  algorithm: str = 'sha256') -> Dict[str, Any]:
        """
        Create integrity metadata for data.
        
        Args:
            data: Data bytes
            algorithm: Hash algorithm
            
        Returns:
            Integrity metadata
        """
        return {
            'checksum': self.calculate_checksum(data, algorithm),
            'algorithm': algorithm,
            'size': len(data),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def verify_integrity_metadata(self,
                                  data: bytes,
                                  metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Verify data against integrity metadata.
        
        Args:
            data: Data bytes
            metadata: Integrity metadata
            
        Returns:
            (is_valid, error_message)
        """
        # Size check
        if len(data) != metadata.get('size', 0):
            return False, f"Size mismatch: expected {metadata['size']}, got {len(data)}"
        
        # Checksum check
        expected_checksum = metadata.get('checksum')
        algorithm = metadata.get('algorithm', 'sha256')
        
        if not self.verify_checksum(data, expected_checksum, algorithm):
            return False, "Checksum verification failed"
        
        return True, None


# Global instances
global_crypto_utils = CryptoUtils()
global_evidence_hasher = EvidenceHasher()
global_token_generator = SecureTokenGenerator()
global_integrity_checker = DataIntegrityChecker()