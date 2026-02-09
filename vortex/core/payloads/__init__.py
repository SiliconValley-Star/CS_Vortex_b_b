"""
VORTEX Payloads Package - PHASE 2.1 Enhanced
Now with 5000+ SecLists payloads
"""
from core.payloads.manager import (
    PayloadManager,
    PayloadType,
    Technology,
    get_payload_manager,
    Payload
)

# PHASE 2.1 - SecLists Integration
try:
    from core.payloads.seclists_loader import (
        SecListsLoader,
        PayloadCategory,
        get_seclists_loader
    )
    SECLISTS_AVAILABLE = True
except ImportError:
    SECLISTS_AVAILABLE = False

__all__ = [
    'PayloadManager',
    'PayloadType',
    'Technology',
    'get_payload_manager',
    'Payload',
    'SecListsLoader',
    'PayloadCategory',
    'get_seclists_loader',
]
