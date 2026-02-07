"""
VORTEX Utilities Module - V17.0 ULTIMATE
Helper functions for encoding, parsing, validation, formatting, and monitoring
"""

from .encoding import (
    url_encode,
    url_decode,
    base64_encode,
    base64_decode,
    html_encode,
    html_decode,
    safe_encode,
    safe_decode,
)

from .parsing import (
    safe_json_parse,
    safe_xml_parse,
    extract_urls,
    extract_parameters,
    parse_headers,
)

from .validation import (
    validate_url,
    validate_domain,
    validate_parameter,
    is_safe_payload,
    check_scope,
)

from .formatting import (
    format_finding_report,
    format_submission,
    format_evidence,
    generate_markdown_report,
    generate_html_report,
)

from .monitoring import (
    track_metric,
    log_event,
    measure_time,
    get_system_stats,
)

__all__ = [
    # Encoding
    'url_encode',
    'url_decode',
    'base64_encode',
    'base64_decode',
    'html_encode',
    'html_decode',
    'safe_encode',
    'safe_decode',
    
    # Parsing
    'safe_json_parse',
    'safe_xml_parse',
    'extract_urls',
    'extract_parameters',
    'parse_headers',
    
    # Validation
    'validate_url',
    'validate_domain',
    'validate_parameter',
    'is_safe_payload',
    'check_scope',
    
    # Formatting
    'format_finding_report',
    'format_submission',
    'format_evidence',
    'generate_markdown_report',
    'generate_html_report',
    
    # Monitoring
    'track_metric',
    'log_event',
    'measure_time',
    'get_system_stats',
]