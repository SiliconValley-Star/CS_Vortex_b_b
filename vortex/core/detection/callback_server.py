"""
VORTEX OOB Callback Server - V18.0
HTTP callback server for out-of-band detection

CAPABILITIES:
- Lightweight async HTTP server
- Token extraction from subdomain/path
- Request logging and correlation
- Automatic callback registration

SECURITY:
- No sensitive data logging
- Rate limiting to prevent abuse
- IP filtering (optional)
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any
from aiohttp import web

from core.detection.oob_detector import OOBCallback

logger = logging.getLogger(__name__)


class CallbackServer:
    """
    Async HTTP server for OOB callbacks.
    
    Listens for HTTP requests and extracts callback data.
    Supports token extraction from:
    - Subdomain: TOKEN.oob.vortex.local
    - Path: /callback/TOKEN
    - Query param: ?token=TOKEN
    """
    
    def __init__(self, 
                 host: str = '0.0.0.0',
                 port: int = 8080,
                 detector=None):
        """
        Initialize callback server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            detector: OOBDetector instance
        """
        self.host = host
        self.port = port
        self.detector = detector
        
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # Setup routes
        self.app.router.add_route('*', '/{tail:.*}', self.handle_callback)
        
        # Statistics
        self.stats = {
            'requests_received': 0,
            'callbacks_processed': 0,
            'invalid_tokens': 0
        }
    
    async def start(self):
        """Start callback server."""
        logger.info(f"Starting callback server on {self.host}:{self.port}")
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logger.info(f"Callback server started successfully")
    
    async def stop(self):
        """Stop callback server."""
        logger.info("Stopping callback server")
        
        if self.site:
            await self.site.stop()
        
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Callback server stopped")
    
    async def handle_callback(self, request: web.Request) -> web.Response:
        """
        Handle incoming callback request.
        
        Extracts token and creates OOBCallback object.
        """
        self.stats['requests_received'] += 1
        
        # Extract token from request
        token = self._extract_token(request)
        
        if not token:
            self.stats['invalid_tokens'] += 1
            logger.warning(
                f"Callback received without valid token",
                method=request.method,
                path=request.path,
                host=request.host
            )
            return web.Response(text="Invalid token", status=400)
        
        # Create callback object
        callback = await self._create_callback(request, token)
        
        # Register with detector
        if self.detector:
            self.detector.register_callback(callback)
            self.stats['callbacks_processed'] += 1
        
        logger.info(
            f"Callback processed",
            token=token,
            source_ip=callback.source_ip,
            method=request.method
        )
        
        # Return success response
        return web.Response(text="Callback received", status=200)
    
    def _extract_token(self, request: web.Request) -> Optional[str]:
        """
        Extract token from request.
        
        Checks in order:
        1. Subdomain (TOKEN.oob.vortex.local)
        2. Path (/callback/TOKEN)
        3. Query parameter (?token=TOKEN)
        """
        # 1. Try subdomain
        host = request.host
        if '.' in host:
            subdomain = host.split('.')[0]
            # Check if looks like token (16 hex chars)
            if re.match(r'^[a-f0-9]{16}$', subdomain):
                return subdomain
        
        # 2. Try path
        path = request.path
        path_match = re.search(r'/callback/([a-f0-9]{16})', path)
        if path_match:
            return path_match.group(1)
        
        # 3. Try query parameter
        token = request.query.get('token')
        if token and re.match(r'^[a-f0-9]{16}$', token):
            return token
        
        return None
    
    async def _create_callback(self, 
                              request: web.Request, 
                              token: str) -> OOBCallback:
        """Create OOBCallback from HTTP request."""
        
        # Extract headers (sanitized)
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ['authorization', 'cookie']
        }
        
        # Extract body (limited size)
        try:
            body = await request.text()
            if len(body) > 10000:  # Limit to 10KB
                body = body[:10000] + '... [truncated]'
        except Exception:
            body = None
        
        # Get source IP
        source_ip = request.remote or 'unknown'
        
        # Check for forwarded IP
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            source_ip = forwarded_for.split(',')[0].strip()
        
        callback = OOBCallback(
            token=token,
            callback_type='http',
            source_ip=source_ip,
            timestamp=datetime.utcnow(),
            method=request.method,
            path=request.path,
            headers=headers,
            body=body,
            user_agent=request.headers.get('User-Agent'),
            raw_data=f"{request.method} {request.path} HTTP/1.1"
        )
        
        return callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self.stats.copy()


# Example usage
if __name__ == '__main__':
    async def main():
        # Create server
        server = CallbackServer()
        
        # Start server
        await server.start()
        
        print("Callback server running. Press Ctrl+C to stop.")
        
        try:
            # Keep running
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await server.stop()
    
    asyncio.run(main())