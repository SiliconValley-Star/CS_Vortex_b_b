"""
VORTEX OpenRouter Client - V17.0 ULTIMATE
OpenRouter API integration per VORTEX_AI_INTEGRATION.md

CRITICAL: All AI responses are ADVISORY ONLY
"""

import structlog
import httpx
import json
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from config.settings import get_settings
from config.prompts import (
    SYSTEM_PROMPT_CORE,
    VULNERABILITY_ASSESSMENT_PROMPT,
    BEHAVIORAL_ANALYSIS_PROMPT
)

# V21.0 - Performance Profiling Integration
try:
    from core.metrics import global_metrics
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

logger = structlog.get_logger()


class OpenRouterClient:
    """
    OpenRouter API client for AI model access
    Per VORTEX_AI_INTEGRATION.md: Multiple model support with fallbacks
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.ai.openrouter_api_key
        self.base_url = self.settings.ai.openrouter_base_url
        
        # V21.0 - Metrics integration
        self.metrics = global_metrics if METRICS_ENABLED else None
        
        # Model configuration
        self.primary_models = [
            "nousresearch/hermes-3-llama-3.1-405b",  # Uncensored primary
            "google/gemini-2.0-flash-exp:free"        # Fast validation
        ]
        
        self.fallback_models = [
            "meta-llama/llama-3.1-70b-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free"
        ]
        
        # HTTP client with retry logic
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        
        # Rate limiting and health tracking
        self.model_health = defaultdict(lambda: {
            'available': True,
            'last_success': None,
            'last_failure': None,
            'failure_count': 0,
            'total_requests': 0,
            'successful_requests': 0
        })
        
        self.rate_limits = defaultdict(lambda: {
            'requests_per_minute': 60,
            'current_count': 0,
            'window_start': datetime.utcnow()
        })
    
    async def check_model_health(self, model: str) -> bool:
        """
        Check if model is healthy and available
        Per VORTEX_AI_INTEGRATION.md: Fallback chain on unavailability
        """
        health = self.model_health[model]
        
        # If too many recent failures, mark as unhealthy
        if health['failure_count'] >= 3:
            # Check if cooldown period has passed (5 minutes)
            if health['last_failure']:
                cooldown = datetime.utcnow() - health['last_failure']
                if cooldown < timedelta(minutes=5):
                    logger.warning(
                        "Model in cooldown period",
                        model=model,
                        failure_count=health['failure_count']
                    )
                    return False
                else:
                    # Reset after cooldown
                    health['failure_count'] = 0
        
        return health['available']
    
    async def call_model(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> Dict:
        """
        Call AI model via OpenRouter API
        
        Returns: Response dict with content and metadata
        """
        if not await self.check_model_health(model):
            raise Exception(f"Model {model} is unhealthy")
        
        # Check rate limit
        if not await self._check_rate_limit(model):
            raise Exception(f"Rate limit exceeded for {model}")
        
        # Prepare request
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"} if "json" in prompt.lower() else None
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://vortex-scanner.local",
            "X-Title": "VORTEX Security Scanner"
        }
        
        # V21.0 - Track AI call metrics
        call_start = datetime.utcnow()
        tokens_used = 0
        
        try:
            logger.info(
                "Calling AI model",
                model=model,
                prompt_length=len(prompt)
            )
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content
            content = result['choices'][0]['message']['content']
            
            # V21.0 - Extract token usage
            usage = result.get('usage', {})
            tokens_used = usage.get('total_tokens', 0)
            
            # Track success
            self._record_success(model)
            
            # V21.0 - Record metrics
            if self.metrics:
                duration = (datetime.utcnow() - call_start).total_seconds()
                self.metrics.record_ai_call(
                    provider='openrouter',
                    model=model,
                    duration=duration,
                    tokens=tokens_used
                )
            
            logger.info(
                "AI model call successful",
                model=model,
                response_length=len(content),
                tokens=tokens_used
            )
            
            return {
                'success': True,
                'content': content,
                'model': model,
                'usage': result.get('usage', {}),
                'timestamp': datetime.utcnow()
            }
            
        except httpx.HTTPStatusError as e:
            self._record_failure(model)
            logger.error(
                "AI model HTTP error",
                model=model,
                status_code=e.response.status_code,
                error=str(e)
            )
            raise
            
        except Exception as e:
            self._record_failure(model)
            logger.error(
                "AI model call failed",
                model=model,
                error=str(e)
            )
            raise
    
    async def call_with_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        prefer_model: Optional[str] = None,
        max_tokens: int = 2000
    ) -> Dict:
        """
        Call AI with automatic fallback to alternative models
        Per VORTEX_AI_INTEGRATION.md: Comprehensive fallback chain
        """
        models_to_try = []
        
        # Preferred model first
        if prefer_model:
            models_to_try.append(prefer_model)
        
        # Primary models
        models_to_try.extend([m for m in self.primary_models if m not in models_to_try])
        
        # Fallback models
        models_to_try.extend(self.fallback_models)
        
        last_error = None
        
        for model in models_to_try:
            try:
                result = await self.call_model(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens
                )
                
                # Mark if this was a fallback
                result['is_fallback'] = model not in self.primary_models
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    "Model failed, trying fallback",
                    failed_model=model,
                    error=str(e)
                )
                continue
        
        # All models failed
        logger.error(
            "All AI models failed",
            tried_models=models_to_try,
            last_error=str(last_error)
        )
        
        raise Exception(f"All AI models failed: {last_error}")
    
    async def parse_json_response(
        self,
        content: str,
        allow_recovery: bool = True
    ) -> Optional[Dict]:
        """
        Parse JSON response from AI
        Per VORTEX_AI_INTEGRATION.md: Malformed recovery is non-authoritative
        """
        try:
            # Try standard JSON parsing
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON parsing failed",
                error=str(e),
                content_preview=content[:200]
            )
            
            if not allow_recovery:
                return None
            
            # Attempt recovery
            # Per VORTEX_AI_INTEGRATION.md: Recovery results are NON-AUTHORITATIVE
            return self._attempt_json_recovery(content)
    
    def _attempt_json_recovery(self, content: str) -> Optional[Dict]:
        """
        Attempt to recover parseable JSON from malformed response
        CRITICAL: Recovered results are NON-AUTHORITATIVE
        """
        import re
        
        try:
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                logger.info(
                    "JSON recovery successful (NON-AUTHORITATIVE)",
                    recovered=True
                )
                
                # Mark as recovered
                data['_recovered'] = True
                data['_authoritative'] = False
                
                return data
        except Exception:
            pass
        
        logger.error("JSON recovery failed")
        return None
    
    async def _check_rate_limit(self, model: str) -> bool:
        """Check if rate limit allows request."""
        limit_info = self.rate_limits[model]
        
        # Reset window if needed
        now = datetime.utcnow()
        if (now - limit_info['window_start']).total_seconds() >= 60:
            limit_info['current_count'] = 0
            limit_info['window_start'] = now
        
        # Check limit
        if limit_info['current_count'] >= limit_info['requests_per_minute']:
            logger.warning(
                "Rate limit exceeded",
                model=model,
                limit=limit_info['requests_per_minute']
            )
            return False
        
        limit_info['current_count'] += 1
        return True
    
    def _record_success(self, model: str) -> None:
        """Record successful model call."""
        health = self.model_health[model]
        health['last_success'] = datetime.utcnow()
        health['total_requests'] += 1
        health['successful_requests'] += 1
        health['failure_count'] = max(0, health['failure_count'] - 1)  # Decay failures
        health['available'] = True
    
    def _record_failure(self, model: str) -> None:
        """Record failed model call."""
        health = self.model_health[model]
        health['last_failure'] = datetime.utcnow()
        health['total_requests'] += 1
        health['failure_count'] += 1
        
        # Mark unavailable if too many failures
        if health['failure_count'] >= 3:
            health['available'] = False
    
    def get_model_stats(self, model: Optional[str] = None) -> Dict:
        """Get health statistics for model(s)."""
        if model:
            health = self.model_health[model]
            return {
                'model': model,
                'available': health['available'],
                'total_requests': health['total_requests'],
                'successful_requests': health['successful_requests'],
                'success_rate': health['successful_requests'] / health['total_requests'] if health['total_requests'] > 0 else 0.0,
                'failure_count': health['failure_count'],
                'last_success': health['last_success'],
                'last_failure': health['last_failure']
            }
        else:
            # All models
            return {
                model_name: self.get_model_stats(model_name)
                for model_name in self.model_health.keys()
            }
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()


# Global OpenRouter client (lazy initialization)
global_openrouter_client = None

def get_openrouter_client() -> OpenRouterClient:
    """Get or create global OpenRouter client instance."""
    global global_openrouter_client
    if global_openrouter_client is None:
        global_openrouter_client = OpenRouterClient()
    return global_openrouter_client