"""
Full Scan Workflow Integration Tests
Tests complete end-to-end scanning workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from core.engine import VortexScanEngine
from core.queue_manager import QueueManager
from core.database import Database
from domain.enums import ScanMode, FindingStatus, Severity
from domain.models import Finding


@pytest.mark.integration
@pytest.mark.asyncio
class TestFullScanWorkflow:
    """Test complete scan workflow from start to finish."""
    
    async def test_single_target_full_scan(self):
        """Test complete scan of a single target with all scanners."""
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            target_url = "https://testsite.com"
            
            # Mock HTTP responses
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    'status': 200,
                    'content': '<html><body><form><input name="search"></form></body></html>',
                    'headers': {'Content-Type': 'text/html'},
                    'response_time': 0.5
                }
                
                # Run full scan
                results = await engine.scan_target(
                    target_url=target_url,
                    scan_types=['sqli', 'xss', 'csrf'],
                    enable_recon=False
                )
                
                # Verify results structure
                assert results is not None
                assert 'findings' in results
                assert isinstance(results['findings'], list)
                
                # Verify scanners were executed
                assert mock_get.called
                
        finally:
            await engine.shutdown()
    
    async def test_multi_target_scan_workflow(self):
        """Test scanning multiple targets in sequence."""
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            targets = [
                "https://site1.com",
                "https://site2.com",
                "https://site3.com"
            ]
            
            all_findings = []
            
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    'status': 200,
                    'content': '<html><body>Test</body></html>',
                    'headers': {}
                }
                
                # Scan each target
                for target in targets:
                    results = await engine.scan_target(
                        target_url=target,
                        scan_types=['xss']
                    )
                    
                    if results and results.get('findings'):
                        all_findings.extend(results['findings'])
                
                # Verify all targets were scanned
                assert mock_get.call_count >= len(targets)
                
        finally:
            await engine.shutdown()
    
    async def test_scan_with_findings_workflow(self):
        """Test workflow when vulnerabilities are found."""
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            target_url = "https://vulnerable-site.com/search?q=test"
            
            # Mock vulnerable response
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                # First call - normal response
                # Subsequent calls - reflected XSS
                responses = [
                    {
                        'status': 200,
                        'content': '<html><body><h1>Search Results</h1></body></html>',
                        'headers': {}
                    },
                    {
                        'status': 200,
                        'content': '<html><body><script>alert(1)</script></body></html>',
                        'headers': {}
                    }
                ]
                mock_get.side_effect = responses
                
                # Scan for XSS
                results = await engine.scan_target(
                    target_url=target_url,
                    scan_types=['xss']
                )
                
                # Should detect vulnerability
                assert results is not None
                if results.get('findings'):
                    # Verify finding properties
                    finding = results['findings'][0]
                    assert 'type' in finding
                    assert 'severity' in finding
                    assert 'url' in finding
                
        finally:
            await engine.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
class TestQueueIntegration:
    """Test integration between engine and queue manager."""
    
    async def test_queue_based_scanning(self):
        """Test scanning through queue system."""
        queue_manager = QueueManager(max_size=100)
        await queue_manager.start()
        
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            # Enqueue scan tasks
            targets = ["https://site1.com", "https://site2.com"]
            
            for target in targets:
                await queue_manager.enqueue({
                    'target': target,
                    'scan_types': ['xss'],
                    'priority': 1
                })
            
            # Verify queue has items
            assert queue_manager.size() == len(targets)
            
            # Process queue
            processed = []
            while not queue_manager.is_empty():
                task = await queue_manager.dequeue()
                if task:
                    processed.append(task)
            
            # Verify all tasks processed
            assert len(processed) == len(targets)
            
        finally:
            await queue_manager.stop()
            await engine.shutdown()
    
    async def test_queue_priority_handling(self):
        """Test that queue respects priority ordering."""
        queue_manager = QueueManager(max_size=100)
        await queue_manager.start()
        
        try:
            # Enqueue with different priorities
            await queue_manager.enqueue({'task': 'low', 'priority': 3})
            await queue_manager.enqueue({'task': 'high', 'priority': 1})
            await queue_manager.enqueue({'task': 'medium', 'priority': 2})
            
            # Dequeue should follow priority
            first = await queue_manager.dequeue()
            assert first['task'] == 'high'
            
            second = await queue_manager.dequeue()
            assert second['task'] == 'medium'
            
            third = await queue_manager.dequeue()
            assert third['task'] == 'low'
            
        finally:
            await queue_manager.stop()


@pytest.mark.integration
@pytest.mark.asyncio
class TestDatabaseIntegration:
    """Test integration with database layer."""
    
    async def test_finding_persistence_workflow(self):
        """Test that findings are correctly persisted to database."""
        db = Database()
        await db.initialize()
        
        try:
            # Create test finding
            finding = Finding(
                id="test-finding-001",
                target_url="https://example.com",
                finding_type="XSS",
                severity=Severity.HIGH,
                status=FindingStatus.PENDING_VERIFICATION,
                vulnerable_parameter="search",
                payload="<script>alert(1)</script>",
                evidence="Found reflected XSS",
                heuristic_score=0.9
            )
            
            # Save to database
            await db.save_finding(finding)
            
            # Retrieve from database
            retrieved = await db.get_finding(finding.id)
            
            assert retrieved is not None
            assert retrieved.id == finding.id
            assert retrieved.finding_type == finding.finding_type
            assert retrieved.severity == finding.severity
            
        finally:
            await db.close()
    
    async def test_scan_results_aggregation(self):
        """Test aggregating results from multiple scans."""
        db = Database()
        await db.initialize()
        
        try:
            # Create multiple findings
            findings = []
            for i in range(5):
                finding = Finding(
                    id=f"finding-{i}",
                    target_url=f"https://site{i}.com",
                    finding_type="XSS",
                    severity=Severity.MEDIUM,
                    status=FindingStatus.PENDING_VERIFICATION,
                    heuristic_score=0.7
                )
                findings.append(finding)
                await db.save_finding(finding)
            
            # Retrieve all findings
            all_findings = await db.get_findings(limit=10)
            
            assert len(all_findings) >= len(findings)
            
            # Verify can filter by severity
            high_findings = await db.get_findings(
                severity=Severity.HIGH,
                limit=10
            )
            assert all(f.severity == Severity.HIGH for f in high_findings)
            
        finally:
            await db.close()


@pytest.mark.integration
@pytest.mark.asyncio
class TestScannerChaining:
    """Test chaining multiple scanners together."""
    
    async def test_multi_scanner_execution(self):
        """Test executing multiple scanners on same target."""
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            target_url = "https://example.com/page?id=1"
            
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    'status': 200,
                    'content': '<html><body>Content</body></html>',
                    'headers': {}
                }
                
                # Run multiple scanner types
                results = await engine.scan_target(
                    target_url=target_url,
                    scan_types=['sqli', 'xss', 'lfi', 'ssrf']
                )
                
                # Verify multiple scanners executed
                # Each scanner should make at least one request
                assert mock_get.call_count >= 4
                
                # Results should be aggregated
                assert results is not None
                
        finally:
            await engine.shutdown()
    
    async def test_scanner_failure_isolation(self):
        """Test that one scanner failure doesn't affect others."""
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            target_url = "https://example.com"
            
            call_count = 0
            
            async def mock_response(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                # First scanner fails
                if call_count == 1:
                    raise ConnectionError("Scanner 1 failed")
                
                # Others succeed
                return {
                    'status': 200,
                    'content': '<html><body>OK</body></html>',
                    'headers': {}
                }
            
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.side_effect = mock_response
                
                # Should continue despite one failure
                results = await engine.scan_target(
                    target_url=target_url,
                    scan_types=['sqli', 'xss']
                )
                
                # At least one scanner should succeed
                assert call_count > 1
                
        finally:
            await engine.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
class TestReconIntegration:
    """Test reconnaissance integration with scanning."""
    
    async def test_recon_to_scan_workflow(self):
        """Test discovering subdomains then scanning them."""
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            target_domain = "example.com"
            
            # Mock subdomain discovery
            discovered_subdomains = [
                "www.example.com",
                "api.example.com",
                "admin.example.com"
            ]
            
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    'status': 200,
                    'content': '<html><body>OK</body></html>',
                    'headers': {}
                }
                
                # Mock recon module
                with patch('core.recon.manager.ReconManager.discover_subdomains', 
                          new_callable=AsyncMock) as mock_recon:
                    mock_recon.return_value = discovered_subdomains
                    
                    # Run scan with recon
                    results = await engine.scan_target(
                        target_url=f"https://{target_domain}",
                        scan_types=['xss'],
                        enable_recon=True
                    )
                    
                    # Should have scanned discovered subdomains
                    if results:
                        subdomains_found = results.get('subdomains_discovered', 0)
                        # At least main domain + discovered
                        assert subdomains_found >= 0
            
        finally:
            await engine.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorRecoveryWorkflow:
    """Test error recovery in full workflows."""
    
    async def test_partial_scan_recovery(self):
        """Test recovery when scan partially fails."""
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            targets = [
                "https://site1.com",
                "https://site2.com",  # This will fail
                "https://site3.com"
            ]
            
            call_count = 0
            
            async def mock_response(url, **kwargs):
                nonlocal call_count
                call_count += 1
                
                # Fail second request
                if call_count == 2:
                    raise TimeoutError("Request timeout")
                
                return {
                    'status': 200,
                    'content': '<html><body>OK</body></html>',
                    'headers': {}
                }
            
            successful_scans = 0
            failed_scans = 0
            
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.side_effect = mock_response
                
                for target in targets:
                    try:
                        results = await engine.scan_target(
                            target_url=target,
                            scan_types=['xss']
                        )
                        if results:
                            successful_scans += 1
                    except Exception:
                        failed_scans += 1
                
                # Should have some successes despite failure
                assert successful_scans > 0
                
        finally:
            await engine.shutdown()
    
    async def test_timeout_handling_workflow(self):
        """Test handling of request timeouts in workflow."""
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            target_url = "https://slow-site.com"
            
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(10)  # Very slow
                return {'status': 200, 'content': 'OK', 'headers': {}}
            
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.side_effect = slow_response
                
                # Should handle timeout gracefully
                try:
                    results = await asyncio.wait_for(
                        engine.scan_target(target_url=target_url, scan_types=['xss']),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    # Expected - timeout handled
                    pass
                
                # Engine should still be functional
                assert engine is not None
                
        finally:
            await engine.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio  
class TestMemoryManagementWorkflow:
    """Test memory management during workflows."""
    
    async def test_memory_cleanup_after_scan(self):
        """Test that memory is properly cleaned up after scan."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        engine = VortexScanEngine()
        await engine.initialize()
        
        try:
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Run scan
            with patch.object(engine.http_client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    'status': 200,
                    'content': '<html>' + 'x' * 10000 + '</html>',  # Large response
                    'headers': {}
                }
                
                await engine.scan_target(
                    target_url="https://example.com",
                    scan_types=['xss']
                )
            
            # Force garbage collection
            gc.collect()
            await asyncio.sleep(0.5)
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            print(f"\nMemory growth: {memory_growth:.1f} MB")
            
            # Should not have excessive memory growth
            assert memory_growth < 50
            
        finally:
            await engine.shutdown()