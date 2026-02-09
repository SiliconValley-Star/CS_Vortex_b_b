"""
VORTEX Scanner Test Runner
Run all scanner tests with coverage reporting
"""

import sys
import pytest


def main():
    """Run scanner tests."""
    args = [
        'tests/test_scanners/',
        '-v',
        '--tb=short',
        '--color=yes',
        '-ra',
        '--disable-warnings'
    ]
    
    # Run with coverage if available
    try:
        import pytest_cov
        args.extend([
            '--cov=scanners',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov'
        ])
        print("Running tests with coverage...")
    except ImportError:
        print("Running tests without coverage (install pytest-cov for coverage reports)...")
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n✅ All scanner tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())