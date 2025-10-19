#!/usr/bin/env python3
"""
Comprehensive Test Runner for Study Assistant RAG System

This script runs all comprehensive tests including:
- Integration tests for the complete pipeline
- Comprehensive scenario tests for real-world usage
- Performance and load testing
- Error handling and edge cases

Usage:
    python tests/run_comprehensive_tests.py [--verbose] [--performance] [--quick]
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite(test_file, verbose=False, capture_output=True):
    """Run a specific test suite and return results."""
    cmd = [sys.executable, "-m", "pytest", test_file, "-v" if verbose else "-q"]
    
    if not capture_output:
        cmd.append("--tb=short")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=capture_output,
            text=True,
            timeout=300  # 5 minute timeout per test suite
        )
        end_time = time.time()
        
        return {
            'success': result.returncode == 0,
            'duration': end_time - start_time,
            'stdout': result.stdout if capture_output else '',
            'stderr': result.stderr if capture_output else '',
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'duration': 300,
            'stdout': '',
            'stderr': 'Test suite timed out after 5 minutes',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'duration': 0,
            'stdout': '',
            'stderr': f'Failed to run test: {str(e)}',
            'returncode': -2
        }

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive tests for Study Assistant RAG System')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--performance', '-p', action='store_true', help='Include performance tests')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick test run (skip comprehensive scenarios)')
    parser.add_argument('--test', '-t', help='Run specific test file')
    
    args = parser.parse_args()
    
    # Define test suites in order of execution
    test_suites = [
        ('Unit Tests - PDF Processor', 'tests/test_pdf_processor.py'),
        ('Unit Tests - Database', 'tests/test_database.py'),
        ('Unit Tests - Embedding Generator', 'tests/test_embedding_generator.py'),
        ('Unit Tests - Vector Store', 'tests/test_vector_store.py'),
        ('Unit Tests - API Endpoints', 'tests/test_api_endpoints.py'),
        ('Unit Tests - Query Processing', 'tests/test_query_processing.py'),
        ('Integration Tests - End to End', 'tests/test_end_to_end_integration.py'),
    ]
    
    if not args.quick:
        test_suites.append(('Comprehensive Scenarios', 'tests/test_comprehensive_scenarios.py'))
    
    if args.performance:
        # Add performance-specific tests if they exist
        performance_tests = [
            ('Performance Tests - Load Testing', 'tests/test_performance_load.py'),
            ('Performance Tests - Memory Usage', 'tests/test_performance_memory.py'),
        ]
        # Only add if files exist
        for name, path in performance_tests:
            if os.path.exists(os.path.join(project_root, path)):
                test_suites.append((name, path))
    
    # If specific test requested, run only that
    if args.test:
        test_suites = [('Specific Test', args.test)]
    
    print("=" * 80)
    print("STUDY ASSISTANT RAG SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Running {len(test_suites)} test suite(s)")
    print(f"Verbose: {args.verbose}")
    print(f"Performance tests: {args.performance}")
    print(f"Quick mode: {args.quick}")
    print("=" * 80)
    
    results = []
    total_start_time = time.time()
    
    for suite_name, test_file in test_suites:
        print(f"\n📋 Running: {suite_name}")
        print(f"   File: {test_file}")
        print("-" * 60)
        
        # Check if test file exists
        test_path = os.path.join(project_root, test_file)
        if not os.path.exists(test_path):
            print(f"❌ Test file not found: {test_file}")
            results.append({
                'name': suite_name,
                'file': test_file,
                'success': False,
                'duration': 0,
                'error': 'Test file not found'
            })
            continue
        
        # Run the test suite
        result = run_test_suite(test_file, verbose=args.verbose, capture_output=not args.verbose)
        
        # Store results
        results.append({
            'name': suite_name,
            'file': test_file,
            'success': result['success'],
            'duration': result['duration'],
            'stdout': result['stdout'],
            'stderr': result['stderr'],
            'returncode': result['returncode']
        })
        
        # Print immediate results
        if result['success']:
            print(f"✅ PASSED in {result['duration']:.2f}s")
        else:
            print(f"❌ FAILED in {result['duration']:.2f}s")
            if result['stderr'] and not args.verbose:
                print(f"   Error: {result['stderr'][:200]}...")
        
        # Show output if verbose or if test failed
        if args.verbose or not result['success']:
            if result['stdout']:
                print("\nSTDOUT:")
                print(result['stdout'])
            if result['stderr']:
                print("\nSTDERR:")
                print(result['stderr'])
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"Total test suites: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total duration: {total_duration:.2f}s")
    
    if failed > 0:
        print("\n❌ FAILED TEST SUITES:")
        for result in results:
            if not result['success']:
                print(f"   - {result['name']} ({result['file']})")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
                elif result['stderr']:
                    print(f"     Error: {result['stderr'][:100]}...")
    
    print("\n📊 DETAILED RESULTS:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"   {status} {result['name']:<40} ({result['duration']:.2f}s)")
    
    # Performance summary if performance tests were run
    if args.performance:
        print("\n🚀 PERFORMANCE SUMMARY:")
        perf_results = [r for r in results if 'Performance' in r['name']]
        if perf_results:
            avg_duration = sum(r['duration'] for r in perf_results) / len(perf_results)
            print(f"   Average performance test duration: {avg_duration:.2f}s")
        else:
            print("   No performance tests were executed")
    
    # Exit with appropriate code
    exit_code = 0 if failed == 0 else 1
    
    print(f"\n{'🎉 ALL TESTS PASSED!' if exit_code == 0 else '💥 SOME TESTS FAILED!'}")
    print("=" * 80)
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)