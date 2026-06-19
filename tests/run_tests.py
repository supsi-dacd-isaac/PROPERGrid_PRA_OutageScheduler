import os
import sys
import pytest
import argparse
from datetime import datetime

def run_tests(args):
    """Run tests with specified configuration"""
    # Base pytest arguments
    pytest_args = [
        '--verbose',
        '--tb=short',
        '--cov=optimizers',
        '--cov-report=term-missing',
        '--cov-report=html',
    ]

    # Add markers if specified
    if args.markers:
        pytest_args.extend(['-m', args.markers])

    # Add test files if specified
    if args.files:
        pytest_args.extend(args.files)
    else:
        pytest_args.append('tests')

    # Add parallel execution if requested
    if args.parallel:
        pytest_args.extend(['-n', 'auto'])

    # Add timeout if specified
    if args.timeout:
        pytest_args.extend(['--timeout', str(args.timeout)])

    # Add random seed if specified
    if args.seed:
        pytest_args.extend(['--randomly-seed', str(args.seed)])

    # Run tests
    print(f"\nRunning tests with configuration:")
    print(f"Arguments: {' '.join(pytest_args)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    result = pytest.main(pytest_args)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Exit code: {result}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return result

def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description='Run tests with various configurations')
    
    parser.add_argument('--markers', '-m', type=str,
                      help='Run tests with specific markers (e.g., "unit" or "unit and not slow")')
    
    parser.add_argument('--files', '-f', nargs='+',
                      help='Specific test files to run')
    
    parser.add_argument('--parallel', '-p', action='store_true',
                      help='Run tests in parallel')
    
    parser.add_argument('--timeout', '-t', type=int,
                      help='Timeout for each test in seconds')
    
    parser.add_argument('--seed', '-s', type=int,
                      help='Random seed for test execution')
    
    args = parser.parse_args()
    
    # Run tests
    return run_tests(args)

if __name__ == '__main__':
    sys.exit(main()) 