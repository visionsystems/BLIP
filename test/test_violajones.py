import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from tester import run_tests
from tester import get_test_options, parse_test_options

def all_test(options = {}):
	tests = []
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

