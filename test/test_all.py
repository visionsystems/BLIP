import os
import sys

def print_seperator():
	print '#'*50

if __name__ == '__main__':
	name = sys.argv[0]
	def is_testfile(x): return x.endswith('.py') and x.startswith('test_')
	test_files = [x for x in os.listdir('.') if is_testfile(x) and x != name]

	python_bin = sys.executable


	print_seperator()
	print 'found tests: ', '\n'.join('  ' + x for x in test_files)
	print


	# test execution
	print 'executing test files'
	failures = 0
	for x in test_files:
		cmd = '%s %s --batch'%(python_bin, x)
		print_seperator()
		print'test file ', x
		print cmd
		ret = os.system(cmd)
		if not ret == 0:
			print '    test failed'
			failures += 1
		print
		print

	nr_tests = len(test_files)
	print_seperator()
	print_seperator()
	print '%i of %i test files finished without error'%(nr_tests-failures, nr_tests)
	print
	print

