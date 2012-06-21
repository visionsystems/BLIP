import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from tester import compare_images, run_tests, skip_test
from tester import get_test_options, parse_test_options

from planarity.reference import _gather_local_max, _elementwise_max

def test_local_max():
	''' Test local maximum calculation correctness. '''
	input_data = [
		[1, 2, 3, 4],
		[-1, 0, 2, 5],
		[4, 6, 2, 1]]
	mask = [[False, False, True], [False, False, False], [False, True, False]]
	ref_res = [
		[0, 0, 2, 5],
		[4, 6, 4, 1],
		[0, 2, 5, 0]] # zero padding of max operation
	class FakeFilter(object):
		def __init__(self, mask):
			self.mask = mask
		def size(self):
			return (len(self.mask), len(self.mask[0]))
	test_res = _gather_local_max(input_data, FakeFilter(mask))
	assert ref_res == test_res

def test_elementwise_max():
	m1 = [
		[1, 2, 3, 4],
		[-1, 0, 2, 5],
		[4, 6, 2, 1]]
	m2 = [
		[7, 3,-9, 3],
		[-4, 2, 0, 1],
		[0, 8, 2, 7]]
	test_res = _elementwise_max(m1, m2)
	ref_res = [
		[7, 3, 3, 4],
		[-1, 2, 2, 5],
		[4, 8, 2, 7]]

	assert ref_res == test_res

# ====================================================================================
# eval 
def all_test(options = {}):
	tests = [\
		test_local_max,\
		test_elementwise_max,\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

