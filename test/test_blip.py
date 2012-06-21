import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from blip.code.codegen import Code
from blip.simulator.opcodes import Imm, Add
from tester import run_tests, get_test_options, parse_test_options

# tests
def test_imm_instr():
	assert str(Imm('r0', 5, cond='GT')) == 'imm{GT}  r0, 5'

# new test 5/10/2011
def test_add_instr():
    assert str(Add('r0', 8, 5, cond='GT')) == 'add{GT}  r0, 8, 5'
# end new test

def all_test(options = {}):
	tests = [\
		test_imm_instr,
        test_add_instr,
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

