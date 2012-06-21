import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from tester import run_tests
from tester import get_test_options, parse_test_options
from blip.simulator.opcodes import *
from blip.code.codegen import Code
from blip.analysis.analysis import *

def compare_cnt_dict(gt, test):
	if sorted(gt.keys()) != sorted(test.keys()):
		print 'keys don\'t match'
		print gt.keys(), test.keys()
		return False
	for k in gt.keys():
		if gt[k] != test[k]:
			print 'counts don\'t match'
			print gt, test
			return False
	return True


def test_codesection_profiler():
	code = Code()
	csp = CodeSectionProfiler()

	i1 = Mov(code.r(4), code.r(3))
	tag_instr(i1, 'blah')
	i2 = Mov(code.r(4), code.r(3))
	tag_instr(i2, 'section:test')
	i3 = Mov(code.r(4), code.r(3))
	tag_instr(i3, 'section:test')
	i4 = Mov(code.r(4), code.r(3))
	tag_instr(i4, 'section:test2')

	for i in [i1, i2, i3, i4]:
		csp.process(i)

	gt = {'test':2, 'test2':1}
	assert compare_cnt_dict(gt, csp.section_cnt) 

def test_communication_profiler():
	code = Code()
	comm = Communication()
	i1 = Mov(code.r(4), code.r(3))
	tag_instr(i1, 'communication overhead')
	i2 = Imm(code.r(4), 3)
	i3 = Mov(code.r(4), code.r(3))
	tag_instr(i3, 'communication overhead')
	i4 = Mul(code.r(2), code.r(4), code.r(3))

	for i in [i1, i2, i3, i4]:
		comm.process(i)
	assert comm.nr_instr == 4
	assert comm.overhead == 2	

def test_opcode_profiler():
	code = Code()
	opc = OpcodeFreq()
	i1 = Mov(code.r(4), code.r(3))
	i2 = Imm(code.r(4), 3)
	i3 = Mov(code.r(4), code.r(3))
	i4 = Mul(code.r(2), code.r(4), code.r(3))

	for i in [i1, i2, i3, i4]:
		opc.process(i)
	gt = {'mov':2, 'imm':1,'mul':1}
	assert compare_cnt_dict(gt, opc.freq) 

def all_test(options = {}):
	tests = [\
		test_codesection_profiler,\
		test_communication_profiler,\
		test_opcode_profiler,\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

