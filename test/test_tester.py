from __future__ import with_statement
import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from tester import *


# tests
def test_matcher_1():
	from blip.simulator.opcodes import Mov, Add, Imm
	pattern = '''
mov _,a
...
add a, 3, 5
'''
	test = [Mov('2', '7'), Imm('4', 3), Add('7', '3', '5')]
	matched, err = OpMatcher(test).match(pattern)
	if not matched:
		print 'match error: ', err
		assert False

def test_matcher_2():
	from blip.simulator.opcodes import Mov, Add, Imm
	from blip.code.codegen import scoped_alloc, Code
	def codegen(code):
		with scoped_alloc(code, 3) as (a, b, c):
			yield Imm(a, 3.)
			yield Imm(b, 4.)
			yield Mov(a, b)
			yield Add(c, b, a)
	ops = list(x for x in codegen(Code()))
	# put a Imm opcode to check opcode case invariance
	pattern = '''
Imm a, 3
imm b, 4
mov _,_
add _,_,_
'''
	matched, err = OpMatcher(ops).match(pattern)
	if not matched:
		print 'match error: ', err
		assert False

def test_matcher_3():
	from blip.simulator.opcodes import Mov, Add, Imm
	from blip.code.codegen import scoped_alloc, Code
	def codegen(code):
		yield Imm(code.r(0), 3., cond='GT')
	ops = list(x for x in codegen(Code()))

	pattern = '''
imm {GT} r0, 3
'''
	matched, err = OpMatcher(ops).match(pattern)
	if not matched:
		print 'match error: ', err
		assert False

def test_op_parser():
	from blip.simulator.opcodes import Mov, Add, Imm
	ops = [Mov(1, 2), Add(4, 5, 2, cond='GT'), Imm(5, 4)]
	str_ops = '\n'.join(str(x) for x in ops)
	parsed_ops = OpMatcher.parse_ops(str_ops)

	ref_ops = [('mov', ['1', '2'], None), ('add', ['4', '5', '2'], 'GT'), ('imm', ['5', '4'], None)]
	assert parsed_ops == ref_ops

def test_op_parser_2():
	str_ops = 'imm {GT} 6, 3'
	parsed_ops = OpMatcher.parse_ops(str_ops)
	ref_ops = [('imm', ['6', '3'], 'GT')]
	assert parsed_ops == ref_ops

def all_test(options = {}):
	tests = [\
		test_op_parser,\
		test_op_parser_2,\
		test_matcher_1,\
		test_matcher_2,\
		test_matcher_3\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

