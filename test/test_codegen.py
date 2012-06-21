from __future__ import with_statement
import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from blip.code.codegen import *
from tester import run_tests
from tester import get_test_options, parse_test_options

# Tests
def test_alloc_release_reg():
	code = Code()

	reg = code.alloc_reg()
	assert(code.nr_regs_free() == len(code.regs) - 1)
	code.release_reg(reg)

def test_load_function():
	import os
	listdir_test = load_function('os', 'listdir')
	assert listdir_test == os.listdir

def test_load_function_from_hierarchic_module():
	listsep_test = load_function('os.path', 'sep')
	import os.path
	assert os.path.sep == listsep_test

	blipcode_test = load_function('blip.blipcode', 'gen_conv')
	import blip.blipcode
	assert blip.blipcode.gen_conv == blipcode_test

def test_get_codegen_parameters():
	def f(code, a, b, c, block_size):
		z = 3
		print a, b, c
	assert get_codegen_parameters(f) == ['code', 'a', 'b', 'c', 'block_size']
	assert get_codegen_parameters(f, True) == ['a', 'b', 'c']

def test_is_valid_codegen():
	def inv1(x, y, block_size):
		yield 3
	def inv2(code, x, y, block_size):
		pass
	def inv3(code, x, y):
		yield 3
	def v1(code, block_size, args):
		yield 3
	codegens = [(inv1, False), (inv2, False), (inv3, False), (v1, True)]
	assert all(is_valid_codegen(gen) == v for gen, v in codegens)

def test_scoped_alloc():
	''' Test basic scoped alloc functionality. '''
	code = Code()

	with scoped_alloc(code, 3) as (ra, rb, rc):
		print ra, rb, rc
		assert code.nr_regs_free() == len(code.regs) - 3
	assert code.nr_regs_free() == len(code.regs)

def test_nested_scoped_alloc():
	''' Test nested scoped alloc functionality. '''

	code = Code()
	with scoped_alloc(code, 2) as (ra, rb):
		print ra, rb
		assert(code.nr_regs_free() == len(code.regs) - 2)
		with scoped_alloc(code, 1) as (rc):
			print rc
			assert(code.nr_regs_free() == len(code.regs) - 3)
		with scoped_alloc(code, 2) as (rd, re):
			print rd, re
			assert(code.nr_regs_free() == len(code.regs) - 4)
		assert(code.nr_regs_free() == len(code.regs) - 2)
	assert(code.nr_regs_free() == len(code.regs))

def test_scoped_alloc_in_codegen():
	''' Test interaction with yield and loops. '''
	from blip.simulator.opcodes import Add, Sub, Mov, Imm
	code = Code()

	def codegen(code, block_size):
		yield Mov(code.r(3), code.r(4))
		with scoped_alloc(code, 3) as (ra, rb, rc):
			yield Add(ra, ra, rb)
			yield Sub(rc, ra, rb)
			for x in xrange(3):
				with scoped_alloc(code, 2) as (cnt, rd):
					yield Imm(cnt, x)
					yield Mov(rd, cnt)

	for instr in codegen(code, (3, 3)):
		print instr

def test_convert_code_to_compact_repr():
	''' Test conversion to compact code repr. '''
	from blip.simulator.opcodes import MemWImm, Add, Imm, Cmp, Nop
	code = Code()
	r0, r1, r2 = [code.r(i) for i in xrange(3)]
	instrs = [\
		Add(r0, r1, r2),\
		MemWImm(44, r2),\
		Imm(r2, 33),\
		Nop(cond='GT'),\
		Cmp(r1, r0),\
	]

	ref = [\
		['add', None, 'r0', 'r1', 'r2'],\
		['memw_imm', None, 44, 'r2'],\
		['imm', None, 'r2', 33],\
		['nop', 'GT'],\
		['cmp', None, 'r1', 'r0'],\
	]
	test = convert_code_to_compact_repr(instrs)
	assert test == ref

def test_convert_compact_repr_to_obj():
	''' Test conversion from compact code repr. '''
	from blip.simulator.opcodes import MemWImm, Add, Imm, Cmp, Nop
	code = Code()
	r0, r1, r2 = [code.r(i) for i in xrange(3)]
	instrs = [\
		['add', None, 'r0', 'r1', 'r2'],\
		['memw_imm', None, 44, 'r2'],\
		['imm', None, 'r2', 33],\
		['nop', 'GT'],\
		['cmp', None, 'r1', 'r0'],\
	]
	ref = [\
		Add(r0, r1, r2),\
		MemWImm(44, r2),\
		Imm(r2, 33),\
		Nop(cond='GT'),\
		Cmp(r1, r0),\
	]
	test = convert_compact_repr_to_obj(instrs)
	for t, r in zip(test, ref):
		print t, r
		assert t.opcode() == r.opcode()
		assert t.cond() == r.cond()
		for attr in ['dest', 'src', 'src1', 'src2', 'value']:
			assert getattr(t, attr, 'notavailable') == getattr(t, attr, 'notavailable') 
		# need to find a better trick for equal string reps
		#assert str(t) == str(r)

def test_instr_adapter_attrs():
	''' Test setting and retrieving of faked attributes. '''
	i1 = InstrAdapter(['add', None, 'r3', 'r2', 'r1'])
	i1.dest = 'r4'
	assert i1.dest == 'r4'
	assert i1.instr[2] == 'r4'	
	i1.dest = 'r0'
	assert i1.dest == 'r0'
	assert i1.instr[2] == 'r0'	

	i2 = InstrAdapter(['cmp', 'GT', 'r2', 'r1'])
	assert i2.opcode() == 'cmp'
	assert i2.cond() == 'GT'
	assert i2.src1 == 'r2'
	assert i2.src2 == 'r1'

	i3 = InstrAdapter(['imm', None, 'r1', 42])
	assert i3.opcode() == 'imm'
	assert i3.dest == 'r1'
	assert i3.value == 42

def all_test(options = {}):
	tests = [\
		test_alloc_release_reg,\
		test_load_function,\
		test_load_function_from_hierarchic_module,\
		test_get_codegen_parameters,\
		test_is_valid_codegen,\
		test_scoped_alloc,\
		test_nested_scoped_alloc,\
		test_scoped_alloc_in_codegen,\
		test_convert_code_to_compact_repr,\
		test_convert_compact_repr_to_obj,\
		test_instr_adapter_attrs,\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

