from __future__ import with_statement
import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from blip.code.trace_optimiser import *
from blip.code.codegen import scoped_alloc, Code, convert_compact_repr_to_obj, convert_code_to_compact_repr
from blip.simulator.opcodes import *

from tester import run_tests, OpMatcher, match_code, skip_test
from tester import get_test_options, parse_test_options

def print_instrs(test_code, ref_code, expect_code=None):
	pad = 20
	test_len = len(test_code)
	ref_len = len(ref_code)
	expect_len = len(expect_code) if expect_code else 0
	print '         | ' + ' | '.join(x.center(pad) for x in ['original', 'test', 'expect'])
	for x in xrange(max(test_len, ref_len, expect_len)):
		res = '%06i: '%x
		res += ' | ' + (str(ref_code[x] ) if x < ref_len  else '').rjust(pad)
		res += ' | ' + (str(test_code[x]) if x < test_len else '').rjust(pad)
		res += ' | ' + (str(expect_code[x]) if x < expect_len else '').rjust(pad)
		print res

# tests
def test_tracefragment_insert_before():
	def codegen(code):
		with scoped_alloc(code, 3) as (a, b, c):
			yield Imm(b, 3)
			yield Mov(a, b)
			yield Add(c, b, a)

	code = Code()
	trace = []
	reg_usage = []
	for instr in codegen(code):
		trace.append(instr)
		reg_usage.append(Optimiser._current_reg_usage(code))
	tf = TraceFragment(code, trace, reg_usage)
	tf.set_ptr(1)
	current_instr_str = str(tf.current_instr())
	tf.insert_instr_before(Imm(code.out, 4))
	pattern = '''
imm a, 3
Imm out, 4
mov _,_
add _,_,_
'''
	matched, err = OpMatcher(tf.trace).match(pattern)
	if not matched:
		print '  incorrect insert'
		print '  ' + err
		print'new trace:'
		print '\n'.join('  '+str(x) for x in tf.trace)
		assert False
	matched, _ = OpMatcher([tf.current_instr()]).match(current_instr_str)
	if not matched:
		print '  incorrect insert'
		print '  pointer points to incorrect instruction'
		assert False

def test_tracefragment_insert_after():
	def codegen(code):
		with scoped_alloc(code, 3) as (a, b, c):
			yield Imm(b, 3)
			yield Mov(a, b)
			yield Add(c, b, a)

	code = Code()
	trace = []
	reg_usage = []
	for instr in codegen(code):
		trace.append(instr)
		reg_usage.append(Optimiser._current_reg_usage(code))
	tf = TraceFragment(code, trace, reg_usage)
	tf.set_ptr(1)
	current_instr_str = str(tf.current_instr())
	tf.insert_instr_after(Imm(code.out, 4))
	pattern = '''
imm a, 3
mov _,_
imm out, 4
add _,_,_
'''
	matched, err = OpMatcher(tf.trace).match(pattern)
	if not matched:
		print '  incorrect insert'
		print '  ' + err
		print'new trace:'
		print '\n'.join('  '+str(x) for x in tf.trace)
		assert False
	matched, _ = OpMatcher([tf.current_instr()]).match(current_instr_str)
	if not matched:
		print '  incorrect insert'
		print '  pointer points to incorrect instruction'
		assert False

def test_tracefragment_remove_current_instr():
	def codegen(code):
		with scoped_alloc(code, 3) as (a, b, c):
			yield Imm(b, 3)
			yield Mov(a, b)
			yield Add(c, b, a)

	code = Code()
	trace = []
	reg_usage = []
	for instr in codegen(code):
		trace.append(instr)
		reg_usage.append(Optimiser._current_reg_usage(code))
	tf = TraceFragment(code, trace, reg_usage)
	tf.set_ptr(0)
	tf.remove_current_instr()
	pattern = '''
mov _,_
add _,_,_
'''
	matched, err = OpMatcher(tf.trace).match(pattern)
	if not matched:
		print '  incorrect insert'
		print '  ' + err
		print'new trace:'
		print '\n'.join('  '+str(x) for x in tf.trace)
		assert False

def test_no_optimiser_artifacts():
	''' Test if code with no optimisations passes through the optimiser unharmed. '''
	from blip.simulator.opcodes import Mov, Imm, Sub, Add, Mul

	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		for x in xrange(4):
			with scoped_alloc(code, 3) as (a, b, c):
				yield Imm(b, x*4)
				yield Mov(a, b)
				yield Imm(a, x)
				yield Mov(b, a)
				yield Add(a, a, b)
				yield Sub(c, a, b)
				yield Mul(c, c, c)
	ref_code= list(codegen(Code(), block_size, args))

	optimiser = Optimiser(5)
	test_code = list(optimiser.run(Code(), codegen, block_size, args))

	def print_instrs(test_code, ref_code):
		pad = 20
		test_len = len(test_code)
		ref_len = len(ref_code)
		for x in xrange(max(test_len, ref_len)):
			res  = (str(ref_code[x] ) if x < ref_len  else '').rjust(pad)
			res += (str(test_code[x]) if x < test_len else '').rjust(pad)
			print res
	if len(ref_code) != len(test_code):
		print_instrs(test_code, ref_code)
		assert False

	for i in xrange(len(test_code)):
		if str(test_code[i]) != str(ref_code[i]):
			print_instrs(test_code, ref_code)
			assert False

def test_imm_pass_1():
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 2) as (acc, imm_r):
			yield Xor(acc, acc, acc)
			for i in xrange(2):
				yield Imm(imm_r, 3)
				yield Add(acc, acc, imm_r)
				yield Sub(acc, acc, imm_r)
				yield Imm(imm_r, 2)
				yield Add(acc, acc, imm_r)
				yield Sub(acc, acc, imm_r)
	# expect code when trace_len == code len
	# not that if the codegen function changes,
	# the expect output should be changed as well!
	expect = '''
xor r0, r0, r0
imm r1, 3
add r0, r0, r1
sub r0, r0, r1
imm r2, 2
add r0, r0, r2
sub r0, r0, r2
add r0, r0, r1
sub r0, r0, r1
add r0, r0, r2
sub r0, r0, r2
'''

	ref_code = list(codegen(Code(), block_size, args))
	optimiser = Optimiser(len(ref_code))
	optimiser.register_pass(ImmediatePass(optimiser, 1)) # th determines result, should not be modified
	test_code = list(optimiser.run(Code(), codegen, block_size, args))

	matched, err = OpMatcher(test_code).match(expect)
	if not matched:
		print '  immediate pass doesn\'t give expected result'
		print '  err: ', err
		print_instrs(test_code, ref_code, [x.strip() for x in expect.split('\n') if x.strip()])
		assert False

def test_imm_pass_2():
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 2) as (acc, imm_r):
			yield Xor(acc, acc, acc)
			for i in xrange(3):
				yield Imm(imm_r, 2)
				yield Add(acc, acc, imm_r)
				yield Sub(acc, acc, imm_r)
				yield Imm(imm_r, i)
				yield Add(acc, acc, imm_r)
				yield Sub(acc, acc, imm_r)
	# expect code when trace_len ==  0.5 * code len
	# not that if the codegen function changes,
	# the expect output should be changed as well!
	expect = '''
xor r0, r0, r0
imm r2, 2
add r0, r0, r2
sub r0, r0, r2
imm r1, 0
add r0, r0, r1
sub r0, r0, r1
add r0, r0, r2

imm r2, 2		# resinsert because it is a new fragment
sub r0, r0, r2	# note that the new mapping 2->r2 doesn't need to be the same as previous block
imm r1, 1
add r0, r0, r1
sub r0, r0, r1
add r0, r0, r2
sub r0, r0, r2
add r0, r0, r2

imm r1, 2		# this pure insertion from previous block
sub r0, r0, r1	# restoring the original imm mapping because the  nr of imm 2 is too low
'''

	ref_code = list(codegen(Code(), block_size, args))
	optimiser = Optimiser(len(ref_code)//2)
	optimiser.register_pass(ImmediatePass(optimiser, 1)) # th determines result, should not be modified
	test_code = list(optimiser.run(Code(), codegen, block_size, args))

	matched, err = OpMatcher(test_code).match(expect)
	if not matched:
		print '  immediate pass doesn\'t give expected result'
		print '  err: ', err
		print_instrs(test_code, ref_code, [x.strip() for x in expect.split('\n') if x.strip()])
		assert False

def test_memory_pass_1():
	''' General memory pass test, corner cases in other tests '''
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 1) as acc:
			yield Xor(acc, acc, acc)
			with scoped_alloc(code, 1) as m:
				for i in xrange(3):
					yield MemRImm(m, 2)
					yield Add(acc, acc, m)
					yield Sub(acc, acc, m)
					yield MemRImm(m, i)
					yield Add(acc, acc, m)
					yield MemRImm(m, i)
					yield Sub(acc, acc, m)
				yield Add(m, m, m)
	# expect code when trace_len ==  0.5 * code len
	# not that if the codegen function changes,
	# the expect output should be changed as well!
	expect = '''
xor r0, r0, r0
memr_imm r1, 2
add r0, r0, r1
sub r0, r0, r1
memr_imm r1, 0
add r0, r0, r1
memr_imm r1, 0
sub r0, r0, r1
memr_imm r1, 2
add r0, r0, r1
sub r0, r0, r1

memr_imm r1, 1
add r0, r0, r1
memr_imm r1, 1
sub r0, r0, r1
memr_imm r1, 2
mov r2, r1
add r0, r0, r2
sub r0, r0, r2
add r0, r0, r2
sub r0, r0, r2

memr_imm r1, 2
add r1, r1, r1
'''

	ref_code = list(codegen(Code(), block_size, args))
	optimiser = Optimiser(len(ref_code)//2)
	optimiser.register_pass(MemoryPass(optimiser, 2)) # th should not be changed
	test_code = list(optimiser.run(Code(), codegen, block_size, args))

	matched, err = OpMatcher(test_code).match(expect)
	if not matched:
		print '  pass doesn\'t give expected result'
		print '  err: ', err
		print_instrs(test_code, ref_code, [x.strip() for x in expect.split('\n') if x.strip()])
		assert False

def test_memory_pass_2():
	''' Memory pass test, corner cases: write to location after caching of read '''
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 2) as (a, b):
			yield MemRImm(a, 3)
			yield MemRImm(b, 4)
			yield Add(a, a, b)
			yield MemWImm(3, a)
			yield MemRImm(a, 3)
	expect = '''
memr_imm r0, 3
memr_imm r1, 4
add r0, r0, r1
memw_imm 3, r0
memr_imm r0, 3
'''
	ref_code = list(codegen(Code(), block_size, args))
	optimiser = Optimiser(len(ref_code))
	optimiser.register_pass(MemoryPass(optimiser, 1)) # th should not be changed
	test_code = list(optimiser.run(Code(), codegen, block_size, args))

	matched, err = OpMatcher(test_code).match(expect)
	if not matched:
		print '  pass doesn\'t give expected result'
		print '  err: ', err
		print_instrs(test_code, ref_code, [x.strip() for x in expect.split('\n') if x.strip()])
		assert False

def test_memory_pass_3():
	''' Memory pass test, corner cases: write to out register should disable caching for now. '''
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 2) as (a, b):
			yield MemRImm(code.out, 3)
			yield Mov(a, code.west)
			yield MemRImm(code.out, 3)
			yield Mov(b, code.east)
	expect = '''
memr_imm out, 3
mov r0, west
memr_imm out, 3
mov r1, east
'''
	ref_code = list(codegen(Code(), block_size, args))
	optimiser = Optimiser(len(ref_code))
	optimiser.register_pass(MemoryPass(optimiser, 1)) # th should not be changed
	test_code = list(optimiser.run(Code(), codegen, block_size, args))

	matched, err = OpMatcher(test_code).match(expect)
	if not matched:
		print '  immediate pass doesn\'t give expected result'
		print '  err: ', err
		print_instrs(test_code, ref_code, [x.strip() for x in expect.split('\n') if x.strip()])
		assert False

def test_convert_to_ssa_easy():
	''' Test for the simple case without any branches. '''
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 2) as (a, b):
			yield Imm(a, 1)
			yield Imm(b, 2)
			yield Add(a, a, b)
	expect = '''
imm r0~0, 1
imm r1~0, 2
add r0~1, r0~0, r1~0
'''
	convertor = CodegenToSSAConvertor(100)
	ref_code = convert_code_to_compact_repr(list(codegen(Code(), block_size, args)))
	test_code = convert_compact_repr_to_obj(convertor._convert_code_to_ssa(ref_code)[0])
	
	assert match_code(test_code, expect)
	
def test_convert_to_ssa_ports():
	''' Test for the simple case without any branches, with port communication. '''
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 2) as (a, b):
			yield Imm(a, 1)
			yield Mov(code.out, a)
			yield Mov(b, code.east)
			yield Add(a, a, b)
	expect = '''
imm r0~0, 1
mov out, r0~0
mov r1~0, east
add r0~1, r0~0, r1~0
'''
	convertor = CodegenToSSAConvertor(100)
	ref_code = convert_code_to_compact_repr(list(codegen(Code(), block_size, args)))
	test_code = convert_compact_repr_to_obj(convertor._convert_code_to_ssa(ref_code)[0])
	assert match_code(test_code, expect)
	
def test_convert_to_ssa_mov_phi():
	''' Test ssa conversion with a simple phi case. '''
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 4) as (a, b, c, d):
			yield Imm(a, 1)
			yield Imm(b, 2)
			yield Cmp(a, b)
			yield Mov(c, a)
			yield Imm(a, 3) # to check if a is captured before new assignment
			yield Mov(c, b, cond='GT')
			yield Mov(d, c)
	expect = '''
imm r0~0, 1
imm r1~0, 2
cmp r0~0, r1~0
imm r0~1, 3
phi{GT} r2~1, r1~0, r0~0
mov r3~1, r2~1
'''
	convertor = CodegenToSSAConvertor(100)
	ref_code = convert_code_to_compact_repr(list(codegen(Code(), block_size, args)))
	test_code = convert_compact_repr_to_obj(convertor._convert_code_to_ssa(ref_code)[0])
	assert match_code(test_code, expect)

def test_convert_to_ssa_instr_phi():
	''' Test ssa conversion with a more complicated phi case. '''
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		with scoped_alloc(code, 4) as (a, b, c, d):
			yield Imm(a, 1)
			yield Imm(b, 2)
			yield Cmp(a, b)
			yield Inv(c, a, cond='LE')
			yield Imm(a, 3) # to check if a is captured before new assignment
			yield Mov(c, b, cond='GT')
			yield Mov(d, c)
	expect = '''
imm r0~0, 1
imm r1~0, 2
cmp r0~0, r1~0
inv tmp_1, r0~0
imm r0~1, 3
mov tmp_0, r1~0
phi{LE} r2~1, tmp_1, tmp_0
mov r3~1, r2~1
'''
	convertor = CodegenToSSAConvertor(100)
	ref_code = convert_code_to_compact_repr(list(codegen(Code(), block_size, args)))
	test_code = convert_compact_repr_to_obj(convertor._convert_code_to_ssa(ref_code)[0])
	assert match_code(test_code, expect)

def test_convert_to_ssa_mov_phi_long():
	''' Test ssa conversion with a simple phi case over trace fragments. '''
	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		for i in xrange(2):
			with scoped_alloc(code, 4) as (a, b, c, d):
				yield Imm(a, 1)
				yield Imm(b, 2)
				yield Cmp(a, b)
				yield Mov(c, a)
				yield Imm(a, 3) # to check if a is captured before new assignment
				yield Mov(c, b, cond='GT')
				yield Mov(d, c)
	expect = '''
imm r0~0, 1
imm r1~0, 2
cmp r0~0, r1~0
imm r0~1, 3
phi{GT} r2~1, r1~0, r0~0
mov r3~1, r2~1
imm r0~2, 1
imm r1~1, 2
cmp r0~2, r1~1
imm r0~3, 3
phi{GT} r2~2, r1~1, r0~2
mov r3~2, r2~2
'''
	block_len = 4
	convertor = CodegenToSSAConvertor(block_len)
	test_code_total = []
	for i, current_fragment in enumerate(convertor.gen_ssa_fragments(codegen, Code(), block_size, args)):
		test_code = convert_compact_repr_to_obj(current_fragment.instructions)
		for x in test_code: test_code_total.append(x)
	assert match_code(test_code_total, expect)

def test_convert_to_ssa_vj_no_errors():
	''' Test if ssa conversion is sufficient to convert VJ codegen. '''
	from violajones import gen_code
	from violajones import parse_haar
	cascade_filename = '../data/haarcascade_frontalface_alt.xml'
	
	block_size = (32, 32)
	im_size = block_size
	pe_dim = tuple(s//b for s, b in zip(im_size, block_size))

	# first load the cascade
	cascade = parse_haar.parse_haar_xml(cascade_filename)
	# use single stage to speed up test
	cascade.stages = [cascade.stages[0]]
	args = {'pe_dim': pe_dim, 'haar_classifier':cascade}
	
	block_len = 100 
	convertor = CodegenToSSAConvertor(block_len)
	test_code_total = []
	for i, current_ref_code in enumerate(convertor.gen_ssa_fragments(gen_code.gen_detect_faces_fullintegral, Code(), block_size, args)):
		test_code = convert_compact_repr_to_obj(current_ref_code)
	assert test_code # we just want no errors

@skip_test # this test should be replaced with a functional equivalence test...
def test_no_compiler_optimiser_artifacts():
	''' Test if code with no optimisations passes through the compiler optimiser unharmed. '''
	from blip import Mov, Imm, Sub, Add, Mul

	args = {}
	block_size = (4, 4)
	def codegen(code, block_size, args):
		for x in xrange(4):
			with scoped_alloc(code, 3) as (a, b, c):
				yield Imm(b, x*4)
				yield Mov(a, b)
				yield Imm(a, x)
				yield Mov(b, a)
				yield Add(a, a, b)
				yield Sub(c, a, b)
				yield Mul(c, c, c)
	ref_code= list(codegen(Code(), block_size, args))

	optimiser = CompilerOptimiser(5)
	test_code = list(optimiser.run(Code(), codegen, block_size, args))

	def print_instrs(test_code, ref_code):
		pad = 20
		test_len = len(test_code)
		ref_len = len(ref_code)
		for x in xrange(max(test_len, ref_len)):
			res  = (str(ref_code[x] ) if x < ref_len  else '').rjust(pad)
			res += (str(test_code[x]) if x < test_len else '').rjust(pad)
			print res
	if len(ref_code) != len(test_code):
		print_instrs(test_code, ref_code)
		assert False

	for i in xrange(len(test_code)):
		if str(test_code[i]) != str(ref_code[i]):
			print_instrs(test_code, ref_code)
			assert False

def test_get_allocation_ranges():
	''' Test the get allocation_ranges function. '''
	regs_used_fields = [
		[True, False, True],
		[True, True , False],
		[False,False, True],
		[True, False, True]
	]
	test_regs_used = get_allocation_ranges(regs_used_fields)
	ref_regs_used = [[[0, 1], [3, 3]], [[1, 1]], [[0, 0], [2, 3]]]
	assert test_regs_used == ref_regs_used

def all_test(options = {}):
	tests = [\
		test_tracefragment_insert_before,\
		test_tracefragment_insert_after,\
		test_tracefragment_remove_current_instr,\
		test_no_optimiser_artifacts,\
		test_imm_pass_1,\
		test_imm_pass_2,\
		test_memory_pass_1,\
		test_memory_pass_2,\
		test_memory_pass_3,\
		test_convert_to_ssa_easy,\
		test_convert_to_ssa_ports,\
		test_convert_to_ssa_mov_phi,\
		test_convert_to_ssa_instr_phi,\
		test_convert_to_ssa_mov_phi_long,\
		test_convert_to_ssa_vj_no_errors,\
		test_no_compiler_optimiser_artifacts,\
		test_get_allocation_ranges,\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

