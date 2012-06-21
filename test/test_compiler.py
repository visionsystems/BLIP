import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from tester import run_tests, OpMatcher, match_code
from tester import get_test_options, parse_test_options
from blip.code.BlipCompiler import Compiler, KernelObject, CompilerDriver, BodyVisitor, UndefinedVariableException
from blip.code.codegen import InstrAdapter

def test_undefined_variable_detection():
	''' Test if an undefined variable gives the correct error. '''
	compiler = Compiler()
	src = '''
@kernel
def main(a):
	b = 3
	a += 1
	c = a + d
'''
	try:
		main_object = compiler.compile(src)[0]
	except UndefinedVariableException, e:
		print str(e)
		return # correct execution
	assert False

def test_resolve_seq_values():
	''' Check if sequencer values are correctly resolved. '''
	from blip.code.BlipCompiler import NamedValue, ConstValue, EmitValue
	name1, name2 = [NamedValue('name%i'%i) for i in xrange(2)]
	v1, v2 = [ConstValue(i) for i in xrange(2)]
	v3, v4 = [ConstValue(i, seq_value = True) for i in xrange(2)]
	# case 1: noseq, noseq -> noseq, noseq
	bv1 = BodyVisitor(Compiler(), KernelObject('test'))
	bv1.add_value(name1, v1)
	bv1.add_value(name2, v2)
	rv1, rv2 = bv1.resolve_seq_values([v1, v2])
	assert rv1.seq_value == False and rv2.seq_value == False

	# case 2: noseq, seq = noseq, xemit noseq
	bv2 = BodyVisitor(Compiler(), KernelObject('test'))
	bv2.add_value(name1, v1)
	bv2.add_value(name2, v3)
	rv1, rv3 = bv2.resolve_seq_values([v1, v3])
	assert rv1.seq_value == False and rv3.seq_value == False
	assert isinstance(bv2.values[-1][1], EmitValue)
	assert bv2.values[-1][1].value == v3

	# case 3: seq, noseq = xemit noseq, noseq
	bv3 = BodyVisitor(Compiler(), KernelObject('test'))
	bv3.add_value(name1, v3)
	bv3.add_value(name2, v1)
	rv3, rv1 = bv3.resolve_seq_values([v3, v1])
	assert rv3.seq_value == False and rv1.seq_value == False
	assert isinstance(bv3.values[-1][1], EmitValue)
	assert bv2.values[-1][1].value == v3

	# case 4: seq, seq = seq, seq
	bv4 = BodyVisitor(Compiler(), KernelObject('test'))
	bv4.add_value(name1, v3)
	bv4.add_value(name2, v4)
	rv3, rv4 = bv4.resolve_seq_values([v3, v4])
	assert rv3.seq_value == True and rv4.seq_value == True

def test_compile_add():
	compiler = Compiler()
	src = '''
@kernel
def main():
	b = 4
	a = b + 2
'''
	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm  b@0,4
imm  tmp_0@0,2
add  a@0,b@0,tmp_0@0
'''.strip()
	assert match_code(code, pattern)


def test_compile_conditional():
	compiler = Compiler()
	src = '''
@kernel
def main(a):
	b = 3 if a > 0 else 1
	return b
'''
	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm  tmp_0@0,0
imm  tmp_2@0,3
imm  tmp_3@0,1
cmp  a@0,tmp_0@0
phi {GT}  b@0,tmp_2@0,tmp_3@0
mov  main___return@0,b@0
'''.strip()

	assert match_code(code, pattern)

def test_simple_loop_noseq():
	compiler = Compiler(no_sequencer=True)
	src = '''
@kernel
def main():
	b = 5
	for i in range(2):
		b = b + 1
'''
	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm  b@0,5
imm  i@0,0
imm  tmp_0@0,1
add  b@1,b@0,tmp_0@0
imm  i@1,1
imm  tmp_1@0,1
add  b@2, b@1, tmp_1@0
'''.strip()
	assert match_code(code, pattern)

def test_simple_loop():
	compiler = Compiler()
	src = '''
@kernel
def main():
	b = 5
	for i in range(2):
		b = b + 1
'''
	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm  b@0, 5
ximm  tmp_0@0, 2
xlabel  loop_intro_1
ximm  i@0, 0
ximm  inc_1@0, 1
xlabel  for_2
ximm  tmp_4@0, 0
xemit  tmp_6@0, i@1
imm  tmp_7@0, 0
xcmp  i@1, tmp_4@0
xphi {EQ}  tmp_9@0, i@0, i@1
cmp  tmp_6@0, tmp_7@0
phi {EQ}  tmp_10@0, b@0, b@1
imm  tmp_1@0, 1
add  b@1, tmp_10@0, tmp_1@0
xadd  i@1, tmp_9@0, inc_1@0
xcmp  i@1, tmp_0@0
xjmp {LT}  for_2
'''.strip()

	assert match_code(code, pattern)

def test_simple_loop_emit_noseq():
	compiler = Compiler(no_sequencer=True)
	src = '''
@kernel
def main():
	b = 5
	for i in range(2):
		b = i
'''
	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm  b@0,5
imm  i@0,0
mov  b@1,i@0
imm  i@1,1
mov  b@2,i@1
'''.strip()

	assert match_code(code, pattern)

def test_simple_loop_emit():
	compiler = Compiler()
	src = '''
@kernel
def main():
	b = 5
	for i in range(2):
		b = i
'''
	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm  b@0, 5
ximm tmp_0@0, 2
xlabel  loop_intro_1
ximm  i@0, 0
ximm  inc_1@0, 1
xlabel  for_2
ximm  tmp_3@0, 0
xemit  tmp_5@0, i@1
imm  tmp_6@0, 0
xcmp  i@1, tmp_3@0
xphi {EQ}  tmp_8@0, i@0, i@1
xmov  b@1, tmp_8@0
xadd  i@1, tmp_8@0, inc_1@0
xcmp  i@1, tmp_0@0
xjmp {LT}  for_2
'''.strip()

	assert match_code(code, pattern)

def test_inplace_operator():
	compiler = Compiler()
	src = '''
@kernel
def main():
	b = 5
	b += 1
'''
	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm  b@0,5
imm  tmp_0@0,1
add  b@1,b@0,tmp_0@0
'''.strip()
	assert match_code(code, pattern)

def test_replace_phi_nodes():
	compiler = Compiler()
	src = '''
@kernel
def main(a):
	b = 3 if a > 0 else 1
'''

	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	main_object = compiler.replace_phi_nodes(main_object)
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm  tmp_0@0,0
imm  tmp_2@0,3
imm  tmp_3@0,1
cmp  a@0,tmp_0@0
mov  b@0,tmp_3@0
mov {GT}  b@0,tmp_2@0
'''.strip()
	assert match_code(code, pattern)
	
def test_builtin_loadwest():
	compiler = Compiler()
	src = '''
@kernel
def main():
	b = loadWest()
'''

	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	main_object = compiler.replace_phi_nodes(main_object)
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
mov  b@0, west
'''.strip()
	assert match_code(code, pattern)

def test_builtin_sendout():
	compiler = Compiler()
	src = '''
@kernel
def main():
	a = 3
	sendOut(a)
'''

	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	main_object = compiler.replace_phi_nodes(main_object)
	code = [InstrAdapter(x) for x in main_object.code]
	
	pattern = '''
imm a@0, 3
mov out, a@0
'''.strip()
	assert match_code(code, pattern)

def test_builtin_sendout():
	compiler = Compiler()
	src = '''
@kernel
def main():
	a = 3
	sendOut(a)
'''

	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	main_object = compiler.replace_phi_nodes(main_object)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm a@0, 3
mov out, a@0
'''.strip()
	assert match_code(code, pattern)

def test_builtin_transfer():
	''' Test the transfer builtin functions. '''
	compiler = Compiler()
	src = '''
@kernel
def main():
	a = 3
	b = transferFromNorth(a)
	c = transferFromEast(b)
	d = transferFromSouth(c)
	e = transferFromWest(d)
'''

	kernel_objects= compiler.compile(src)
	main_object = kernel_objects[0]
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm a@0, 3
mov out, a@0
mov b@0, north
mov out, b@0
mov c@0, east
mov out, c@0
mov d@0, south
mov out, d@0
mov d@0, west
'''.strip()
	assert match_code(code, pattern)


def test_th_code():
	src = '''
@kernel
def main(in_ptr, out_ptr, th):
	y = get_local_id(0)
	x = get_local_id(1)
	index = 16*y + x
	in_v = in_ptr[index]
	out_v = 255 if in_v > th else 0
	out_ptr[index] = out_v
'''

	comp = CompilerDriver(8)
	main_object = comp.run(src)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  r2,0
lid  r4,r2
imm  r2,1
lid  r5,r2
imm  r2,16
mul  r6,r2,r4
add  r2,r6,r5
add  r4,r3,r2
memr  r3,r4
imm  r4,255
imm  r5,0
cmp  r3,r0
mov  r0,r5
mov {GT}  r0,r4
add  r3,r1,r2
memw  r3,r0
'''.strip()
	assert match_code(code, pattern)

def test_conv_code_noseq():
	''' Test a simple 3x1 convolution. '''
	src = '''
@kernel
def main(in_ptr, out_ptr, th):
	coeff = [-1, 0, 1]
	y = get_local_id(0)
	x = get_local_id(1)
	index = 16*y
	acc = 0
	current_x = x - 1
	for i in range(3):
		in_v = in_ptr[index+current_x]
		acc += (coeff[i] * in_v if current_x < 16 else 0) if current_x >= 0 else 0
		current_x += 1
	out_ptr[index+x] = acc
'''

	comp = CompilerDriver(16, no_sequencer=True)
	main_object = comp.run(src)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  r0,0
lid  r3,r0
imm  r0,1
lid  r4,r0
imm  r0,16
mul  r5,r0,r3
imm  r0,0
imm  r3,1
sub  r6,r4,r3
imm  r3,0
add  r7,r5,r6
add  r8,r2,r7
memr r7,r8
imm  r8,0
imm  r9,16
imm  r10,-1
mul  r11,r10,r7
imm  r7,0
cmp  r6,r9
mov  r9,r7
mov {LT}  r9,r11
imm  r7,0
cmp  r6,r8
mov  r8,r7
mov {GE}  r8,r9
add  r7,r0,r8
imm  r0,1
add  r8,r6,r0
imm  r0,1
add  r6,r5,r8
add  r9,r2,r6
memr  r6,r9
imm  r9,0
imm  r10,16
imm  r11,0
mul  r12,r11,r6
imm  r6,0
cmp  r8,r10
mov  r10,r6
mov {LT}  r10,r12
imm  r6,0
cmp  r8,r9
mov  r9,r6
mov {GE}  r9,r10
add  r6,r7,r9
imm  r7,1
add  r9,r8,r7
imm  r7,2
add  r8,r5,r9
add  r10,r2,r8
memr  r2,r10
imm  r8,0
imm  r10,16
imm  r11,1
mul  r12,r11,r2
imm  r2,0
cmp  r9,r10
mov  r10,r2
mov {LT}  r10,r12
imm  r2,0
cmp  r9,r8
mov  r8,r2
mov {GE}  r8,r10
add  r2,r6,r8
imm  r6,1
add  r8,r9,r6
add  r6,r5,r4
add  r4,r1,r6
memw  r4,r2
'''.strip()
	assert match_code(code, pattern)


def test_const_array_access():
	src = '''
@kernel
def main():
	r = [1, 2, 3]
	b = r[1]
'''
	comp = CompilerDriver(8)
	main_object = comp.run(src)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm r0,1
imm r1,2
'''.strip()
	assert match_code(code, pattern)

def test_delayed_memderef():
	src = '''
@kernel
def main(q):
	b = 3
	a = q[b]
	b = 5
	c = a
'''
	comp = CompilerDriver(8)
	main_object = comp.run(src)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  r0,3
add  r2,r1,r0
memr  r0,r2
imm  r1,5
'''.strip()
	assert match_code(code, pattern)

def test_delayed_memderef2():
	src = '''
@kernel
def main(q):
	b = 3
	a = q[b]
	q += 1
	c = a
'''
	comp = CompilerDriver(8)
	main_object = comp.run(src)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  r0,3
add  r2,r1,r0
memr  r0,r2
imm  r2,1
add  r3,r1,r2
'''.strip()
	assert match_code(code, pattern)

def test_memderef_inside_condassign():
	src = '''
@kernel
def main():
	ind = 3
	z = 1
	b = z[4] if z > 0 else 0
'''
	comp = CompilerDriver(8)
	main_object = comp.run(src)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  r0,3
imm  r1,1
imm  r2,0
imm  r3,4
memr_imm  r4,5
imm  r5,0
cmp  r1,r2
mov  r1,r5
mov {GT}  r1,r4
'''.strip()
	assert match_code(code, pattern)

def test_logic_variable():
	src = '''
@kernel
def main(b):
	a = b > 0
	c = 3 if a else 1
'''
	comp = CompilerDriver(8)
	main_object = comp.run(src)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  r1,0
imm  r2,3
imm  r3,1
cmp  r0,r1
mov  r0,r3
mov {GT}  r0,r2
'''.strip()
	assert match_code(code, pattern)

def test_logic_variable2():
	''' Test if a delayed condition captures the correct version of the tested variable. '''
	src = '''
@kernel
def main(b):
	b_gt = b > 0
	d = 4 if b_gt else b
	b += 1
	c = 3 if b_gt else 1
'''
	comp = CompilerDriver(8)
	main_object = comp.run(src)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  r1,0
imm  r2,4
cmp  r0,r1
mov  r3,r0
mov {GT}  r3,r2
imm  r2,1
add  r4,r0,r2
imm  r2,3
imm  r5,1
cmp  r0,r1
mov  r0,r5
mov {GT}  r0,r2
'''.strip()
	assert match_code(code, pattern)

def test_phi_codegen():
	from blip.code.BlipCompiler import NamedValue, PhiValue, Codegenerator, Comparison
	kernelObject = KernelObject('test')
	kernelObject.values = [\
		(NamedValue('cmp'), Comparison(NamedValue('x'), 'Gt', NamedValue('y'))),
		(NamedValue('z'), PhiValue(NamedValue('cmp'), NamedValue('x'), NamedValue('y')))
	]
	codegen = Codegenerator(Compiler())
	kernelObject = codegen.gen_code(kernelObject)
	code = [InstrAdapter(x) for x in kernelObject.code]

	pattern = '''
cmp x, y
phi {GT} z, x, y
'''
	assert match_code(code, pattern)

def test_jmp_codegen():
	from blip.code.BlipCompiler import NamedValue, JmpValue, Codegenerator, Comparison
	kernelObject = KernelObject('test')
	kernelObject.values = [\
		(NamedValue('cmp', seq_value=True), Comparison(NamedValue('x', seq_value=True), 'Gt', NamedValue('y', seq_value=True))),
		(NamedValue('tmp', seq_value=True), JmpValue(NamedValue('cmp'), NamedValue('for_3')))
	]
	codegen = Codegenerator(Compiler())
	kernelObject = codegen.gen_code(kernelObject)
	code = [InstrAdapter(x) for x in kernelObject.code]

	pattern = '''
xcmp x, y
xjmp {GT} for_3
'''
	assert match_code(code, pattern)

def test_patch_arguments():
	from blip.code.BlipCompiler import CompilerDriver, Compiler
	src = '''
@kernel
def main(q):
	return q + 1
'''
	comp = CompilerDriver(8)
	main_object = comp.run(src)
	patched_object = Compiler.patch_arguments_before_run(main_object, [41])
	code = [InstrAdapter(x) for x in patched_object.code]

	pattern = '''
imm  r0,41
imm  r1,1
add  r2,r0,r1
mov  _, r2
'''.strip()
	assert match_code(code, pattern)

def test_patch_reg_arguments():
	''' Patching of arguments before run, now with registers as arguments. '''
	from blip.code.BlipCompiler import Compiler, NamedValue

	src = '''
@kernel
def main(p, q):
	b = p - 2
	return q + b
'''
	compiler = Compiler()
	main_object = compiler.compile(src)[0]
	patched_object = Compiler.patch_arguments_before_run(main_object, [NamedValue('test_value@0'), 41])
	code = [InstrAdapter(x) for x in patched_object.code]

	pattern = '''
mov  p@0,test_value@0
imm  q@0,41
imm  tmp_0@0,2
sub  b@0,p@0,tmp_0@0
add  tmp_1@0,q@0,b@0
mov  main___return@0,tmp_1@0
'''.strip()
	assert match_code(code, pattern)

def test_2D_const_list():
	''' Test code generation with 2D constants list.

	Note that this looks easy to optimise, but it is not
	so simple, because q is only know right before execution. '''
	from blip.code.BlipCompiler import CompilerDriver, Compiler
	src = '''
@kernel
def main(q):
	a = [[1, 2, 3], [4, 5, 6]]
	acc = 0
	for i in range(2):
		for j in range(3):
			acc += q*a[i][j]
	return acc
'''
	comp = CompilerDriver(16, no_sequencer=True)
	main_object = comp.run(src)
	patched_object = Compiler.patch_arguments_before_run(main_object, [41])
	code = [InstrAdapter(x) for x in patched_object.code]

	pattern = '''
imm  r0,41
imm  r1,0
imm  r2,0
imm  r3,0
imm  r4,1
mul  r5,r0,r4
add  r4,r1,r5
imm  r1,1
imm  r5,2
mul  r6,r0,r5
add  r5,r4,r6
imm  r4,2
imm  r6,3
mul  r7,r0,r6
add  r6,r5,r7
imm  r5,1
imm  r7,0
imm  r8,4
mul  r9,r0,r8
add  r8,r6,r9
imm  r6,1
imm  r9,5
mul  r10,r0,r9
add  r9,r8,r10
imm  r8,2
imm  r10,6
mul  r11,r0,r10
add  r0,r9,r11
mov  r9,r0
'''.strip()
	assert match_code(code, pattern)

def test_peephole_addorsub_zero():
	''' Test if an addition/subtraction with a zero operand is converted into a mov instr. '''
	src = '''
@kernel
def main(a):
	b = a | 0
	c = a + 0
	d = a - 0
	e = 0 - a # unoptimised case
	f = 4
	g = 3
	h = f + g
	i = f - g
'''

	compiler = Compiler()
	main_object = compiler.compile(src)[0]
	main_object = compiler.opt_peephole(main_object)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  tmp_0@0,0
mov  b@0,a@0
imm  tmp_1@0,0
mov  c@0,a@0
imm  tmp_2@0,0
mov  d@0,a@0
imm  tmp_3@0,0
sub  e@0,tmp_3@0,a@0
imm  f@0,4
imm  g@0,3
imm  h@0,7
imm  i@0,1
'''.strip()
	assert match_code(code, pattern)

def test_peephole_twoconstant_sources():
	src = '''
@kernel
def main():
	a = 1
	b = 12
	c = a + b
	d = b - a
	e = a | b
	a = -2
	b = 3
	f = b & a
	g = a * b
'''

	compiler = Compiler()
	main_object = compiler.compile(src)[0]
	main_object = compiler.opt_peephole(main_object)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  a@0,1
imm  b@0,12
imm  c@0,13
imm  d@0,11
imm  e@0,13
imm  a@1,-2
imm  b@1,3
imm  f@0,2
imm  g@0,-6
'''.strip()
	assert match_code(code, pattern)

get2D_simple_src = '''
@kernel
def main():
	in_ptr = 0
	bwidth = 8
	bheight = 8
	y = get_local_id(0)
	x = get_local_id(1)
	v = get2D(in_ptr, x, y, bwidth, bheight)
'''
def test_get2D():
	''' Test the get2D builtin, this code fetches a value from a 2D buffer with boundary handling. '''
	compiler = Compiler()
	main_object = compiler.compile(get2D_simple_src)[0]
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
# initiate parameters
imm  in_ptr@0, 0
imm  bwidth@0, 8
imm  bheight@0, 8
imm  tmp_0@0, 0
lid  y@0, tmp_0@0
imm  tmp_1@0, 1
lid  x@0, tmp_1@0

# copy parameters
mov  cg_tmp_18@0, in_ptr@0
mov  cg_tmp_4@0, x@0
mov  cg_tmp_10@0, y@0
mov  cg_tmp_5@0, bwidth@0
mov  cg_tmp_11@0, bheight@0

# start of get2D
imm  cg_tmp_1@0, 0
imm  cg_tmp_2@0, 0
add  cg_tmp_3@0, cg_tmp_4@0, cg_tmp_5@0          # cg_tmp_3 = x + bwidth
cmp  cg_tmp_4@0, cg_tmp_1@0			 # comp x, 0
phi {LT}  cg_tmp_6@0, cg_tmp_3@0, cg_tmp_4@0     # cg_tmp_6 = phi(x < 0, cg_tmp_3, x)
sub  cg_tmp_7@0, cg_tmp_6@0, cg_tmp_5@0          # cg_tmp_7 = cg_tmp_6 - bwidth
cmp  cg_tmp_4@0, cg_tmp_5@0                      # comp x, bwidth
phi {GE}  cg_tmp_8@0, cg_tmp_7@0, cg_tmp_6@0     # cg_tmp_4 = phi(x >= bwidth, cg_tmp_7, cg_tmp_6)
add  cg_tmp_9@0, cg_tmp_10@0, cg_tmp_11@0        # cg_tmp_9 = y + bheight
cmp  cg_tmp_10@0, cg_tmp_2@0
phi {LT}  cg_tmp_12@0, cg_tmp_9@0, cg_tmp_10@0
sub  cg_tmp_13@0, cg_tmp_12@0, cg_tmp_11@0
cmp  cg_tmp_10@0, cg_tmp_11@0
phi {GE}  cg_tmp_14@0, cg_tmp_13@0, cg_tmp_12@0
mul  cg_tmp_15@0, cg_tmp_14@0, cg_tmp_5@0
add  cg_tmp_16@0, cg_tmp_15@0, cg_tmp_8@0
add  cg_tmp_17@0, cg_tmp_18@0, cg_tmp_16@0
memr  cg_tmp_19@0, cg_tmp_17@0
mov  out, cg_tmp_19@0
mov  cg_tmp_20@0, west
cmp  cg_tmp_4@0, cg_tmp_1@0
phi {LT}  cg_tmp_21@0, cg_tmp_20@0, cg_tmp_19@0
mov  out, cg_tmp_21@0
mov  cg_tmp_22@0, east
cmp  cg_tmp_4@0, cg_tmp_5@0
phi {GE}  cg_tmp_23@0, cg_tmp_22@0, cg_tmp_21@0
mov  out, cg_tmp_23@0
mov  cg_tmp_24@0, north
cmp  cg_tmp_10@0, cg_tmp_2@0
phi {LT}  cg_tmp_25@0, cg_tmp_24@0, cg_tmp_23@0
mov  out, cg_tmp_25@0
mov  cg_tmp_26@0, south
cmp  cg_tmp_10@0, cg_tmp_11@0
phi {GE}  cg_tmp_27@0, cg_tmp_26@0, cg_tmp_25@0
mov  cg_tmp_28@0, cg_tmp_27@0
mov  v@0, cg_tmp_28@0
'''.strip()
	assert match_code(code, pattern)

def test_get2D_copy_propagation():
	''' Test the get2D builtin, this code fetches a value from a 2D buffer with boundary handling.

	The difference with the previous test is that it also runs the copy propagation optimiser.
	'''


	compiler = Compiler()
	main_object = compiler.compile(get2D_simple_src)[0]
	print '\n'.join(str(InstrAdapter(x)) for x in main_object.code)
	main_object = compiler.opt_copy_propagation(main_object)
	code = [InstrAdapter(x) for x in main_object.code]

	pattern = '''
imm  in_ptr@0, 0
imm  bwidth@0, 8
imm  bheight@0, 8
imm  tmp_0@0, 0
lid  y@0, tmp_0@0
imm  tmp_1@0, 1
lid  x@0, tmp_1@0
imm  cg_tmp_1@0, 0
imm  cg_tmp_2@0, 0
add  cg_tmp_3@0, x@0, bwidth@0
cmp  x@0, cg_tmp_1@0
phi {LT}  cg_tmp_6@0, cg_tmp_3@0, x@0
sub  cg_tmp_7@0, cg_tmp_6@0, bwidth@0
cmp  x@0, bwidth@0
phi {GE}  cg_tmp_8@0, cg_tmp_7@0, cg_tmp_6@0
add  cg_tmp_9@0, y@0, bheight@0
cmp  y@0, cg_tmp_2@0
phi {LT}  cg_tmp_12@0, cg_tmp_9@0, y@0
sub  cg_tmp_13@0, cg_tmp_12@0, bheight@0
cmp  y@0, bheight@0
phi {GE}  cg_tmp_14@0, cg_tmp_13@0, cg_tmp_12@0
mul  cg_tmp_15@0, cg_tmp_14@0, bwidth@0
add  cg_tmp_16@0, cg_tmp_15@0, cg_tmp_8@0
add  cg_tmp_17@0, in_ptr@0, cg_tmp_16@0
memr  cg_tmp_19@0, cg_tmp_17@0
mov  out, cg_tmp_19@0
mov  cg_tmp_20@0, west
cmp  x@0, cg_tmp_1@0
phi {LT}  cg_tmp_21@0, cg_tmp_20@0, cg_tmp_19@0
mov  out, cg_tmp_21@0
mov  cg_tmp_22@0, east
cmp  x@0, bwidth@0
phi {GE}  cg_tmp_23@0, cg_tmp_22@0, cg_tmp_21@0
mov  out, cg_tmp_23@0
mov  cg_tmp_24@0, north
cmp  y@0, cg_tmp_2@0
phi {LT}  cg_tmp_25@0, cg_tmp_24@0, cg_tmp_23@0
mov  out, cg_tmp_25@0
mov  cg_tmp_26@0, south
cmp  y@0, bheight@0
phi {GE}  cg_tmp_27@0, cg_tmp_26@0, cg_tmp_25@0
'''
	assert match_code(code, pattern)

def test_replace_regnames_by_tmp_names():
	''' Test name mangling in code generator. '''
	from blip.code.BlipCompiler import Codegenerator
	codegen = Codegenerator(Compiler())
	ignore = ['in@0']
	code = [
		('mov', None, 'a@1', 'in@0'),
		('imm', None, 'b@2', 3),
		('cmp', None, 'b@2', 'a@1'),
		('sub', None, 'a@2', 'in@0', 'a@1'),
		('mov', None, 'c@1', 'west'),
		('mov', None, 'out', 'c@1'),
		('mov', None, 'get2D___return', 'east'),
		('mov', None, 'c@3', 'get2D___return')
	]

	expect_res = [
		('mov', None, 'cg_tmp_1@0', 'in@0'),
		('imm', None, 'cg_tmp_2@0', 3),
		('cmp', None, 'cg_tmp_2@0', 'cg_tmp_1@0'),
		('sub', None, 'cg_tmp_3@0', 'in@0', 'cg_tmp_1@0'),
		('mov', None, 'cg_tmp_4@0', 'west'),
		('mov', None, 'out', 'cg_tmp_4@0'),
		('mov', None, 'cg_tmp_5@0', 'east'),
		('mov', None, 'cg_tmp_6@0', 'cg_tmp_5@0')
	]
	test_res, _ = codegen.replace_regnames_by_tmp_names(code, ignore)
	assert expect_res == test_res

def test_liveness_analysis():
	''' Test the liveness analysis, note that this analysis only accepts SSA code. '''
	from blip.code.BlipCompiler import RegisterAllocator
	code = [
		('mov', None, 'a', 'e'),
		('add', None, 'b', 'a', 'b'),
		('mov', None, 'out', 'b'),
		('mov', None, 'd', 'west'),
		('cmp', None, 'e', 'a'),
		('memw_imm', None, 34, 'd'),
		('imm', None, 'f', 12)
	]
	test_liveness = RegisterAllocator.liveness_analysis(code)
	expect_liveness = {
		'a':(0, [1,4]), 'b':(1, [1, 2]), 'd':(3, [5]), 'e':(0, [0,4]), 'f':(6, [6])
	}
	assert expect_liveness == test_liveness

def test_register_allocator_leave_special_regs():
	''' Test whether the register allocator does not allocate regs for special registers. '''
	from blip.code.BlipCompiler import RegisterAllocator
	from blip.simulator.opcodes import SPECIAL_REGS
	code = [
		('mov', None, 'a', 'e'),
		('add', None, 'b', 'a', 'b'),
		('mov', None, 'out', 'b'),
		('mov', None, 'd', 'west'),
		('cmp', None, 'e', 'a'),
		('memw_imm', None, 34, 'd'),
		('imm', None, 'f', 12)
	]
	liveness = RegisterAllocator.liveness_analysis(code)
	test_code, test_mapping = RegisterAllocator.register_allocation(code, liveness, 8)
	# test if no special reg is in the mapping of the regalloc
	assert len(set(test_mapping.keys()).intersection(SPECIAL_REGS)) == 0

def all_test(options = {}):
	tests = [\
		test_undefined_variable_detection,\
		test_resolve_seq_values,\
		test_compile_add,\
		test_compile_conditional,\
		test_simple_loop_noseq,\
		test_simple_loop,\
		test_simple_loop_emit_noseq,\
		test_simple_loop_emit,\
		test_inplace_operator,\
		test_replace_phi_nodes,\
		test_builtin_loadwest,\
		test_builtin_sendout,\
		test_builtin_transfer,\
		test_th_code,\
		test_const_array_access,\
		test_delayed_memderef,\
		test_delayed_memderef2,\
		test_memderef_inside_condassign,\
		test_logic_variable,\
		test_logic_variable2,\
		test_phi_codegen,\
		test_jmp_codegen,\
		test_patch_arguments,\
		test_patch_reg_arguments,\
		test_conv_code_noseq,\
		test_2D_const_list,\
		test_peephole_addorsub_zero,\
		test_peephole_twoconstant_sources,\
		test_get2D,\
		test_get2D_copy_propagation,\
		test_replace_regnames_by_tmp_names,\
		test_liveness_analysis,\
		test_register_allocator_leave_special_regs,\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

