import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from blip.simulator.opcodes import *
from blip.code.codegen import Code
from tester import run_tests
from tester import get_test_options, parse_test_options
from blip.simulator.interpreter import PE, Interpreter

def zeros(rsize, csize):
    return [[0 for i in range(csize)] for j in range(rsize)]

def test_register_value_fetch():
    block_size = (rsize, csize) = 64,64
    image = zeros(rsize, csize)

    # test code
    def test_code(code, block_size, args):
         yield Imm(code.r(0), 3)

    code = Code()
    code.set_generator(test_code, block_size)
    print str(code)

    # PE
    pe = PE(None, image, (rsize, csize)) 

    # execute program
    for c in code.plain_instr():
        pe.step1(c)
        pe.step2(c)

    assert pe.get_reg_by_name('r0') == 3

def test_xor_reg_clear():
    block_size = (rsize, csize) = 64,64

    # test program
    def test_code(code, block_size, args):
         yield Xor(code.r(0), code.r(0), code.r(0))

    code = Code()
    code.set_generator(test_code, block_size)
    print str(code)

    image = zeros(rsize, csize)
    pe = PE(None, image, (rsize, csize))
    # warning: this makes certain assumptions on internals of PE
    pe.regs[regname_to_id('r0')] = 42

    for c in code.plain_instr():
        pe.step1(c)
        pe.step2(c)

    assert pe.get_reg_by_name('r0') == 0

def test_neg():
    block_size = (rsize, csize) = 32,32

    # test program
    def test_code(code, block_size, args):
         yield Neg(code.r(0), code.r(0))

    code = Code()
    code.set_generator(test_code, block_size)
    print str(code)

    image = zeros(rsize, csize)
    pe = PE(None, image, (rsize, csize))
    # warning: this makes certain assumptions on internals of PE
    pe.regs[regname_to_id('r0')] = 42

    for c in code.plain_instr():
        pe.step1(c)
        pe.step2(c)

    assert pe.get_reg_by_name('r0') == -42.

def test_lid():
    block_size = (2, 2)
    def test_code(code, block_size, args):
        yield Imm(code.r(7), 0)
        yield Lid(code.r(0), code.r(7))
        yield Imm(code.r(6), 1)
        yield Lid(code.r(1), code.r(6))
        yield Imm(code.r(6), block_size[0])
        yield Mul(code.r(0), code.r(0), code.r(6))
        yield Add(code.r(0), code.r(1), code.r(0))
        yield MemW(code.r(0), code.r(0))

    code = Code()
    code.set_generator(test_code, block_size)
    print str(code)

    image = zeros(block_size[1], block_size[0])
    sim = Interpreter(code, image, block_size)
    sim.run_kernel()

    out_image = sim.gen_output_image(0, False)
    width, height = block_size
    for i in xrange(height):
        for j in xrange(width):
            assert out_image[i][j] == (i*width + j)

def test_sleep_wakeup():
    ''' Test sleep and wakeup opcodes. '''
    block_size = (2, 2)
    def test_code(code, block_size, args):
        yield Imm(code.r(0), 1)
        yield Sleep()
        yield Imm(code.r(0), 2)
        yield WakeUp()

    code = Code()
    code.set_generator(test_code, block_size)
    print str(code)

    image = zeros(block_size[1], block_size[0])
    sim = Interpreter(code, image, block_size)
    sim.run_kernel()
    # second imm should not be executed
    assert sim.procs[0][0].get_reg_by_name('r0') == 1

def test_sleep_wakeup2():
    ''' Test sleep and wakeup opcodes. '''
    block_size = (2, 2)
    def test_code_sleep(code, block_size, args):
        yield Sleep()

    code = Code()
    code.set_generator(test_code_sleep, block_size)

    image = zeros(block_size[1], block_size[0])
    sim = Interpreter(code, image, block_size)

    # check if attribute is correct
    assert sim.procs[0][0].is_powerdown() == False
    sim.run_kernel()
    assert sim.procs[0][0].is_powerdown() == True

    # check if attribute is correct after wakeup
    def test_code_sleep_wakeup(code, block_size, args):
        yield Sleep()
        yield WakeUp()
    code2 = Code()
    code2.set_generator(test_code_sleep_wakeup, block_size)

    sim2 = Interpreter(code2, image, block_size)
    assert sim2.procs[0][0].is_powerdown() == False
    sim2.run_kernel()
    assert sim2.procs[0][0].is_powerdown() == False 

def all_test(options = {}):
    tests = [\
	test_register_value_fetch,\
	test_neg,\
	test_xor_reg_clear,\
        test_lid,\
	test_sleep_wakeup,\
	test_sleep_wakeup2,\
    ]
    return run_tests(tests, options)

if __name__ == '__main__':
    opt_parser = get_test_options()
    test_options = parse_test_options(opt_parser)
    if not all_test(test_options): exit(1)

