from __future__ import with_statement
import math
from blip.simulator.opcodes import *
from blip.code.skeletons import map_pixel_to_pixel
from blip.code.codegen import scoped_alloc, scoped_preg_alloc, Code
from blip.simulator.series import *

# collection of some codegen functions for common
# vision algorithms

def gen_threshold(code, block_size, args):
    th = args['th']
    with scoped_alloc(code, 6) as (out_ptr_r, th_r, const_1, const_255, const_0, in_ptr_r):
        # out pointer
        yield Imm(out_ptr_r, block_size[0]*block_size[1]) 

	# constants
        yield Imm(th_r, th) 
        yield Imm(const_1, 1) 
        yield Imm(const_255, 255) 
        yield Imm(const_0, 0) 

        # in pointer
        yield Mov(in_ptr_r, const_0) 
    
        for i in xrange(block_size[0]):
            for j in xrange(block_size[1]):
                with scoped_alloc(code, 1) as tmp:
                    yield MemR(tmp, in_ptr_r) 

                    yield Cmp(tmp, th_r)  
                    yield Mov(tmp, const_0)  
                    yield Mov(tmp, const_255, 'GT')   

                    yield MemW(out_ptr_r, tmp) 
                    if(not (j == block_size[1]-1 and i == block_size[0]-1)):
                        yield Add(in_ptr_r, in_ptr_r, const_1)  
                        yield Add(out_ptr_r, out_ptr_r, const_1)

def gen_threshold_2(code, block_size, args):
	th = args['th']
	''' Example of skeletons + optimisation. '''
	from code.trace_optimiser import Optimiser, ImmediatePass, MemoryPass
	def pixel_op(code, pin, pout, args, block_size):
		th = args['th']
		with scoped_alloc(code, 3) as (th_r, v, const_255):
			yield Imm(th_r, th)
			yield Cmp(pin, th_r)
			yield Imm(pout, 255, cond='GT')
			yield Xor(pout, pout, pout, cond='LE')

	in_ptr = 0
	out_ptr = block_size[0]*block_size[1]
	def codegen(code, block_size, args):
		return map_pixel_to_pixel(code, in_ptr, out_ptr, pixel_op, args, block_size)

	if 'disable_optimization' in args and args['disable_optimization']:
		for x in codegen(code, block_size, args):
			yield x
	else:
		optimiser = Optimiser(30)
		optimiser.register_pass(ImmediatePass(optimiser))
		optimiser.register_pass(MemoryPass(optimiser))

		for x in optimiser.run(code, codegen, block_size, args):
			yield x

def gen_gray_image_code(code, block_size, args):
    ''' generate flat gray image '''

    with scoped_alloc(code, 3) as (out_ptr_r, const_1, const_gray):
        # init pointer to output memory
        yield Imm(out_ptr_r, block_size[0]*block_size[1]) 
        yield Imm(const_1, 1) 
      
        # gen gray image
        yield Imm(const_gray, 128) 
        for i in xrange(block_size[0]):
            for j in xrange(block_size[1]):
                yield MemW(out_ptr_r, const_gray) 
                yield Add(out_ptr_r, out_ptr_r, const_1)
    

def gen_copy_to_out(code, block_size, args):
    with scoped_alloc(code, 3) as (out_ptr_r, in_ptr_r, const_1):
        # init pointer to output memory
        yield Imm(out_ptr_r, block_size[0]*block_size[1]) 

	# init src ptr
	yield Xor(in_ptr_r, in_ptr_r, in_ptr_r)

        # inc value
        yield Imm(const_1, 1) 

        for i in xrange(block_size[0]):
            for j in xrange(block_size[1]):
                with scoped_alloc(code, 1) as tmp:
                    yield MemR(tmp, in_ptr_r) 
                    yield MemW(out_ptr_r, tmp) 
    
                    yield Add(in_ptr_r, in_ptr_r, const_1)
                    yield Add(out_ptr_r, out_ptr_r, const_1)


def gen_invert_im(code, block_size, args):
    with scoped_alloc(code, 4) as (out_ptr_r, in_ptr_r, const_1, const_255):
        # init pointer to output memory
        yield Imm(out_ptr_r, block_size[0]*block_size[1]) 

	# init src ptr
	yield Xor(in_ptr_r, in_ptr_r, in_ptr_r)

        # inc value
        yield Imm(const_1,1) 
        yield Imm(const_255, 255)

        for i in xrange(block_size[0]):
            for j in xrange(block_size[1]):
                with scoped_alloc(code, 1) as tmp:
                    yield MemR(tmp, in_ptr_r) 
                    yield Sub(tmp, const_255, tmp)
                    yield MemW(out_ptr_r, tmp) 
    
                    yield Add(in_ptr_r, in_ptr_r, const_1)
                    yield Add(out_ptr_r, out_ptr_r, const_1)

def default_argument_setup():
    return {'mask_size':64, 'pe_dim':1}

def gen_erosion(code, block_size, args):
    mask_size = args['mask_size']
    mask_size = (mask_size, mask_size)
    hrow, hcol = [int(math.floor(x/2.)) for x in mask_size]
    rsize, csize = block_size

    with scoped_alloc(code, 5) as (const_0, out_ptr_r, in_ptr_r, const_1, const_255):
        # init pointer to output memory
        # r2 is the pointer to the output block buffer
        yield Imm(out_ptr_r, block_size[0]*block_size[1]) 

	# init src ptr
        # r0 is the pointer to the input block buffer
        yield Imm(in_ptr_r, 0)        
        #yield Xor(in_ptr_r, in_ptr_r, in_ptr_r)

        # inc value
        yield Imm(const_1, 1) 

        # 0 const
        yield Imm(const_0, 0) 

        # 255 const
        yield Imm(const_255, 255) 

        # generate erosion instructions, unoptimised
        for i in xrange(rsize):
            for j in xrange(csize):
                with scoped_alloc(code, 2) as (acc, mov_tmp):
                    # accumulator set to 255 first 
                    yield Mov(acc, const_255) 
                    for ii in xrange(mask_size[0]):
                        for jj in xrange(mask_size[1]):
                            # add the data transfers from neighboring blocks
                            r_rind, c_rind = -hrow + ii, -hcol + jj
                            r_ind, c_ind = i + r_rind, j + c_rind
                            mov_instr = []
                            if r_ind < 0:
                                r_ind += rsize
                                mov_instr.append(Mov(code.out, mov_tmp))
                                mov_instr.append(Mov(mov_tmp, code.north))
                            elif r_ind >= rsize:
                                r_ind -= rsize
                                mov_instr.append(Mov(code.out, mov_tmp))
                                mov_instr.append(Mov(mov_tmp, code.south))
                            if c_ind < 0:
                                c_ind += csize
                                mov_instr.append(Mov(code.out, mov_tmp))
                                mov_instr.append(Mov(mov_tmp, code.west))
                            elif c_ind >= csize:
                                c_ind -= csize
                                mov_instr.append(Mov(code.out, mov_tmp))
                                mov_instr.append(Mov(mov_tmp, code.east))

                            # load the image value
                            yield MemRImm(mov_tmp, r_ind * block_size[1] + c_ind) 

                            # now add the neighbor communication
                            for instr in mov_instr: 
                                tag_instr(instr, 'communication overhead')
                                yield instr

                            # acc = mov_tmp && acc
                            yield And(acc, acc, mov_tmp) 

                # write back data to frame buffer
                yield MemW(out_ptr_r, acc) 

                # increment data pointers
                yield Add(in_ptr_r, in_ptr_r, const_1) 
                yield Add(out_ptr_r, out_ptr_r, const_1) 


def gen_th_block_erosion(code, block_size, args):
    th = args['th']
    mask_size = args['mask_size']
    mask_size = (mask_size, mask_size)
    hrow, hcol = [int(math.floor(x/2.)) for x in mask_size]
    rsize, csize = block_size

    with scoped_alloc(code, 6) as (const_0, const_th, out_ptr_r, in_ptr_r, const_1, const_255):
        # init pointer to output memory
        # r3 is the pointer to the output block buffer
        yield Imm(out_ptr_r, block_size[0]*block_size[1]) 

	# init src ptr
        # r0 is the pointer to the input block buffer
	yield Xor(in_ptr_r, in_ptr_r, in_ptr_r)

        # inc value
        yield Imm(const_1, 1) 

        # 0 const
        yield Imm(const_0, 0) 

        # 255 const
        yield Imm(const_255, 255) 
        # threshold value
        yield Imm(const_th, th) # yield Imm(const_th) before

        # generate erosion instructions, unoptimised
        for i in xrange(rsize):
            for j in xrange(csize):
                with scoped_alloc(code, 2) as (acc, mov_tmp):
                    # accumulator set to 255 first 
                    yield Mov(acc, const_255) 
                    for ii in xrange(mask_size[0]):
                        for jj in xrange(mask_size[1]):
                            # add the data transfers from neighboring blocks
                            r_rind, c_rind = -hrow + ii, -hcol + jj
                            r_ind, c_ind = i + r_rind, j + c_rind
                            mov_instr = []
                            if r_ind < 0:
                                r_ind += rsize
                                mov_instr.append(Mov(code.out, mov_tmp))
                                mov_instr.append(Mov(mov_tmp, code.north))
                            elif r_ind >= rsize:
                                r_ind -= rsize
                                mov_instr.append(Mov(code.out, mov_tmp))
                                mov_instr.append(Mov(mov_tmp, code.south))
                            if c_ind < 0:
                                c_ind += csize
                                mov_instr.append(Mov(code.out, mov_tmp))
                                mov_instr.append(Mov(mov_tmp, code.west))
                            elif c_ind >= csize:
                                c_ind -= csize
                                mov_instr.append(Mov(code.out, mov_tmp))
                                mov_instr.append(Mov(mov_tmp, code.east))

                            # load the image value
                            yield MemRImm(mov_tmp, r_ind * block_size[1] + c_ind) 

                            # now add the neighbor communication
                            for instr in mov_instr: 
                                tag_instr(instr, 'communication overhead')
                                yield instr

                            # perform calculation
                            # threshold
                            yield Cmp(mov_tmp, const_th) 
                            yield Mov(mov_tmp, const_0)  # mov_tmp = 0
                            yield Mov(mov_tmp, const_255, 'GT')  # mov_tmp = 255 if r1 > th

                            # acc = mov_tmp && acc
                            yield And(acc, acc, mov_tmp) 

                # write back data to frame buffer
                yield MemW(out_ptr_r, acc) 

                # increment data pointers
                yield Add(in_ptr_r, in_ptr_r, const_1) 
                yield Add(out_ptr_r, out_ptr_r, const_1) 

    
def gen_block_filter_coef(size):
    return [[1./(size*size) for i in xrange(size)] for j in xrange(size)]

def gen_conv(code, block_size, args):
    coeff = args['coeff']
    coeff_size = (len(coeff), len(coeff[0]))
    hrow, hcol = [int(math.floor(x/2.)) for x in coeff_size]
    rsize, csize = block_size

    with scoped_alloc(code, 3) as (out_ptr_r, in_ptr_r, const_1):
        # r0 is the pointer to the input block buffer
        yield Xor(in_ptr_r, code.r(0), code.r(0)) 
    
        # r3 is the pointer to the output block buffer
        yield Imm(out_ptr_r, rsize*csize) 

        # inc value
        yield Imm(const_1, 1) 

        # generate convolution instructions, unoptimised
        for i in xrange(rsize):
            for j in xrange(csize):
                with scoped_alloc(code, 2) as (acc, tmp):
                    # r2 is used as accumulator, clean reg first
                    yield Xor(acc, acc, acc) 
                    for ii, r in enumerate(coeff):
                        for jj, c in enumerate(r):
                            # add the data transfers from neighboring blocks
                            r_rind, c_rind = -hrow + ii, -hcol + jj
                            r_ind, c_ind = i + r_rind, j + c_rind
                            mov_instr = []
                            if r_ind < 0:
                                r_ind += rsize
                                mov_instr.append(Mov(code.out, tmp))
                                mov_instr.append(Mov(tmp, code.north))
                            elif r_ind >= rsize:
                                r_ind -= rsize
                                mov_instr.append(Mov(code.out, tmp))
                                mov_instr.append(Mov(tmp, code.south))
                            if c_ind < 0:
                                c_ind += csize
                                mov_instr.append(Mov(code.out, tmp))
                                mov_instr.append(Mov(tmp, code.west))
                            elif c_ind >= csize:
                                c_ind -= csize
                                mov_instr.append(Mov(code.out, tmp))
                                mov_instr.append(Mov(tmp, code.east))

                            # load the image value
                            yield MemRImm(tmp, r_ind * block_size[1] + c_ind) 

                            # now add the neighbor communication
                            for instr in mov_instr: 
                                tag_instr(instr, 'communication overhead')
                                yield instr

                            # the value is now in r1 and coefficient is loaded
                            # perform calculation
                            with scoped_alloc(code, 1) as c_r:
                                yield Imm(c_r, c) 
                                yield Mul(tmp, tmp, c_r) 
                            yield Add(acc, acc, tmp) 

                        # write back data to frame buffer
                        yield MemW(out_ptr_r, acc) 

                        # increment data pointers
                        yield Add(in_ptr_r, in_ptr_r, const_1) 
                        yield Add(out_ptr_r, out_ptr_r, const_1) 


def gen_bbs(code, block_size, args):
    th = args['th']
    alpha = args['alpha']
    width, height = block_size
    block_mem_size = width * height
    # pointers:
    src_ptr = 0
    res_ptr = block_mem_size
    back_ptr = 2*block_mem_size

    with scoped_alloc(code, 3) as (const_alpha, const_1_m_alpha, const_th):
        # setup parameters
        yield Imm(const_alpha, alpha)
        yield Imm(const_1_m_alpha, 1-alpha)
        yield Imm(const_th, th)

        # regs:
        # ip_n     : I_p[n]
        # ib_n_1   : Ibackground_p[n-1]
        # ib_n     : Ibackground_p[n]
        # abbsdiff : abs(I_p[n] - Ibackground_p[n-1]

        for i in xrange(block_mem_size):
            with scoped_alloc(code, 5) as (ip_n, ib_n_1, ib_n, absdiff, res):
                # I_background(n) = I*alpha + I_background(n-1)*(1-alpha)
                yield MemRImm(ip_n, src_ptr + i)
                yield Mul(ib_n, ip_n, const_alpha)
                yield MemRImm(ib_n_1, back_ptr + i)
                yield Mul(ib_n_1, ib_n_1, const_1_m_alpha)
                yield Add(ib_n, ib_n, ib_n_1)
                yield MemWImm(back_ptr + i, ib_n)

                # I_res = abs(I - I_background) > th
                # equivalent to:
                # if I >= I_background:
                #   I_res = (I - I_background) > th
                # else:
                #   I_res = (I_background - I) > th
                yield Cmp(ip_n, ib_n)
                yield Sub(absdiff, ip_n, ib_n, cond='GE')
                yield Sub(absdiff, ib_n, ip_n, cond='LT')
                yield Cmp(absdiff, const_th)
                yield Imm(res, 0)
                yield Imm(res, 255, cond='GT')
                yield MemWImm(res_ptr + i, res)


def gen_conn_comp_lbl3(code, block_size, args):
    block_mem_size = block_size[0] * block_size[1]
    src_ptr = 0
    res_ptr = block_mem_size
    test_ptr = block_mem_size
    number_of_runs = args['number_of_runs']
    with scoped_alloc(code, 15) as (const_0, const_1, const_2, const_3, const_255, left, top, bottom, right, curr, res, tmp, tmp2, x, y):
        yield Imm(const_0, 0)
        yield Imm(const_1, 1)
        yield Imm(const_2, 2)
        yield Imm(const_3, 3)
        yield Imm(const_255, 255)

        for i in xrange(block_mem_size):
            yield Imm(x, i%block_size[0])
            yield Imm(y, i/block_size[0]) 
            if(i % block_size[0] == 0):
                if i == 0:
                    yield MemRImm(curr, src_ptr + i)
                    yield Cmp(curr, const_255)
                    yield Pxid(res, x, y, cond='EQ')
                    yield Mov(res, const_0, cond='NEQ')
                else:
                    yield MemRImm(top, res_ptr + i - block_size[0])
                    yield MemRImm(curr, src_ptr + i)

                    yield Cmp(curr, const_255)
                    yield Pxid(res, x, y, cond='EQ')
                    yield Mov(res, const_0, cond='NEQ')

                    yield Mov(tmp, const_0)
                    yield Cmp(top, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(top, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, top, cond='EQ')

                yield MemWImm(res_ptr + i, res)
            elif(i < block_size[0]):
                
                yield MemRImm(curr, src_ptr + i)
                yield MemRImm(left, res_ptr + i - 1)

                yield Cmp(curr, const_255)
                yield Pxid(res, x, y, cond='EQ')
                yield Mov(res, const_0, cond='NEQ')

                yield Mov(tmp, const_0)
                yield Cmp(left, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(left, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, left, cond='EQ')

                yield MemWImm(res_ptr + i, res)
            else:
                yield MemRImm(top, res_ptr + i - block_size[0])
                yield MemRImm(curr, src_ptr + i)
                yield MemRImm(left, res_ptr + i - 1)

                yield Cmp(curr, const_255)
                yield Pxid(res, x, y, cond='EQ')
                yield Mov(res, const_0, cond='NEQ')

                yield Mov(tmp, const_0)
                yield Cmp(top, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(top, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, top, cond='EQ')

                yield Mov(tmp, const_0)
                yield Cmp(left, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(left, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, left, cond='EQ')

                yield MemWImm(res_ptr + i, res)

        DIFFERENT_SCANS = 4
        for s in xrange(number_of_runs):
            if(s % DIFFERENT_SCANS == 0):
                r = diagonal_snake_start_left_up_dir_down(block_size)
                north = True
                east = False
                south = False
                west = True
            elif(s % DIFFERENT_SCANS == 1):
                r = diagonal_snake_start_right_up_dir_down(block_size)
                north = True
                east = True
                south = False
                west = False
            elif(s % DIFFERENT_SCANS == 2):
                r = diagonal_snake_start_right_down_dir_down(block_size)
                north = False
                east = True
                south = True
                west = False
            elif(s % DIFFERENT_SCANS == 3):
                r = diagonal_snake_start_left_down_dir_down(block_size)
                north = False
                east = False
                south = True
                west = True
            setIdle = True
            for i in r:

                yield MemRImm(res, res_ptr + i)

                mov_instr = []
                if(north):
                    if i < block_size[0]:
                        yield MemRImm(tmp, res_ptr + block_mem_size - block_size[0] + i)
                        mov_instr.append(Mov(code.out, tmp))
                        mov_instr.append(Mov(top, code.north))
                    else:
                        yield MemRImm(top, res_ptr + i - block_size[0])
                if(south):
                    if i >= block_mem_size - block_size[0]:
                        yield MemRImm(tmp, res_ptr  + i - block_mem_size + block_size[0])
                        mov_instr.append(Mov(code.out, tmp))
                        mov_instr.append(Mov(bottom, code.south))
                    else:
                        yield MemRImm(bottom, res_ptr + i + block_size[0])
                if(east):
                    if i % block_size[0] == block_size[0]-1:
                        yield MemRImm(tmp, res_ptr + i - block_size[0] + 1)
                        mov_instr.append(Mov(code.out, tmp))
                        mov_instr.append(Mov(right, code.east))
                    else:
                        yield MemRImm(right, res_ptr + i + 1)
                if(west):
                    if i % block_size[0] == 0:
                        yield MemRImm(tmp, res_ptr + i + block_size[0] - 1)
                        mov_instr.append(Mov(code.out, tmp))
                        mov_instr.append(Mov(left, code.west))
                    else:
                        yield MemRImm(left, res_ptr + i - 1)

                # now add the neighbour communication
                for instr in mov_instr: 
                    tag_instr(instr, 'communication overhead')
                    yield instr

                if(north):
                    yield Mov(tmp, const_0)
                    yield Cmp(top, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(top, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, top, cond='EQ')

                if(west):
                    yield Mov(tmp, const_0)
                    yield Cmp(left, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(left, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, left, cond='EQ')

                if(south):
                    yield Mov(tmp, const_0)
                    yield Cmp(bottom, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(bottom, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, bottom, cond='EQ')

                if(east):
                    yield Mov(tmp, const_0)
                    yield Cmp(right, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(right, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, right, cond='EQ')

                yield MemWImm(res_ptr + i, res)


def gen_conn_comp_lbl2(code, block_size, args):
    block_mem_size = block_size[0] * block_size[1]
    src_ptr = 0
    res_ptr = block_mem_size
    test_ptr = block_mem_size
    with scoped_alloc(code, 15) as (const_0, const_1, const_2, const_3, const_255, left, top, bottom, right, curr, res, tmp, tmp2, x, y):
        yield Imm(const_0, 0)
        yield Imm(const_1, 1)
        yield Imm(const_2, 2)
        yield Imm(const_3, 3)
        yield Imm(const_255, 255)

        for i in xrange(block_mem_size):
            yield Imm(x, i%block_size[0])
            yield Imm(y, i/block_size[0]) 
            if(i % block_size[0] == 0):
                if i == 0:
                    yield MemRImm(curr, src_ptr + i)
                    yield Cmp(curr, const_255)
                    yield Pxid(res, x, y, cond='EQ')
                    yield Mov(res, const_0, cond='NEQ')
                else:
                    yield MemRImm(top, res_ptr + i - block_size[0])
                    yield MemRImm(curr, src_ptr + i)

                    yield Cmp(curr, const_255)
                    yield Pxid(res, x, y, cond='EQ')
                    yield Mov(res, const_0, cond='NEQ')

                    yield Mov(tmp, const_0)
                    yield Cmp(top, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(top, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, top, cond='EQ')

                yield MemWImm(res_ptr + i, res)
            elif(i < block_size[0]):
                
                yield MemRImm(curr, src_ptr + i)
                yield MemRImm(left, res_ptr + i - 1)

                yield Cmp(curr, const_255)
                yield Pxid(res, x, y, cond='EQ')
                yield Mov(res, const_0, cond='NEQ')

                yield Mov(tmp, const_0)
                yield Cmp(left, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(left, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, left, cond='EQ')

                yield MemWImm(res_ptr + i, res)
            else:
                yield MemRImm(top, res_ptr + i - block_size[0])
                yield MemRImm(curr, src_ptr + i)
                yield MemRImm(left, res_ptr + i - 1)

                yield Cmp(curr, const_255)
                yield Pxid(res, x, y, cond='EQ')
                yield Mov(res, const_0, cond='NEQ')

                yield Mov(tmp, const_0)
                yield Cmp(top, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(top, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, top, cond='EQ')

                yield Mov(tmp, const_0)
                yield Cmp(left, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(left, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, left, cond='EQ')

                yield MemWImm(res_ptr + i, res)

        DIFFERENT_SCANS = 4
        for s in xrange(20):
            '''if(s % DIFFERENT_SCANS == 0):
                r = diagonal_start_left_up_dir_down(block_size)
            elif(s % DIFFERENT_SCANS == 1):
                r = diagonal_start_right_up_dir_down(block_size)
            elif(s % DIFFERENT_SCANS == 2):
                r = diagonal_start_right_down_dir_down(block_size)
            elif(s % DIFFERENT_SCANS == 3):
                r = diagonal_start_left_down_dir_down(block_size)'''
            if(s % DIFFERENT_SCANS == 0):
                r = diagonal_snake_start_left_up_dir_down(block_size)
            elif(s % DIFFERENT_SCANS == 1):
                r = diagonal_snake_start_right_up_dir_down(block_size)
            elif(s % DIFFERENT_SCANS == 2):
                r = diagonal_snake_start_right_down_dir_down(block_size)
            elif(s % DIFFERENT_SCANS == 3):
                r = diagonal_snake_start_left_down_dir_down(block_size)
                '''elif(s % DIFFERENT_SCANS == 4):
                    r = vertical_snake_start_up_left(block_size)
                elif(s % DIFFERENT_SCANS == 3):
                    r = horizontal_snake_start_up_right(block_size)'''
            for i in r:

                yield MemRImm(res, res_ptr + i)

                mov_instr = []
                if i < block_size[0]:
                    yield MemRImm(tmp, res_ptr + block_mem_size - block_size[0] + i)
                    mov_instr.append(Mov(code.out, tmp))
                    mov_instr.append(Mov(top, code.north))
                else:
                    yield MemRImm(top, res_ptr + i - block_size[0])
                if i >= block_mem_size - block_size[0]:
                    yield MemRImm(tmp, res_ptr  + i - block_mem_size + block_size[0])
                    mov_instr.append(Mov(code.out, tmp))
                    mov_instr.append(Mov(bottom, code.south))
                else:
                    yield MemRImm(bottom, res_ptr + i + block_size[0])
                if i % block_size[0] == block_size[0]-1:
                    yield MemRImm(tmp, res_ptr + i - block_size[0] + 1)
                    mov_instr.append(Mov(code.out, tmp))
                    mov_instr.append(Mov(right, code.east))
                else:
                    yield MemRImm(right, res_ptr + i + 1)
                if i % block_size[0] == 0:
                    yield MemRImm(tmp, res_ptr + i + block_size[0] - 1)
                    mov_instr.append(Mov(code.out, tmp))
                    mov_instr.append(Mov(left, code.west))
                else:
                    yield MemRImm(left, res_ptr + i - 1)

                # now add the neighbor communication
                for instr in mov_instr: 
                    tag_instr(instr, 'communication overhead')
                    yield instr

                yield Mov(tmp, const_0)
                yield Cmp(top, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(top, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, top, cond='EQ')

                yield Mov(tmp, const_0)
                yield Cmp(left, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(left, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, left, cond='EQ')

                yield Mov(tmp, const_0)
                yield Cmp(bottom, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(bottom, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, bottom, cond='EQ')

                yield Mov(tmp, const_0)
                yield Cmp(right, res)
                yield Add(tmp, tmp, const_1, cond='LE')
                yield Cmp(right, const_0)
                yield Add(tmp, tmp, const_1, cond='NEQ')
                yield Cmp(tmp, const_2)
                yield Mov(res, right, cond='EQ')

                yield MemWImm(res_ptr + i, res)

# export PYTHONPATH=$PYTHONPATH:.
# python blip/simulator/interpreter.py --codegen blip.blipcode.gen_conn_comp_lbl -s data/test_conn_comp_lbl.png -o conn_comp_lbl_out.png
def gen_conn_comp_lbl(code, block_size, args):
    width, height = block_size
    block_mem_size = width * height
    src_ptr = 0
    res_ptr = block_mem_size
    test_ptr = block_mem_size
    with scoped_alloc(code, 16) as (const_0, const_1, const_2, const_3, const_255, lbl, left, top, bottom, right, curr, res, tmp, tmp2, const_4, const_5):
        yield Imm(const_0, 0)
        yield Imm(const_1, 1)
        yield Imm(const_255, 255)
        yield Imm(const_2, 2)
        yield Imm(const_3, 3)
        yield Imm(const_4, 4)
        yield Imm(const_5, 5)
        yield Imm(lbl, 0)
        yield Imm(tmp, 0)
        yield Imm(tmp2, 0)
            
        yield MemRImm(curr, src_ptr)
        yield MemRImm(left, src_ptr)
        yield MemRImm(top, src_ptr)

        for i in xrange(block_mem_size):
                if(i % block_size[0] == 0):
                    if i == 0:
                        yield MemRImm(curr, src_ptr + i)
                        yield Cmp(curr, const_255)
                        yield Add(lbl, lbl, const_1, cond='EQ')
                        yield Mov(res, lbl, cond='EQ')
                        yield Mov(res, const_0, cond='NEQ')
                    else:
                        yield MemRImm(curr, src_ptr + i)
                        yield MemRImm(top, src_ptr + i - block_size[0])
                        yield Mov(res, const_0)

                        yield Mov(tmp, const_0)
                        yield Cmp(curr, const_255)
                        yield Add(tmp, tmp, const_1, cond='EQ')
                        yield Cmp(top, const_0, cond='EQ')
                        yield Add(tmp, tmp, const_1, cond='EQ')
                        yield Cmp(tmp, const_2)
                        yield Add(lbl, lbl, const_1, cond='EQ')
                        yield Mov(res, lbl, cond='EQ')

                        yield Mov(tmp, const_0)
                        yield Cmp(curr, const_255)
                        yield Add(tmp, tmp, const_1, cond='EQ')
                        yield Cmp(top, const_255, cond='EQ')
                        yield Add(tmp, tmp, const_1, cond='EQ')
                        yield Cmp(tmp, const_2)
                        yield MemRImm(top, res_ptr + i - block_size[0])
                        yield Mov(res, top, cond='EQ')

                    yield MemWImm(res_ptr + i, res)
                elif(i < block_size[0]):
                    yield Mov(res, const_0)
                    
                    yield MemRImm(curr, src_ptr + i)
                    yield MemRImm(left, src_ptr + i - 1)

                    yield Mov(tmp, const_0)
                    yield Cmp(curr, const_255)
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(left, const_0, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(tmp, const_2)
                    yield Add(lbl, lbl, const_1, cond='EQ')
                    yield Mov(res, lbl, cond='EQ')

                    yield Mov(tmp, const_0)
                    yield Cmp(curr, const_255)
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(left, const_255, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(tmp, const_2)
                    yield MemRImm(tmp, res_ptr + i - 1, cond='EQ')
                    yield Mov(res, tmp, cond='EQ')
                    yield MemWImm(res_ptr + i, res)
                else:
                    yield Mov(res, const_0)

                    yield MemRImm(top, src_ptr + i - block_size[0])
                    yield MemRImm(curr, src_ptr + i)
                    yield MemRImm(left, src_ptr + i - 1)

                    yield Mov(tmp, const_0)
                    yield Mov(tmp2, const_0)
                    yield Cmp(curr, const_255)
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(left, const_255, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(top, const_255, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(tmp, const_3)
                    yield Mov(tmp, const_0)
                    yield MemRImm(tmp, res_ptr + i - block_size[0], cond='EQ')
                    yield MemRImm(tmp2, res_ptr + i - 1, cond='EQ')
                    yield Cmp(tmp, tmp2, cond='EQ')
                    yield Mov(res, tmp2, cond='GE')
                    yield Mov(res, tmp, cond='LE')

                    yield Mov(tmp, const_0)
                    yield Cmp(curr, const_255)
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(left, const_255, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(top, const_0, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(tmp, const_3)
                    yield MemRImm(tmp, res_ptr + i - 1, cond='EQ')
                    yield Mov(res, tmp, cond='EQ')

                    yield Mov(tmp, const_0)
                    yield Cmp(curr, const_255)
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(left, const_0, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(top, const_255, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(tmp, const_3)
                    yield MemRImm(tmp, res_ptr + i - block_size[0], cond='EQ')
                    yield Mov(res, tmp, cond='EQ')

                    yield Mov(tmp, const_0)
                    yield Cmp(curr, const_255)
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(left, const_0, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(top, const_0, cond='EQ')
                    yield Add(tmp, tmp, const_1, cond='EQ')
                    yield Cmp(tmp, const_3)
                    yield Add(lbl, lbl, const_1, cond='EQ')
                    yield Mov(res, lbl, cond='EQ')

                    yield MemWImm(res_ptr + i, res)

        for s in xrange(10):
            for j in xrange(block_size[0]):
                if(s % 4 == 0):
                    if(j % 2 == 0):
                        r = range(block_size[0]*j, block_size[0]*(j+1), 1)
                    else:
                        r = range(block_size[0]*(j+1)-1, block_size[0]*j-1, -1)
                elif(s % 4 == 1):
                    if(j % 2 == 0):
                        r = range(j, block_mem_size, block_size[0])
                    else:
                        r = range(block_mem_size-block_size[0]+j, j-1, -block_size[0])
                elif(s % 4 == 2):
                    if(j % 2 == 0):
                        r = range(block_size[0]*(block_size[0]-j), block_size[0]*(block_size[0]-j+1), 1)
                    else:
                        r = range(block_size[0]*(block_size[0]-j+1)-1, block_size[0]*(block_size[0]-j)-1, -1)
                elif(s % 4 == 3):
                    if(j % 2 == 0):
                        r = range(block_mem_size-j-1, j-1, -block_size[0])
                    else:
                        r = range(block_size[0]-j-1, block_mem_size, block_size[0])
                for i in r:

                    yield MemRImm(curr, res_ptr + i)
                    yield MemRImm(bottom, res_ptr + i + block_size[0])
                    yield MemRImm(right, res_ptr + i + 1)
                    yield MemRImm(top, res_ptr + i - block_size[0])
                    yield MemRImm(left, res_ptr + i - 1)

                    mov_instr = []
                    if i < block_size[0]:
                        yield MemRImm(tmp, res_ptr + block_mem_size - block_size[0] + i)
                        mov_instr.append(Mov(code.out, tmp))
                        mov_instr.append(Mov(top, code.north))
                    if i >= block_mem_size - block_size[0]:
                        yield MemRImm(tmp, res_ptr  + i - block_mem_size + block_size[0])
                        mov_instr.append(Mov(code.out, tmp))
                        mov_instr.append(Mov(bottom, code.south))
                    if i % block_size[0] == block_size[0]-1:
                        yield MemRImm(tmp, res_ptr + i - block_size[0] + 1)
                        mov_instr.append(Mov(code.out, tmp))
                        mov_instr.append(Mov(right, code.east))
                    if i % block_size[0] == 0:
                        yield MemRImm(tmp, res_ptr + i + block_size[0] - 1)
                        mov_instr.append(Mov(code.out, tmp))
                        mov_instr.append(Mov(left, code.west))

                    # now add the neighbor communication
                    for instr in mov_instr: 
                        tag_instr(instr, 'communication overhead')
                        yield instr

                    # -- #
                    yield Mov(res, curr)
                    yield Mov(tmp, const_0)
                    yield Cmp(top, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(top, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, top, cond='EQ')

                    yield Mov(tmp, const_0)
                    yield Cmp(left, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(left, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, left, cond='EQ')

                    yield Mov(tmp, const_0)
                    yield Cmp(bottom, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(bottom, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, bottom, cond='EQ')

                    yield Mov(tmp, const_0)
                    yield Cmp(right, res)
                    yield Add(tmp, tmp, const_1, cond='LE')
                    yield Cmp(right, const_0)
                    yield Add(tmp, tmp, const_1, cond='NEQ')
                    yield Cmp(tmp, const_2)
                    yield Mov(res, right, cond='EQ')

                    yield MemWImm(res_ptr + i, res)









def four_connected(res, curr, ctr, r1, r2, r3, const_0, const_1, const_5, tmp):
    yield Mov(tmp, const_0)
    yield Cmp(curr, const_0)
    yield Add(tmp, tmp, const_1, cond='NEQ')
    yield Cmp(ctr, r1)
    yield Add(tmp, tmp, const_1, cond='LE')
    yield Cmp(ctr, r2)
    yield Add(tmp, tmp, const_1, cond='LE')
    yield Cmp(ctr, r3)
    yield Add(tmp, tmp, const_1, cond='LE')
    yield Cmp(ctr, const_0)
    yield Add(tmp, tmp, const_1, cond='NEQ')
    yield Cmp(tmp, const_5)
    yield Mov(res, ctr, cond='EQ')

# main program
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    (options, args) = parser.parse_args()

