from __future__ import with_statement

from blip.code.codegen import Code
from blip.simulator.opcodes import *
from blip.support import imageio
import math

def ucharRgb(mag, cmin, cmax):
	"""
	Return a tuple of floats between 0 and 1 for the red, green and
	blue amplitudes.
	"""

	try:
		# normalize to [0,1]
		x = float(mag-cmin)/float(cmax-cmin)
	except:
		# cmax = cmin
		x = 0.5
	blue = int(min((max((4*(0.75-x), 0.)), 1.)) *255)
	red  = int(min((max((4*(x-0.25), 0.)), 1.)) *255)
	green= int(min((max((4*math.fabs(x-0.5)-1., 0.)), 1.)) *255)
	return (red, green, blue)  
		
class Memory(object):
    def __init__(self, size):
        self.size = size
        self.buffer = [0.]*size
    def set(self, index, value):
        self.buffer[index] = value
    def get(self, index):
        return self.buffer[index]

class PE(object):
    ''' Processing Element simulator.

    This class implements the actual instruction set as defined in blip.

    '''
    def set_source_image(self, image, buffer_nr = 0):
        ''' Copy a new source image inside (first) block buffer '''
        bwidth, bheight = self.block_size
        offset = bwidth * bheight * buffer_nr
        assert(len(image) == bheight and len(image[0]) == bwidth)
        for i, row in enumerate(image):
            for j, e in enumerate(row):
                self.memory.set(offset + i*len(row) + j, float(e))

    def __init__(self, host, pe_id, image, block_size, nr_buffer = 4, nr_reg = NR_REG, nr_pregs = NR_POINTER_REGS):
        # config
        self.pe_id = int(pe_id)
        self.host = host
        self.nr_buffer = nr_buffer 
        self.nr_reg = nr_reg
        self.nr_pregs = nr_pregs
        self.block_size = block_size
        self.powerdown = False

        # instantiate memory
        self.memory = Memory(block_size[0]*block_size[1]*self.nr_buffer)
        # copy image into memory
        self.set_source_image(image)

        # registers
        self.regs = [0. for i in xrange(self.nr_reg + len(SPECIAL_REGS))]
        self.pregs = [0. for i in xrange(self.nr_pregs)]
        self.OUT_REG_ID = regname_to_id('out')

        # condition state
        self.cond = {'EQ': 0, 'NEQ': 0, 'GT':0,'LT':0, 'LE':0, 'GE':0}
        self.next_cond = {'EQ': 0, 'NEQ': 0, 'GT':0,'LT':0, 'LE':0, 'GE':0}

        # assignment scheduling
        self.register_assignments = []
        self.pregister_assignments = []
        self.memory_assignments = []

        # run kernel mode variables
        self.current_lid = (0, 0)
        self.current_grid = (0, 0)

    def is_powerdown(self):
        return self.powerdown

    def assign_reg(self, dest_reg_name, value):
        self.register_assignments.append((dest_reg_name, value))

    def assign_preg(self, dest_preg_name, value):
        self.pregister_assignments.append((dest_preg_name, value))

    def assign_mem(self, dest_addr, value):
        self.memory_assignments.append((dest_addr, value))

    def get_reg(self, reg_id):
        ''' Fetch a register or port value by nummeric id. '''
        return self.regs[reg_id]

    def get_reg_by_name(self, reg_name):
        ''' Fetch a register or port value by name. '''
        return self.get_reg(regname_to_id(str(reg_name)))

    def get_preg(self, preg_id):
        ''' Fetch a pointer register by nummeric id. '''
        return self.pregs[preg_id]

    def get_preg_by_name(self, preg_name):
        ''' Fetch a pointer register by name. '''
        return self.get_preg(regname_to_id(str(preg_name)))

    def step1(self, instr):
        ''' First step: instruction interpretation. '''
        opcode = instr.opcode()

        # if there is a condition and the condition is not fullfilled
        # or the processor is in powerdown mode
        # skip execution of this instruction
        cond = instr.cond()
        if (cond and not self.cond[cond]):  return 
        if self.powerdown and opcode != 'wku': return

        if opcode == 'imm':
            src = float(instr.value)
            self.assign_reg(instr.dest, src)
        elif opcode == 'mov':
            src = self.get_reg(instr.src.reg_id)
            self.assign_reg(instr.dest, src)
        elif opcode == 'memr':
            self.assign_reg(instr.dest, self.memory.get(int(self.get_reg(instr.src.reg_id))))
        elif opcode == 'memr_imm':
            self.assign_reg(instr.dest, self.memory.get(instr.src))
        elif opcode == 'memw':
            self.assign_mem(int(self.get_reg(instr.dest.reg_id)), self.get_reg(instr.src.reg_id))
        elif opcode == 'memw_imm':
            self.assign_mem(instr.dest, self.get_reg(instr.src.reg_id))
        elif opcode == 'pregr':
            self.assign_reg(instr.dest, self.get_preg(instr.src.preg_id))
        elif opcode == 'pregw':
            self.assign_preg(instr.dest, self.get_reg(instr.src.reg_id))
        elif opcode == 'pregr_imm':
            self.assign_reg(instr.dest, self.get_preg(instr.src.preg_id))
        elif opcode == 'pregw_imm':
            self.assign_preg(instr.dest, self.get_reg(instr.src.reg_id))
        elif opcode == 'add':
            src1 = self.get_reg(instr.src1.reg_id)
            src2 = self.get_reg(instr.src2.reg_id)
            self.assign_reg(instr.dest, src1+src2)
        elif opcode == 'mul':
            src1 = self.get_reg(instr.src1.reg_id)
            src2 = self.get_reg(instr.src2.reg_id)
            self.assign_reg(instr.dest, src1*src2)
        elif opcode == 'sub':
            src1 = self.get_reg(instr.src1.reg_id)
            src2 = self.get_reg(instr.src2.reg_id)
            self.assign_reg(instr.dest, src1-src2)
        elif opcode == 'xor':
            src1 = self.get_reg(instr.src1.reg_id)
            src2 = self.get_reg(instr.src2.reg_id)
            self.assign_reg(instr.dest, float(int(src1)^int(src2)))
        elif opcode == 'and':
            src1 = self.get_reg(instr.src1.reg_id)
            src2 = self.get_reg(instr.src2.reg_id)
            self.assign_reg(instr.dest, float(int(src1)&int(src2)))
        elif opcode == 'or':
            src1 = self.get_reg(instr.src1.reg_id)
            src2 = self.get_reg(instr.src2.reg_id)
            self.assign_reg(instr.dest, float(int(src1)|int(src2)))
        elif opcode == 'cmp':
            src1 = self.get_reg(instr.src1.reg_id)
            src2 = self.get_reg(instr.src2.reg_id)
            self.next_cond['EQ'] = 1 if src1 == src2 else 0
            self.next_cond['NEQ'] = 1 if src1 != src2 else 0
            self.next_cond['GT'] = 1 if src1 > src2 else 0
            self.next_cond['LT'] = 1 if src1 < src2 else 0
            self.next_cond['GE'] = 1 if src1 >= src2 else 0
            self.next_cond['LE'] = 1 if src1 <= src2 else 0
        elif opcode == 'sqrt':
            src = self.get_reg(instr.src.reg_id)
            self.assign_reg(instr.dest, math.sqrt(src))
        elif opcode == 'inv':
            src = self.get_reg(instr.src.reg_id)
            self.assign_reg(instr.dest, 1./src)
        elif opcode == 'neg':
            src = self.get_reg(instr.src.reg_id)
            self.assign_reg(instr.dest, -src)
        elif opcode == 'pxid':
            x = self.get_reg(instr.src1.reg_id)
            y = self.get_reg(instr.src2.reg_id)
            self.assign_reg(instr.dest, 1 + y * self.block_size[0] + x + self.pe_id * self.block_size[0] * self.block_size[1] + 1000)
        elif opcode == 'lid':
            # note that this is only valid when in run_kernel mode
            mode = int(self.get_reg(instr.src.reg_id))
            self.assign_reg(instr.dest, self.current_lid[mode])
        elif opcode == 'grid':
            # note that this is only valid when in run_kernel mode
            mode = int(self.get_reg(instr.src.reg_id))
            self.assign_reg(instr.dest, self.current_grid[mode])
        elif opcode == 'slp':
            self.powerdown = True
        elif opcode == 'wku':
            self.powerdown = False
        elif opcode == 'nop':
            pass
        else:
            print 'warning: instruction %s not implemented, skipping line'%opcode

    def step2(self, instr):
        ''' Second step: execute scheduled instructions. '''
        for dest, value in self.register_assignments:
            self.regs[dest.reg_id] = float(value)
        self.register_assignments = []
        for dest, value in self.pregister_assignments:
            self.pregs[dest.preg_id] = float(value)
        self.pregister_assignments = []
        for dest_addr, value in self.memory_assignments:
            self.memory.set(int(dest_addr), float(value))
        self.memory_assignments = []
        self.cond = self.next_cond.copy()

    def get_output(self):
        return self.regs[self.OUT_REG_ID]

    def set_port(self, port, value):
        self.regs[port] = value

        
 

class Interpreter(object):
    ''' Interpreter simulating an array of processing elements.

    The interpreter is responsible for the management of Processing Elements,
    it takes care of image data i/o and the execution of instructions.
    In order to deal with the parallellism of the array, the execution of instructions
    is split in two Phases. First the actual instruction is executed for each PE,
    scheduling all memory, port and register assignments. In a second phase these
    assignments are actually executed.

    '''
    @staticmethod
    def copy_image_block(image, pos, block_size):
        rsize, csize = block_size
        sr = pos[0] * rsize
        sc = pos[1] * csize 
        er = (pos[0]+1) * rsize
        ec = (pos[1]+1) * csize
        return [[image[i][j] for j in xrange(sc, ec)] for i in xrange(sr, er)]

    def __init__(self, code, image, block_size, pe_nr_buffer = 4, pe_nr_reg = NR_REG, pe_nr_preg = NR_POINTER_REGS):
        ''' Initialise the interpreter and the processing elements.
        
        Arguments:
        code -- Code object with an initialised code generator (yielding Instruction objects)
        image -- Grayscale 2D array of image values, 
                 the size together with block_size determines the number of PE's
        block_size -- Size of each PE memory block
        pe_nr_buffer -- Number of block memory banks
        pe_nr_reg -- Number of registers for each PE

        '''
        
        self.temp = 0
        
        # execution
        self.code = code
        self.codegen = None
        self.pc = 0

        # i/o
        self.image = image
        self.image_size = (len(self.image), len(self.image[0]))

        # some constants
        self.NORTH_REG_ID = regname_to_id('north')
        self.EAST_REG_ID = regname_to_id('east')
        self.SOUTH_REG_ID = regname_to_id('south')
        self.WEST_REG_ID = regname_to_id('west')

        # config
        self.block_size = block_size
        self.pe_dim = [i/b for i,b in zip(self.image_size, self.block_size)] 

        # instantiate PE array
        self.procs = [[PE(self, j* self.pe_dim[0] + i, self.copy_image_block(image, (i, j), block_size), block_size, pe_nr_buffer, pe_nr_reg, pe_nr_preg) \
        for j in xrange(self.pe_dim[1])] for i in xrange(self.pe_dim[0])]

    def set_src_image(self, image, buffer_nr = 0):
        ''' Set a new source image for all PEs in the (first) block buffer. '''
        nheight, nwidth = len(image), len(image[0])
        if self.image: # if image was already defined, sizes should remain the same
            oheight, owidth = len(self.image), len(self.image[0])
            assert(nwidth == owidth and nheight == oheight)
        self.image = image
        for i, prow in enumerate(self.procs):
            for j, p in enumerate(prow):
                p.set_source_image(self.copy_image_block(image, (i, j), self.block_size), buffer_nr)

    def reset(self):
        self.codegen = self.code.gen(self.code)

    def set_all_id(self, local_x, local_y):
        swidth, sheight = self.block_size
        for i, row in enumerate(self.procs):
            for j, p in enumerate(row):
                p.current_lid = (local_y, local_x)
                p.current_grid = (i, j)

    def run_kernel(self):
        ''' Run a kernel, this simulates the automatic iteration of all output pixels. '''
        self.reset()
        swidth, sheight = self.block_size
        for i in xrange(sheight):
            for j in xrange(swidth):
                self.reset()
                for instr in self.codegen:
                    self.set_all_id(j, i)
                    self.step(instr)

    def run(self):
        ''' Run the program in self.codegen '''
        self.reset()
        for instr in self.codegen:
            self.step(instr)

    def step(self, instr):
        ''' Execute a single instruction '''
        # step 1
        for row in self.procs:
            for p in row:
                p.step1(instr)

        # step 2
        for row in self.procs:
            for p in row:
                p.step2(instr)
   
        # update neighbor ports
        for i, row in enumerate(self.procs):
            for j,p in enumerate(row):
                out = p.get_output()
                if i-1 >= 0:
                    self.procs[i-1][j].set_port(self.SOUTH_REG_ID, out)
                if j-1 >= 0:
                    self.procs[i][j-1].set_port(self.EAST_REG_ID, out)
                if i+1 < self.pe_dim[0]:
                    self.procs[i+1][j].set_port(self.NORTH_REG_ID, out)
                if j+1 < self.pe_dim[1]:
                    self.procs[i][j+1].set_port(self.WEST_REG_ID, out)
                 
    def get_output_buffer(self):
        bank_sel = 1
        width = self.block_size[1] * self.pe_dim[1]
        height = self.block_size[0] * self.pe_dim[0]
        offset = bank_sel * self.block_size[0] * self.block_size[1]        
        out_image = [[0 for i in xrange(width)] for j in xrange(height)]
        for i, row in enumerate(self.procs):
            for j, p in enumerate(row):
                for ii in xrange(self.block_size[0]):
                    for jj in xrange(self.block_size[1]):
                        v = p.memory.get(offset + ii*self.block_size[1] +jj)
                        out_image[i*self.block_size[1] + ii][j*self.block_size[0] + jj] = v      
        return out_image

    def gen_output_image(self, bank_sel = 1, scaling = True, clipping = True, float_out = False, doRGB = False):
        ''' Glue all pe memories together, bank_sel selects the memory buffer. '''
        width = self.block_size[1] * self.pe_dim[1]
        height = self.block_size[0] * self.pe_dim[0]
        offset = bank_sel * self.block_size[0] * self.block_size[1]
        doRGB = True
        if doRGB:
            out_image = [[0 for i in xrange(width*3)] for j in xrange(height)]
            colors = []
            print 'generating color map'
            '''print str(len(out_image[0])) + ' x ' + str(len(out_image))'''
            for i, row in enumerate(self.procs):
                for j, p in enumerate(row):
                    for ii in xrange(self.block_size[0]):
                        for jj in xrange(self.block_size[1]):
                            v = p.memory.get(offset + ii*self.block_size[1] +jj)
                            if v != 0 and colors.count(v) == 0:
                                colors.append(v)
            self.temp = str(len(colors))
            print 'colors = ' + self.temp
            for i, row in enumerate(self.procs):
                for j, p in enumerate(row):
                    for ii in xrange(self.block_size[0]):
                        for jj in xrange(self.block_size[1]):
                            v = p.memory.get(offset + ii*self.block_size[1] +jj)
                            if v != 0:
                                c = ucharRgb(colors.index(v),0,len(colors))
                                out_image[i*self.block_size[1] + ii][(j*self.block_size[0] + jj)*3] = c[0]
                                out_image[i*self.block_size[1] + ii][(j*self.block_size[0] + jj)*3+1] = c[1]
                                out_image[i*self.block_size[1] + ii][(j*self.block_size[0] + jj)*3+2] = c[2]		  	

		'''else:
		    # XXX image scaling hack
		    if scaling:
		        print 'scaling enabled'
		        minv = 9999999
		        maxv = -999999
		        for i, row in enumerate(self.procs):
		            for j, p in enumerate(row):
		                for ii in xrange(self.block_size[0]):
		                    for jj in xrange(self.block_size[1]):
		                         v = p.memory.get(offset + ii*self.block_size[1] +jj)
		                         minv = min(minv, v)
		                         maxv = max(maxv, v)
		        vrange = maxv-minv
		        voffset = -minv
		        vscale = 255./vrange if not vrange == 0 else 1.
		        print 'minv %f maxv %f'%(minv, maxv)
		    else:
		        voffset = 0.
		        vscale  = 1.

		    out_image = [[0 for i in xrange(width)] for j in xrange(height)]
		    for i, row in enumerate(self.procs):
		        for j, p in enumerate(row):
		            for ii in xrange(self.block_size[0]):
		                for jj in xrange(self.block_size[1]):
		                    v = p.memory.get(offset + ii*self.block_size[1] +jj)
		                    v = (v + voffset)*vscale
                            if not float_out: 
	                            v = int(v)
                            if v > 255 and clipping:
	                            print 'clip v = %i to 255'%v
	                            v = 255
		                    out_image[i*self.block_size[1] + ii][j*self.block_size[0] + jj] = v'''
        return out_image

def main(block_size, code_gen, args, image_filename, output_filename):
    # load image
    image = imageio.read(image_filename)

    #print image[0][0]

    if not image:
        print 'could not read image, continuing with empty 128x128 image'
        image = [[0 for i in xrange(128)] for j in xrange(128)]

    code = Code()
    code.set_generator(code_gen, block_size, args)

    interpreter = Interpreter(code, image, block_size)
    interpreter.run()

    doRGB = True
    out_image = interpreter.gen_output_image(1, True, True, False, doRGB)
    if doRGB:
        imageio.write(output_filename, out_image, 3)
    else:
	    imageio.write(output_filename, out_image, 1)
    
    '''f = open('curr_codegen.txt', 'w')
    f.write(str(code))'''

    '''print len(code)'''
    
    return interpreter


if __name__ == '__main__':
    # argument parsing
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-b', '--block_size', dest='block_size', default=64, help='PE block size')
    parser.add_option('-s', '--src_image', dest='src_image', default='data/lena.png', help='source image')
    parser.add_option('-o', '--output', dest='output', default='interpreter_out.png', help='output prefix')
    parser.add_option('--codegen', dest='codegen_impl', default='blip.blipcode.gen_conv',\
                      help='override the codegen implementation')
    parser.add_option('-a', '--codegen_args', dest='codegen_args', default="{'coeff':[[-1,0,1]]*3}",\
                      help='codegen arguments, evaluated as dict')
    (options, args) = parser.parse_args()
    block_size = (int(options.block_size), int(options.block_size))

    # codegen arguments processing
    # XXX hack
    try:
        codegen_args = eval(options.codegen_args)
        if not isinstance(codegen_args, dict):
            raise TypeError()
    except:
        print 'error: unable to parse the codegen argument'
        print '       please enclose the dict in double quotes'
        print '       raw args: >>>%s<<<'%options.codegen_args
        exit(1)

    # load codegen
    from blip.code.codegen import load_codegen, get_codegen_parameters # wrap_codegen commented, not in that file!
    from blip.code.codegen import InvalidCodegenArgumentsException

    module_name, _, codegen_name = options.codegen_impl.rpartition('.')
    codegen_impl = load_codegen(module_name, codegen_name)

    if not codegen_impl:
        print 'error: could not load codegen %s'%codegen_impl
        exit(1)

    # finally execute function
    main(block_size, codegen_impl, codegen_args, options.src_image, options.output)

    print 'interpreter finished'

