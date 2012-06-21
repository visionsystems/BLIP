# SETTINGS
NR_REG = 20 # 8
SPECIAL_REGS = ['north', 'east', 'south', 'west', 'out']
NR_POINTER_REGS = 2048

# UTILITY FUNCTIONS
def regname_to_id(regname):
    ''' This function calculates the port or register id by its name. There are 4 ports (north, east, south, west) and 9 registers (out, r0 - r7). The id's arr given in this sequence 0 - 12. '''
    nr_other_regs = len(SPECIAL_REGS)
    reg_lookup = {}
    for i, x in enumerate(SPECIAL_REGS):
        reg_lookup[x] = i
    if regname in reg_lookup:
        return reg_lookup[regname]
    elif regname[0].lower() == 'r':
        return nr_other_regs + int(regname[1:])
    elif regname[0].lower() == 'p' and regname[1].lower() == 'r':
        return nr_other_regs + NR_REG + int(regname[2:])
    else:
        raise Exception('Invalid register name %s'%regname)

def tag_instr(instr, tag):
    if not hasattr(instr, 'tag'): instr.tag = []
    instr.tag.append(tag)
    return instr


# INSTRUCTIONS
class Instruction(object):
    def __init__(self, opcode, cond = None):
        self._opcode = opcode
        self._cond = cond
    def opcode(self):
        return self._opcode
    def cond(self):
        return self._cond
    def cond_str(self):
        return '{%s} '%self._cond if self._cond else ''
    def add_tag(self, tag):
        if not hasattr(self, 'tag'):
            self.tag = [tag]
        else:
            self.tag.append(tag)

class Pxid(Instruction):
    ''' instruction that calculates a unique value for every pixel Px(src1,src2) in the image starting from 1'''
    def __init__(self, dest, src1, src2, cond = None):
        Instruction.__init__(self, 'pxid', cond)
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
    def __str__(self):
        return 'pxid%s %s, %s, %s'%(self.cond_str(), str(self.dest),str(self.src1), str(self.src2))

class Lid(Instruction):
    ''' '''
    def __init__(self, dest, mode, cond = None):
        Instruction.__init__(self, 'lid', cond)
        self.dest = dest
        self.src = mode # store mode in src property
    def __str__(self):
        return 'lid%s %s %s'%(self.cond_str(), str(self.dest), str(self.src))

class Glid(Instruction):
    ''' '''
    def __init__(self, dest, mode, cond = None):
        Instruction.__init__(self, 'glid', cond)
        self.dest = dest
        self.src = mode
    def __str__(self):
        return 'glid%s %s %s'%(self.cond_str(), str(self.dest), str(self.src))

class Grid(Instruction):
    ''' '''
    def __init__(self, dest, mode, cond = None):
        Instruction.__init__(self, 'grid', cond)
        self.dest = dest
        self.src = mode
    def __str__(self):
        return 'grid%s %s %s'%(self.cond_str(), str(self.dest), str(self.src))

class Cmp(Instruction):
    ''' '''
    def __init__(self, src1, src2, cond = None):
        Instruction.__init__(self, 'cmp', cond)
        self.src1 = src1
        self.src2 = src2
    def __str__(self):
        return 'cmp%s %s %s'%(self.cond_str(), str(self.src1), str(self.src2))

class Imm(Instruction):
    ''' '''
    def __init__(self, dest, value, cond = None):
        Instruction.__init__(self, 'imm', cond)
        self.dest = dest
        self.value = value
    def __str__(self):
        return 'imm%s %s, %i'%(self.cond_str(), str(self.dest), self.value)

# data xfer
class Mov(Instruction):
    ''' move data from src to dest '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'mov', cond)
        self.dest = dest
        self.src = src
    def __str__(self):
        return 'mov%s %s, %s'%(self.cond_str(), str(self.dest),str(self.src))

class PRegR(Instruction):
    ''' read from pointer register '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'pregr', cond)
        self.dest = dest
        self.src = src
    def __str__(self):
        return 'pregr%s %s, %i'%(self.cond_str(), str(self.dest), str(self.src))
    
class PRegW(Instruction):
    ''' write to pointer register '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'pregw', cond)
        self.dest = dest
        self.src = src
    def __str__(self):
        return 'pregw%s %s, %i'%(self.cond_str(), str(self.dest), str(self.src))

class PRegRImm(Instruction):
    ''' read from pointer register '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'pregr_imm', cond)
        self.dest = dest
        self.src = int(src)
    def __str__(self):
        return 'pregr_imm%s %s, %i'%(self.cond_str(), str(self.dest), str(self.src))
    
class PRegWImm(Instruction):
    ''' write to pointer register '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'pregw_imm', cond)
        self.dest = int(dest)
        self.src = src
    def __str__(self):
        return 'pregw_imm%s %s, %i'%(self.cond_str(), str(self.dest), str(self.src))
    
class MemR(Instruction):
    ''' memory read from register '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'memr', cond)
        self.dest = dest
        self.src = src
    def __str__(self):
        return 'mov%s %s, mem[%s]'%(self.cond_str(), str(self.dest),str(self.src))

class MemRImm(Instruction):
    ''' memory read with the address directly encoded into the instruction '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'memr_imm', cond)
        self.dest = dest
        self.src = int(src) # only int arguments
    def __str__(self):
        return 'mov%s %s, mem[%s]'%(self.cond_str(), str(self.dest),str(self.src))

class MemW(Instruction):
    ''' memory write from register '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'memw', cond)
        self.dest = dest
        self.src = src
    def __str__(self):
        return 'mov%s mem[%s], %s'%(self.cond_str(), str(self.dest),str(self.src))

class MemWImm(Instruction):
    ''' memory write with the address directly encoded into the instruction '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'memw_imm', cond)
        self.dest = int(dest) # only int arguments
        self.src = src
    def __str__(self):
        return 'mov%s mem[%s], %s'%(self.cond_str(), str(self.dest),str(self.src))

# arithmetic
class Add(Instruction):
    ''' add src1 to src2 and store result in dest '''
    def __init__(self, dest, src1, src2, cond = None):
        Instruction.__init__(self, 'add', cond)
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
    def __str__(self):
        return 'add%s %s, %s, %s'%(self.cond_str(), str(self.dest),str(self.src1), str(self.src2))

class Sub(Instruction):
    ''' substract src2 from src1 and store in dest (to be confirmed) '''
    def __init__(self, dest, src1, src2, cond = None):
        Instruction.__init__(self, 'sub', cond)
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
    def __str__(self):
        return 'sub%s %s, %s, %s'%(self.cond_str(), str(self.dest),str(self.src1), str(self.src2))

class Mul(Instruction):
    ''' multiply src1 and src2 and store in dest '''
    def __init__(self, dest, src1, src2, cond = None):
        Instruction.__init__(self, 'mul', cond)
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
    def __str__(self):
        return 'mul%s %s, %s, %s'%(self.cond_str(), str(self.dest),str(self.src1), str(self.src2))

# logic
class Xor(Instruction):
    ''' '''
    def __init__(self, dest, src1, src2, cond = None):
        Instruction.__init__(self, 'xor', cond)
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
    def __str__(self):
        return 'xor%s %s, %s, %s'%(self.cond_str(), str(self.dest),str(self.src1), str(self.src2))

class And(Instruction):
    ''' '''
    def __init__(self, dest, src1, src2, cond = None):
        Instruction.__init__(self, 'and', cond)
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
    def __str__(self):
        return 'and%s %s, %s, %s'%(self.cond_str(), str(self.dest),str(self.src1), str(self.src2))

class Or(Instruction):
    ''' '''
    def __init__(self, dest, src1, src2, cond = None):
        Instruction.__init__(self, 'or', cond)
        self.dest = dest
        self.src1 = src1
        self.src2 = src2
    def __str__(self):
        return 'or%s %s, %s, %s'%(self.cond_str(), str(self.dest),str(self.src1), str(self.src2))

class Sqrt(Instruction):
    ''' '''
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'sqrt', cond)
        self.dest = dest
        self.src = src
    def __str__(self):
        return 'sqrt%s %s, %s'%(self.cond_str(), str(self.dest), str(self.src))

class Inv(Instruction):
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'inv', cond)
        self.dest = dest
        self.src = src
    def __str__(self):
        return 'inv%s %s, %s'%(self.cond_str(), str(self.dest), str(self.src))

class Neg(Instruction):
    def __init__(self, dest, src, cond = None):
        Instruction.__init__(self, 'neg', cond)
        self.dest = dest
        self.src = src
    def __str__(self):
        return 'neg%s %s, %s'%(self.cond_str(), str(self.dest), str(self.src))

class Nop(Instruction):
    ''' '''
    def __init__(self, cond = None):
        Instruction.__init__(self, 'nop', cond)
    def __str__(self):
        return 'nop'

class Sleep(Instruction):
    ''' '''
    def __init__(self, cond = None):
        Instruction.__init__(self, 'slp', cond)
    def __str__(self):
        return 'slp'

class WakeUp(Instruction):
    ''' '''
    def __init__(self, cond = None):
        Instruction.__init__(self, 'wku', cond)
    def __str__(self):
        return 'wku'

    
def instr_class_from_opcode(opcode):
    def is_valid_instr_cls(cls):
        try:
            return issubclass(cls, Instruction) and cls != Instruction 
        except:
            return False
    instruction_map = dict((name.lower(), v) for name, v in globals().items() if is_valid_instr_cls(v))
    opcode = opcode.lower().replace('_', '')
    if opcode in instruction_map:
        return instruction_map[opcode]
    return None

if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	(options, args) = parser.parse_args()

