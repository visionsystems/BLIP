from __future__ import with_statement
''' Code object and utility functions for flexible usage of codegen functions. '''

from blip.simulator.opcodes import NR_REG, NR_POINTER_REGS, regname_to_id, tag_instr, instr_class_from_opcode, Instruction

# Exceptions
class InvalidCodegenArgumentsException(Exception): pass
class RegisterAllocationExeption(Exception): pass
class RegisterReleaseExeption(Exception): pass

class PointerRegister(object):
    def __init__(self, id_):
        self.id = id_
        self.str_rep = ''
        try:
            self.str_rep = 'pr%i'%self.id
        except:
            self.str_rep = str(self.id)
        self.preg_id = regname_to_id(self.str_rep)
    def __str__(self):
        return self.str_rep

class Register(object):
    def __init__(self, id_):
        self.id = id_
        self.str_rep = ''
        try:
            self.str_rep = 'r%i'%self.id
        except:
            self.str_rep = str(self.id)
        self.reg_id = regname_to_id(self.str_rep)
    def __str__(self):
        return self.str_rep

class Port(object):
    def __init__(self, ddir):
        self.ddir = ddir
        self.reg_id = regname_to_id(ddir)
    def __str__(self):
        return self.ddir

class Code(object):
    def __init__(self, nr_regs = NR_REG, nr_pregs = NR_POINTER_REGS):
        self.gen = None
        self.regs = [Register(i) for i in xrange(nr_regs)]
        self.pregs = [PointerRegister(i) for i in xrange(nr_pregs)]
        self.used_regs = dict((i, False) for i in xrange(nr_regs))
        self.used_pregs = dict((i, False) for i in xrange(nr_pregs))
        self.north = Port('north')
        self.east = Port('east')
        self.south = Port('south')
        self.west = Port('west')
        self.out = Register('out')
    def set_generator(self, gen, block_size, args = {}):
        def codegen_wrapper(code): return gen(code, block_size, args)
        self.gen = codegen_wrapper
    def __str__(self):
        if not self.gen: return ''
        res = '\n'.join(['%4i: %s'%(i,str(instr)) for i, instr in enumerate(self.gen(self))])
        return 'Code:\n%s'%res
    def __len__(self):
        if not self.gen: return 0
        length = 0
        for i, instr in enumerate(self.gen(self)):
            length += 1
        return length
        
    # register
    def r(self, nr):
        return self.regs[nr]
    def alloc_reg(self):
        for r in self.regs:
            rid = r.id
            if not self.used_regs[rid]:
                self.used_regs[rid] = True
                return r
        raise RegisterAllocationException('no free registers')
    def alloc_preg(self):
        for pr in self.pregs:
            prid = pr.id
            if not self.used_pregs[prid]:
                self.used_pregs[prid] = True
                return pr
        raise RegisterAllocationException('no free pointer registers')
    def release_reg(self, reg):
        rid = reg.id
        if not self.used_regs[rid]:
            raise RegisterReleaseException('error: freeing free reg %i'%rid)
        self.used_regs[rid] =  False
    def release_preg(self, preg):
        prid = preg.id
        if not self.used_pregs[prid]:
            raise RegisterReleaseException('error: freeing free pointer reg %i'%prid)
        self.used_pregs[prid] =  False
    def nr_regs_free(self):
        return sum(1 for k,v in self.used_regs.iteritems() if not v)
    def nr_pregs_free(self):
        return sum(1 for k,v in self.used_pregs.iteritems() if not v)
    def set_allocator_hooks(alloc_hook, release_hook):
        self.alloc_hook = alloc_hook
        self.release_hook = release_hook
    def instr_size(self):
        if not self.gen: return 0
        g = self.gen(self)
        cnt = 0
        for x in g: cnt+=1
        return cnt
    def communication_overhead(self):
        if not self.gen: return 0
        g = self.gen(self)
        comm_overhead = 0
        for x in g:
            if hasattr(x,'tag') and x.tag == 'communication overhead':
                comm_overhead += 1
        return comm_overhead
    def tag_com_overhead_instr(self, instr):
        tag_instr(instr, 'communication overhead')
    def plain_instr(self):
        if not self.gen: return []
        return [x for x in self.gen(self)]

# Actual code
def load_function(module_name, function_name):
	''' Load functions from modules. '''
	func = None
	try:
		module_path = module_name.split('.')
		mod = __import__(module_name)
		for m in module_path[1:]:
			mod = getattr(mod, m)
		func = getattr(mod, function_name)
	except:
		pass
	return func

def get_codegen_parameters(function, omit_standard_args = False):
	''' Inspect parameters of a codegen function. '''
	from inspect import getargspec
	args = getargspec(function)[0]

	standard_args = ['code', 'block_size', 'args']
	if omit_standard_args:
		return [x for x in args if not x in standard_args]
	else:
		return args

def is_valid_codegen(function):
	''' Check if function has the required arguments and returns a generator. '''
	args = get_codegen_parameters(function)
	if not all((x in args) for x in ['code', 'block_size', 'args']): return False

	# as long as we don't iterate over the generator
	# no code is called so the arguments can be faked
	code = None

	# this function is not implemented in 2.5
	def isgenerator(function): return hasattr(function, 'next')

	if not isgenerator(function(code, (4, 4), {})): return False
	return True

def load_codegen(module_name, function_name, no_validation = False):
	''' Load codegen functions from modules. '''
	func = load_function(module_name, function_name)
	if not func: return None

	if no_validation: return func 
	return func if is_valid_codegen(func) else None

class scoped_alloc(object):
	''' Automatic handling of scoped register allocation/freeing. '''
	def __init__(self, code, nr):
		self.code = code
		self.nr = nr
		self.regs = []
	def __enter__(self):
		self.regs = [self.code.alloc_reg() for x in xrange(self.nr)]
		if self.nr == 1:
			return self.regs[0]
		else:
			return self.regs
	def __exit__(self, *args):
		for r in self.regs:
			self.code.release_reg(r)

class scoped_preg_alloc(object):
	''' Automatic handling of scoped register allocation/freeing. '''
	def __init__(self, code, nr = NR_POINTER_REGS):
		self.code = code
		self.nr = nr
		self.pregs = []
	def __enter__(self):
		self.pregs = [self.code.alloc_preg() for x in xrange(self.nr)]
		if self.nr == 1:
			return self.pregs[0]
		else:
			return self.pregs
	def __exit__(self, *args):
		for pr in self.pregs:
			self.code.release_preg(pr)

def instr_compact_repr(instr):
		''' XXX Hackish code to convert instruction using a general method. '''
		args = []
		if instr.opcode() == 'imm':
			args = [str(instr.dest), instr.value]
		else:
			try:
				args.append(instr.dest)
			except AttributeError: pass
			# use src2 to detect the instruction format
			src2 = None
			try:
				src2 = instr.src2
			except AttributeError: pass
			if src2:
				try:
					args.append(instr.src1)
					args.append(instr.src2)
				except AttributeError: pass
			else:
				try:
					args.append(instr.src)
				except AttributeError: pass
		return [instr.opcode(), instr.cond()] +  [str(x) if not type(x) in (float, int) else x for x in args]

def convert_code_to_compact_repr(code):
	'''  Convert between original object based format and compact compiler repr. '''
	return [instr_compact_repr(instr) for instr in code]

class InstrAdapter(Instruction):
	def __init__(self, instr, str_f = None, **kwargs):
		self.instr = instr
		self.str_f = str_f
		self.use_reg_wrapper = 'use_reg_wrapper' in kwargs and kwargs['use_reg_wrapper']
		self.field_map = {'dest':2, 'src':3, 'src1':3, 'src2':4, 'value':3}
		self.comp_field_map = {'src1':2, 'src2':3}
	def __getattr__(self, name):
		op = self.instr[0]
		try:
			if name in ['dest', 'src', 'src1', 'src2', 'value']:
				field_nr = self.field_map[name] if op != 'cmp' else self.comp_field_map[name]
				field = self.instr[field_nr]
				if self.use_reg_wrapper and name != 'value':
					return Register(field)
				else:
					return field
			elif name == '_cond':
				return self.instr[1]
			elif name == '_opcode':
				return op
		except Exception, e:
			raise AttributeError('failed to fetch attribute %s, error:%s'%(name, str(e)))
	def __setattr__(self, name, value):
		if name in ['dest']:
			self.instr[2] = value
		elif name in ['src', 'src1', 'value']:
			self.instr[3] = value
		elif name in ['src2']:
			self.instr[4] = value
		else:
			object.__setattr__(self, name, value)
		
	def cond_str(self):
        	return '{%s} '%self.cond() if self.cond() else ''
	def __str__(self):
		return '%s %s %s'%(self.instr[0], self.cond_str(), ' '.join(str(x) for x in self.instr[2:]))

def convert_compact_repr_to_obj(compact_code):
	def cvt(instr):
		# retrieving the class would be usefull for implementing
		# __str__
		#cls = instr_class_from_opcode(instr[0])
		instr_obj = InstrAdapter(instr)
		return instr_obj
	return [cvt(instr) for instr in compact_code]

if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	(options, args) = parser.parse_args()

