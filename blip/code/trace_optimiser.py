from __future__ import with_statement
from itertools import izip

from blip.simulator.opcodes import Mov, MemRImm, tag_instr, Imm
from blip.code.codegen import Code, scoped_alloc, convert_code_to_compact_repr, instr_compact_repr
from blip.code.BlipCompiler import Compiler, RegisterAllocator

class NotImplementedException(Exception): pass


class TraceFragment(object):
	''' Represents a fragment of the total instruction trace. '''
	def __init__(self, code, trace, reg_used):
		self.code = code
		# needs to copied because this trace is modified.
		self.trace = trace[:]
		self.compact_trace = convert_code_to_compact_repr(self.trace)
		self.reg_used = reg_used[:]
		self.ptr = 0
	def __len__(self):
		return len(self.trace)
	def _set_reg_used(self, reg_id, instr_nr, value=True):
		self.reg_used[instr_nr][reg_id] = value
	def _free_regs(self):
		reg_used_in_w = []
		for i, regs in enumerate(self.reg_used):
			for j, used in enumerate(regs):
				if used: reg_used_in_w.append(j)
		return [r for r in self.code.regs if not r.id in reg_used_in_w]
	def insert_instr_before(self, instr):
		''' Insert instruction before ptr, ptr points at operation before insert. '''
		instr_reg_used  = self.reg_used[self.ptr]
		self.trace.insert(self.ptr, instr)
		self.reg_used.insert(self.ptr, [x for x in instr_reg_used])
		self.ptr += 1
	def insert_instr_after(self, instr):
		''' Insert instruction after ptr, ptr points at operation before insert. '''
		instr_reg_used  = self.reg_used[self.ptr]
		self.trace.insert(self.ptr+1, instr)
		self.reg_used.insert(self.ptr+1, [x for x in instr_reg_used])
	def alloc_local_reg(self):
		''' Allocate a register local to the trace fragment. '''
		current_free_regs = self._free_regs()
		if len(current_free_regs) < 1:
			raise Exception('no more free regs')
		reg = current_free_regs[0]
		# update usage
		for i in xrange(self.ptr, len(self.trace)):
			self._set_reg_used(reg.id, i, True)
		return reg
	def release_local_reg(self, reg):
		''' Free a register local to the trace fragment. '''
		if reg in self._free_regs():
			raise Exception('freeing unallocated reg')
		for i in xrange(self.ptr, len(self.trace)):
			self._set_reg_used(reg.id, i, False)
	def set_ptr(self, location):
		''' Set the instruction pointer. '''
		self.ptr = location
	def inc_ptr(self):
		''' Increment the instruction pointer. '''
		self.set_ptr(self.ptr + 1)
	def dec_ptr(self):
		''' Decrement the instruction pointer. '''
		self.set_ptr(self.ptr - 1)
	def current_instr(self):
		''' Dereference the instruction pointer. '''
		return self.trace[self.ptr]
	def remove_current_instr(self):
		''' Remove an instruction, the pointer points at the next instruction. '''
		del self.trace[self.ptr]
		del self.reg_used[self.ptr]
	def nr_free_regs(self):
		''' Return the number of register that are free during the whole trace fragment. '''
		reg_used_in_w = []
		for i, regs in enumerate(self.reg_used):
			for j, used in enumerate(regs):
				if used: reg_used_in_w.append(j)
		return sum(1 for r in self.code.regs if not r.id in reg_used_in_w)

class VariableManager(object):
	def __init__(self):
		self.variables = {}
		self.max_variables_id = {}
		self.cnt = 0
	@classmethod
	def gen_variable_name(cls, name, version):
		return '%s@%i'%(name, version)
	@classmethod
	def symbolic_name(cls, name):
		name, _, _ = name.partition('@')
		return name
	@classmethod
	def has_symbolic_name(cls, name_var, symbol_name):
		return cls.symbolic_name(name_var.name) == symbol_name
	@classmethod
	def name_version(cls, name):
		_, _, version = name.partition('@')
		return int(version)
	def def_var(self, var, seq_value=False):
		version = 0
		if var in self.max_variables_id:
			version = self.max_variables_id[var] + 1

		self.max_variables_id[var] = version
		new_var = self.gen_variable_name(var, version)
		self.variables[var] = new_var
		return new_var
	def use_var(self, var, seq_value = False):
		if var in self.variables:
			return self.variables[var]
		return None
	def get_tmp_name(self):
		name = 'tmp_%i'%self.cnt
		self.cnt += 1
		return name
	@classmethod
	def get_def_use(cls, instr):
		''' Determine defines and uses for current instruction. '''
		def_ = None
		uses = []
		op = instr[0]
		# skip these instructions, they don't use registers
		if op in ['nop']:
			return None, []
		regs = instr[2:]
		# specialcase cmp etc., their first reg is not a def reg
		use_regs = []
		if op in ['cmp', 'memw']:
			uses = regs
		elif op in ['memw_imm']:
			uses = regs[1:]
		elif op in ['memr_imm', 'imm']:
			def_ = regs[0]
		elif len(regs):
			if regs[0] in ['north', 'east', 'south', 'west']:
				return None, []
			if not op in ['memw_imm']:
				def_ = regs[0]
			uses = regs[1:]
		return def_, uses

class CodeFragment(object):
	''' Code fragment, containing the instructions and allocation information. '''
	def __init__(self, instr = [], regs_used = None, name = ''):
		if len(instr) and regs_used is not None:
			assert len(instr) == len(regs_used)
		self.instructions = instr
		self.regs_used = [None for _ in self.instructions] if not regs_used else regs_used
		self.name = name
	def append(self, instr, reg_usage = None):
		''' Add a new instruction and the corresponding allocation info. '''
		self.instructions.append(instr)
		self.regs_used.append(reg_usage)
	def clear(self):
		''' Clear all instructions and allocation info. '''
		self.instructions = []
		self.regs_used = []
	def __iter__(self):
		''' Iterate over each (instruction, reg usage pair). '''
		return  iter(izip(self.instructions, self.regs_used))
	def __len__(self):
		''' Number of instructions. '''
		return len(self.instructions)

class CodegenToSSAConvertor(object):
	invert_condition = {'EQ': 'NEQ', 'NEQ':'EQ', 'LT':'GE', 'GE':'LT', 'LE':'GT', 'GT':'LE'}
	def __init__(self, fragment_len, use_allocation_info = False):
		self.fragment_len = fragment_len
		self.use_allocation_info = use_allocation_info

		self.def_hist = {}
		self.var_manager = VariableManager()
		self.last_cmp_ind = -1
	def gen_ssa_fragments(self, code_gen, code, block_size, args):
		''' Generate code fragments in ssa format, while trying to split at cmp instructions. '''
		self.last_cmp_ind = -1
		self.def_hist = {}
		self.var_manager = VariableManager()
		source_fragment = CodeFragment([], [], 'source')
		no_cmp_in_current_buffer = True
		# Run code generator, converting the instructions, one fragment at a time
		for i, instr in enumerate(code_gen(code, block_size, args)):
			th_reached = i > 0 and len(source_fragment)%self.fragment_len == 0
			if instr.opcode() == 'cmp':
				# Start new fragment from comp, split it in parts if it becomes larger then fragment_len
				no_cmp_in_current_buffer = False # from now on ssa is converted in blocks starting with cmp
				new_ssa_code = self._convert_code_to_ssa_adapter(source_fragment, i)
				result_fragment = CodeFragment([], [], 'result')
				for ind, (new_instr, new_regs_used) in enumerate(new_ssa_code):
					if ind > 0 and  ind%self.fragment_len == 0:
						yield result_fragment
						result_fragment.clear()
					result_fragment.append(new_instr, new_regs_used)
				# yield final instructions
				if len(result_fragment):
					yield result_fragment
				source_fragment.clear()
			elif th_reached and no_cmp_in_current_buffer:
				# If the length treshold is reached and there was no cmp instruction,
				# convert this fragment
				yield self._convert_code_to_ssa_adapter(source_fragment, i)
				source_fragment.clear()
			# Always add the new instruction to the source fragment
			source_fragment.append(instr, self._current_reg_usage(code) if self.use_allocation_info else None)
		# Convert anything left in the buffer
		yield self._convert_code_to_ssa_adapter(source_fragment, i)
	@classmethod
	def _current_reg_usage(cls, code):
		''' Get register usage from original non SSA code. '''
		return [code.used_regs[r.id] for r in code.regs]
	def _convert_code_to_ssa_adapter(self, source_fragment, code_start_ind = 0, regs_used_buffer = []):
		''' First convert code to compiler format before executing _convert_code_to_ssa. '''
		code_buffer, regs_used_buffer = source_fragment.instructions, source_fragment.regs_used
		res_code, res_usage = self._convert_code_to_ssa(convert_code_to_compact_repr(code_buffer), code_start_ind, regs_used_buffer)
		return CodeFragment(res_code, res_usage)
	def _convert_code_to_ssa(self, compact_code, code_start_ind = 0, regs_used_buffer = []):
		''' Reconstruct SSA code from binary instructions. '''
		# first find all possible phi results, these are identified
		# by finding all uses and evaluating the different paths
		# resulting in this used variable
		self.def_hist = {} # XXX unsure if this should be reset for each conversion
		phi_nodes = []

		if self.use_allocation_info:
			assert(len(compact_code) == len(regs_used_buffer))

		# Determine all phi nodes
		for i, instr in enumerate(compact_code):
			def_, uses = self.var_manager.get_def_use(instr)
			opcode = instr[0]
			# update position of last cmp instruction
			# this is the lower bottom index of the
			# defs that have an influence in this part
			# XXX not entirely correct
			if opcode == 'cmp':
				self.last_cmp_ind = i
				self.def_hist = {}
			# record defs
			if def_:
				if not def_ in self.def_hist: self.def_hist[def_] = []
				self.def_hist[def_].append(i)
			# trace back all use vars
			for use in uses:
				if use in self.def_hist:
					valid_defs = []
					# there are multiple defines for the value used
					# find the last define without cond and all with masks
					max_i = -1
					last_def = None
					for d in self.def_hist[use]:
						if d <= self.last_cmp_ind: continue # skip defs from previous cmp
						if compact_code[d][1]:
							valid_defs.append(d)
						elif d > max_i:
							max_i = d
					if max_i >= 0:
						valid_defs.append(max_i)
					valid_defs = set(valid_defs)
					if len(valid_defs) < 2:
						continue
					elif len(valid_defs) != 2:
						raise NotImplementedException('only two elements are allowed right now')
					phi_nodes.append((use, valid_defs))

		# Convert to SSA representation
		ssa_code = []
		# note that the comp block should be inherited from the previous
		# trace fragment as cmp.. conditionals can cross loop bounds
		self.def_hist = {}
		for i, instr in enumerate(compact_code):
			def_, uses = self.var_manager.get_def_use(instr)
			opcode = instr[0]
			cond = instr[1]
			if opcode == 'cmp':
				if cond:
					raise NotImplementedException('conditions on cmp instructions are not yet supported')
			for use in uses:
				new_use = self.var_manager.use_var(use)
				# replace variables with ssa var
				start_use = 3 if def_ else 2
				for i, op in enumerate(instr[start_use:]):
					if op == use:
						instr[start_use+i] = new_use
			# def always after use!
			if def_:
				if not def_ in self.def_hist: self.def_hist[def_] = i
				new_def = self.var_manager.def_var(def_)
				instr[2] = new_def
			ssa_code.append(instr)

		# Insert phi nodes
		# Note that this needs to be done after SSA conversion
		# to make sure that the phi node captures the correct version of a variable
		# when both parts of the phi node are some instructions apart
		for target, (v1, v2) in phi_nodes:
			if ssa_code[v1][1]: v1, v2 = v2, v1
			cond1, cond2 = ssa_code[v1][1], ssa_code[v2][1]
			orig_dest1, orig_dest2 = ssa_code[v1][2], ssa_code[v2][2]
			if cond1 and cond2:
				if self.invert_condition[cond1] != cond2:
					raise Exception('if both have a condition, they should have complimentary conditions')
			if ssa_code[v1][0] == ssa_code[v2][0] == 'mov':
				ssa_code[v2] = ['phi', ssa_code[v2][1], ssa_code[v2][2], ssa_code[v2][3], ssa_code[v1][3]]
				ssa_code[v1][0] = 'kill'
			else:
				# if the instructions are not movs, some more work is needed.
				cond = ssa_code[v2][1]
				phi_dest = ssa_code[v2][2]
				ssa_code[v2][1] = ssa_code[v1][1] = None
				dest1, dest2 = [self.var_manager.get_tmp_name() for _ in xrange(2)]
				ssa_code[v1][2] = dest1
				ssa_code[v2][2] = dest2
				# Hack: append phi node as extra argument to instr,
				# these will be expanded after all phi nodes are handled (position dependant)
				attach_to = max(v1, v2)
				phi_dest = orig_dest1 if v1 == attach_to else orig_dest2
				ssa_code[attach_to] += [['phi', cond, phi_dest, dest2, dest1]]

		# Expand phi nodes
		# Note that the allocation info is in sync up until now
		expanded_ssa_code = []
		for instr in ssa_code:
			try:
				if instr[-1][0] == 'phi':
					# expand embedded phi node if needed
					expanded_ssa_code.append(instr[:-1])
					expanded_ssa_code.append(instr[-1])
				else:
					expanded_ssa_code.append(instr)
			except (IndexError, TypeError), e:
				expanded_ssa_code.append(instr)
		# Finally remove the redundant instructions
		result_code = [x for x in expanded_ssa_code if not x[0] == 'kill']

		# If we are using allocation info, also emit this info
		expanded_regs_used_buffer = []
		if self.use_allocation_info:
			assert len(ssa_code) == len(regs_used_buffer), '%i %i'%(len(ssa_code), len(regs_used_buffer))
			for instr, regs_used in izip(ssa_code, regs_used_buffer):
				try:
					if instr[-1][0] == 'phi':
						# expand embedded phi node if needed
						for _ in xrange(2):
							expanded_regs_used_buffer.append(regs_used)
					else:
						expanded_regs_used_buffer.append(regs_used)
				except (IndexError, TypeError), e:
					expanded_regs_used_buffer.append(regs_used)
			result_use = [ru for instr, ru in izip(expanded_ssa_code, expanded_regs_used_buffer) if not instr[0] == 'kill']
		else:
			result_use = [None for instr in expanded_ssa_code if not instr[0] == 'kill']

		return result_code, result_use
		

class OptimiserPass(object):
	''' Base for all optimiser passes. '''
	@classmethod
	def _replace_source_reg(cls, instr, reg, replacement):
		replaced = False
		try:
			if instr.src == reg:
				instr.src = replacement
				replaced = True
		except AttributeError: pass
		try:
			if instr.src1 == reg:
				instr.src1 = replacement
				replaced = True
			if instr.src2 == reg:
				instr.src2 = replacement
				replaced = True
		except AttributeError: pass
		return replaced

class RegisterOptimiser(OptimiserPass):
	''' Register usage optimiser. '''
	def __init__(self, host):
		pass
	def process_window(self, trace_fragment):
		''' Process a trace fragment. '''
		# XXX todo: by changing the register allocations
		# more registers could be freed to be used by the
		# other optimisations. Care should be taken to make sure
		# that the register values between trace_fragments are
		# still compatible!
		return trace_fragment
		

class MemoryPass(OptimiserPass):
	''' Memory optimiser. '''
	def __init__(self, host, access_threshold = 2):
		self.access_th = access_threshold
		self.last_cached_value = None
	def _same_dest_reg(self, instr, dest):
		instr_dest = None
		try:
			if str(instr.dest) == str(dest): return True 
		except AttributeError: pass
		return False
	def process_window(self, trace_fragment):
		''' Process a trace fragment. '''
		# XXX todo: memory reordening (ie instructions reordening)
		reads = {}
		writes = {}
		for i, instr in enumerate(trace_fragment.trace):
			opcode = instr.opcode()
			if opcode == 'memr_imm':
				addr = instr.src
				if not addr in reads: reads[addr] = []
				reads[addr].append(i)
			elif opcode  == 'memw_imm':
				addr = instr.dest
				if not addr in writes: writes[addr] = []
				writes[addr].append(i)
			elif opcode in ['memr', 'memw']:
				# XXX The sequence Imm(addr), MemR(rx, code.imm) | MemW(code.imm, rx)
				# should be allowed, this could also be folded by PeepHole optim.
				print 'can\'t analyse code with memw or memr, skipping optim'
				self.last_cached_value = None
				return trace_fragment

		frequent_read = sorted(((len(accesses), v) for v, accesses in reads.iteritems() \
				        if len(accesses) > self.access_th), reverse=True)
		# we need to check whether there is a write between the reads
		# for a first version, reject a value if there is a write to that addr in the trace frag
		frequent_read = [(acces, addr) for acces, addr in frequent_read if addr not in writes.keys()]
		

		nr_free_regs = trace_fragment.nr_free_regs()

		nr_cached_read = min(len(frequent_read), nr_free_regs)
		cached_reads = dict((frequent_read[i][1], None) for i in xrange(nr_cached_read))

		# now modify the trace fragment
		# add the loading of the cached values
		trace_fragment.set_ptr(0)
		if self.last_cached_value:
			addr, dest = self.last_cached_value
			# make sure that last immediate from last block is available
			trace_fragment.insert_instr_before(MemRImm(dest, addr))

		# make sure that pass starts at 0, even after MemRImm injection
		trace_fragment.set_ptr(0)
		cached_addr = None
		cached_mem = None
		dest_reg = None
		while trace_fragment.ptr < len(trace_fragment):
			instr = trace_fragment.current_instr()
			# XXX condition handling
			if instr.opcode() == 'memr_imm' and not instr.cond() and str(instr.dest) != 'out':
				addr = instr.src
				dest_reg = instr.dest
				if addr in cached_reads:
					if not cached_reads[addr]:
						reg = trace_fragment.alloc_local_reg()
						cached_reads[addr] = reg
						trace_fragment.insert_instr_after(Mov(reg, dest_reg))
						trace_fragment.inc_ptr() # skip this mov in processing
					else:
						trace_fragment.remove_current_instr()
						trace_fragment.dec_ptr() # make sure that next op isn't skipped
					cached_mem = cached_reads[addr]
					cached_addr = addr
				else:
					# just copy instrs if addr not cached
					cached_mem = None
					cached_addr = None
					dest_reg = None
			else:
				if cached_mem:
					if self._same_dest_reg(instr, dest_reg):
						# the value of the memory read is overwritten
						# so stop rewrite
						cached_mem = None
						cached_addr = None
						dest_reg = None
					else:
						try:
							if instr.src == dest_reg: instr.src = cached_mem
						except AttributeError: pass
						try:
							if instr.src1.id == dest_reg.id: instr.src1 = cached_mem
							if instr.src2.id == dest_reg.id: instr.src2 = cached_mem
						except AttributeError: pass
			# release of cache values after last use
			trace_fragment.inc_ptr()

		self.last_cached_value = (cached_addr, dest_reg) if cached_mem else None
		return trace_fragment


class PeepholePass(OptimiserPass):
	''' Peephole optimiser. '''
	def __init__(self, host):
		# instructions that are safe to remove
		# if an instructions is duplicated  in consequtive instrs
		# note mov instructions are save as long as they don't match the pattern
		# mov(out, dir) so they are excluded for now.
		self.remove_duplicate = ['imm']
	def _instr_equal(self, a, b):
		if not a.opcode() == b.opcode(): return False
		if not a.cond() == b.cond(): return False
		if a.opcode() == 'imm':
			if not a.value == b.value: return False
		try:
			if not a.dest == b.dest: return False
		except AttributeError: pass
		try:
			if not str(a.src)== str(b.src): return False
		except AttributeError: pass
		try:
			if not str(a.src2)== str(b.src2): return False
		except AttributeError: pass
		try:
			if not str(a.src1)== str(b.src1): return False
		except AttributeError: pass
		return True
	def process_window(self, trace_fragment):
		''' Process a trace fragment. '''
		trace_fragment.set_ptr(0)
		while trace_fragment.ptr < len(trace_fragment):
			instr = trace_fragment.current_instr()
			if instr.opcode() == 'nop':
				trace_fragment.remove_current_instr()
			trace_fragment.inc_ptr()
		return trace_fragment


class ImmediatePass(OptimiserPass):
	''' The immediate pass tries to cache frequently used immediate values in free registers. '''
	def __init__(self, host, frequency_threshold = 1):
		self.old_cached_coeffs = {}
		self.old_register_rewrite = {}
		self.freq_th = frequency_threshold
	def _get_frequent_immediates(self, trace_fragment):
		imm_frequency = {}
		for instr in trace_fragment.trace:
			if instr.opcode() == 'imm':
				# XXX possible optimization: only count imm without cond
				# these instructions will be ignored anyway in the optim
				value = instr.value
				if not value in imm_frequency:
					imm_frequency[value] = 0
				imm_frequency[value] += 1

		return sorted(((cnt, v) for v, cnt in imm_frequency.iteritems() if cnt > self.freq_th), reverse=True)
	def process_window(self, trace_fragment):
		''' Process a trace fragment. '''
		# stupid hack to convert between register objects and register strings
		special_regs = [trace_fragment.code.north, trace_fragment.code.east,\
						trace_fragment.code.south, trace_fragment.code.west, trace_fragment.code.out]
		name2reg = dict((str(x), x) for x in trace_fragment.code.regs + special_regs)

		# reinsert immediates that were cached in previous fragment
		# to their original register, all the cached registers that were not
		# overwritten yet can be found in old_register_rewrite.keys()
		old_targets = dict((v, k) for k, v in self.old_register_rewrite.iteritems())
		immediates_to_restore = [(r, v) for v, r in self.old_cached_coeffs.iteritems() if r in old_targets]
		for reg, value in immediates_to_restore:
			orig_reg = old_targets[reg]
			trace_fragment.insert_instr_before(Imm(orig_reg, value))

		# add the loading of the cached values
		trace_fragment.set_ptr(0)

		frequent_imm  = self._get_frequent_immediates(trace_fragment)
		nr_free_regs = trace_fragment.nr_free_regs()

		nr_cached_imm = min(len(frequent_imm), nr_free_regs)
		cached_coeffs = dict((frequent_imm[i][1], None) for i in xrange(nr_cached_imm))
		# mapping from original imm destination regs to new reg in cached_coeffs
		register_rewrite = {}

		while trace_fragment.ptr < len(trace_fragment):
			instr = trace_fragment.current_instr()
			def_, _ = VariableManager.get_def_use(instr_compact_repr(instr))
			if def_:	# handle None values
				def_ = name2reg[def_] # convert to register object
			if instr.opcode() == 'imm' and not instr.cond():
				value = instr.value
				if value in cached_coeffs:
					if not cached_coeffs[value]:
						# XXX could be improved by first finding the first and last usage
						# and finding the register with the tightest free range fit for this interval
						reg = trace_fragment.alloc_local_reg()
						# update book keeping
						cached_coeffs[value] = reg
						register_rewrite[instr.dest] = reg

						# assign new reg
						# XXX the old register usage could be freed!!!
						instr.dest = reg
					else:
						# update rewrite mapping
						register_rewrite[instr.dest] = cached_coeffs[value]
						trace_fragment.remove_current_instr()
						trace_fragment.dec_ptr() # make sure that next op isn't skipped
				else:
					if def_ in register_rewrite:
						# new definition of register, value of immediate is gone
						# so mapping is not needed anymore
						del register_rewrite[def_]
			else:
				if def_ in register_rewrite:
					# new definition of register, value of immediate is gone
					# so mapping is not needed anymore
					del register_rewrite[def_]
				for reg, replacement in register_rewrite.iteritems():
					replaced = self._replace_source_reg(instr, reg, replacement)
					if replaced:
						value = [k for k, v in cached_coeffs.iteritems() if v == replacement][0]
						tag_instr(instr, 'cache_use_imm:%s'%str(value))
			trace_fragment.inc_ptr()

		# now determine the release time of all cached variables
		# note that a second run is needed to get the these line nrs
		last_cached_usage = dict((k, -1) for k, v in cached_coeffs.iteritems())
		for i, instr in enumerate(trace_fragment.trace):
			cache_use = None
			if hasattr(instr, 'tag'):
				try:
					cache_use = [x.split(':')[1] for x in instr.tag if 'cache_use_imm' in x][0]
				except (IndexError): pass
			if cache_use:
				value = float(cache_use)
				if value in last_cached_usage:
					last_cached_usage[value] = i

		# finally release the variables at the appropriate moment
		for value, line in last_cached_usage.iteritems():
			trace_fragment.set_ptr(line)
			trace_fragment.release_local_reg(cached_coeffs[value])

		self.old_register_rewrite = register_rewrite
		self.old_cached_coeffs = cached_coeffs
		return trace_fragment

class Optimiser(object):
	def __init__(self, optim_window = 1000, use_loop_hints = False):
		self.optim_window = optim_window
		self.use_loop_hints = use_loop_hints
		self.passes = []
	def register_pass(self, optim_pass):
		''' Add a new optimiser pass to the optimiser. '''
		self.passes.append(optim_pass)
	@classmethod
	def _current_reg_usage(cls, code):
		return [code.used_regs[r.id] for r in code.regs]
	@classmethod
	def _is_loopstart(cls, instr):
		return hasattr(instr, 'tag') and 'loop_start' in instr.tag
	def run(self, code, codegen, block_size, args):
		''' Optimise the instruction stream generated by code.gen. '''
		# gather instructions and register usage while running the codegen
		# the instruction will be processed in fragments by the optimiser passes.
		instr_window = []
		reg_used_window = []
		for x in codegen(code, block_size, args):
			# restart tracing when a loop start hint is found
			# this syncs the optimiser window to the loop bodies
			# XXX adapt window to loop trace size
			loop_start = self.use_loop_hints and self._is_loopstart
			if len(instr_window) < self.optim_window and not loop_start:
				instr_window.append(x)
				reg_used_window.append(self._current_reg_usage(code))
			else:
				trace_fragment  = TraceFragment(code, instr_window, reg_used_window)
				for px in self.process_window(trace_fragment):
					yield px
				instr_window = [x]
				reg_used_window = [self._current_reg_usage(code)]
		# process remaining code
		for x in self.process_window(TraceFragment(code, instr_window, reg_used_window)):
			yield x

	def process_window(self, trace_fragment):
		''' Apply all optimisation passes to the current trace fragment. '''
		for optim_pass in self.passes:
			trace_fragment = optim_pass.process_window(trace_fragment)

		for instr in trace_fragment.trace:
			yield instr

class CompilerOptimiser(object):
	''' Try to use the compiler optimisation toolchain for trace optimisations. '''
	def __init__(self, fragment_len):
		self.ssa_convertor = CodegenToSSAConvertor(fragment_len, True)
		self.passes = []
	def add_optimiser(self, optimiser):
		self.passes.append(optimiser)
	def run(self, code, codegen, block_size, args):
		''' Optimise the instruction stream generated by code.gen. '''
		for ssa_code_fragment in self.ssa_convertor.gen_ssa_fragments(codegen, code, block_size, args):
			# optimise code
			for optim_pass in self.passes:
				ssa_code_fragment = optim_pass.process_window(ssa_code_fragment)
			for instr,_ in ssa_code_fragment:
				yield instr

def get_allocation_ranges(all_usage):
	''' Calculate allocation ranges from a bool allocation table. '''
	if any(x is None for x in all_usage):
		raise Exception('missing allocation information')
	nr_regs = len(all_usage[0])
	reg_allocated = [False for _ in xrange(nr_regs)]
	ranges = [[] for _ in xrange(nr_regs)]
	for i, usage in enumerate(all_usage):
		for rid, (allocated, was_allocated) in enumerate(izip(usage, reg_allocated)):
			if allocated != was_allocated:
				if allocated:
					ranges[rid].append([i, None])
				else:
					ranges[rid][-1][1] = i-1
				reg_allocated[rid] = allocated # update status
	for rid, reg_range in enumerate(ranges):
		if not len(reg_range): continue
		if reg_range[-1][1] is None:
			ranges[rid][-1][1] = len(all_usage)-1
	return ranges

if __name__ == '__main__':
	from optparse import OptionParser
	from blip.code.codegen import convert_compact_repr_to_obj
	from blip.simulator.opcodes import *

	parser = OptionParser()
	(options, args) = parser.parse_args()

	code = Code()
	def test_codegen(code, block_size, args):
		with scoped_alloc(code, 2) as (a, b):
			yield Imm(b, 3)
			yield Mov(a, b)
			with scoped_alloc(code, 1) as c:
				for x in xrange(3):
					with scoped_alloc(code, 2) as (e, f):
						yield Imm(f, 2)
						yield Add(e, b, f)
					with scoped_alloc(code, 2) as (g, h):
						yield Imm(g, 1)
						yield Add(c, a, g)
						yield Add(c, a, g)
						yield Imm(h, 1)
						yield Mov(c, h)
						yield Mov(c, h)
			yield Xor(a, a, a)
			with scoped_alloc(code, 1) as const_1:
				yield Imm(const_1, 1)
				yield Mov(a, const_1)
			yield Mov(code.out, a)
			#yield Mov(b, code.east)
			yield Mov(b, a)
			
	args = {}
	block_size = (4, 4)
	block_len = 100
	convertor = CodegenToSSAConvertor(block_len, True)

	try:
		all_code = []
		all_usage = []
		for i, current_ref_code in enumerate(convertor.gen_ssa_fragments(test_codegen, Code(), block_size, args)):
			test_code = convert_compact_repr_to_obj(current_ref_code)
			for instr, reg_usage in current_ref_code:
				all_code.append(instr)
				all_usage.append(reg_usage)
			print '\n'.join(str(x) for x in test_code)

		liveness = RegisterAllocator.liveness_analysis(all_code)
		RegisterAllocator.print_liveness(all_code, liveness)

		print '\n'.join(' '.join('x' if x else '.' for x in y) for y in all_usage)
		original_reg_ranges = get_allocation_ranges(all_usage)
		print 'original register ranges:'
		print original_reg_ranges

		new_code, reg_mapping = RegisterAllocator.register_allocation(all_code, liveness, 30)
		print '\n'.join(str(x) for x in new_code)
		print reg_mapping

	except Exception, e:
		print str(e)
		import pdb, traceback, sys
                traceback.print_tb(sys.exc_traceback)
                pdb.post_mortem(sys.exc_traceback)

