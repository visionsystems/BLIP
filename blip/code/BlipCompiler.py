from __future__ import with_statement
import ast
import copy
import sys
import pdb
import traceback
import operator

from blip.simulator.opcodes import SPECIAL_REGS

class InvalidCodeException(Exception): pass
class UnImplementedFeatureException(Exception): pass
class InvalidArgsException(InvalidCodeException): pass
class InternalCompilerException(Exception): pass
class RegisterAlreadyDefinedException(InternalCompilerException): pass
class UndefinedVariableException(InvalidCodeException): pass

def kernel(f):
	f.__kernel__ = True
	return f

def ast_is_kernel(f):
	try:
		for x in f.decorator_list:
			if x.id == 'kernel': return True
	except Exception, e:
		print e
	return False

class VariableContextManager(object):
	class VariableContext(object):
		def __init__(self):
			self.variables = {}

	class LocalContext(object):
		''' Nicer local scope definition. '''
		def __init__(self, varcontext):
			self.varcontext = varcontext
		def __enter__(self):
			self.varcontext.push()
			return self.varcontext.context()
		def __exit__(self, *args):
			self.varcontext.pop()
	def local_context(self):
		return self.LocalContext(self)

	def __init__(self):
		self.contexts = [VariableContextManager.VariableContext()]
		self.max_variables_id = {}
		self.cnt = 0
	def context(self):
		return self.contexts[-1]
	def push(self):
		self.contexts.append(VariableContextManager.VariableContext())
	def pop(self):
		self.contexts.pop()
	@classmethod
	def gen_variable_name(cls, name, version):
		return '%s@%i'%(name, version)
	@classmethod
	def symbolic_name(cls, name):
		name, _, _ = name.rpartition('@')
		return name
	@classmethod
	def has_symbolic_name(cls, name_var, symbol_name):
		return cls.symbolic_name(name_var.name) == symbol_name
	@classmethod
	def name_version(cls, name):
		_, _, version = name.rpartition('@')
		return int(version)
	def def_var(self, var, seq_value=False):
		version = 0	
		if var in self.max_variables_id:
			version = self.max_variables_id[var] + 1

		self.max_variables_id[var] = version
		new_var = NamedValue(self.gen_variable_name(var, version), seq_value = seq_value)
		# add new ssa version add the context where the var was originally defined
		old_context = self.find_var_def_context(var)
		if old_context:
			old_context.variables[var] = new_var
		else:
			self.context().variables[var] = new_var
		return new_var
	def find_var_def_context(self, var):
		for i in xrange(len(self.contexts)-1, -1, -1):
			context = self.contexts[i]
			if var in context.variables:
				return context 
		return None
	def use_var(self, var, seq_value = False):
		for i in xrange(len(self.contexts)-1, -1, -1):
			context = self.contexts[i]
			if var in context.variables:
				res = context.variables[var]
				return res
		raise UndefinedVariableException('variable %s undefined'%var)
	def get_tmp_name(self):
		name = 'tmp_%i'%self.cnt
		self.cnt += 1
		return name 

# Value decorators
def do_not_remove(cls):
	orig_init  = cls.__init__
	def __init__(self, *args, **kwargs):
		orig_init(self, *args, **kwargs)
		self.do_not_remove = True
	cls.__init__ = __init__
	return cls

def seq_value(cls):
	orig_init  = cls.__init__
	def __init__(self, *args, **kwargs):
		orig_init(self, *args, **kwargs)
		self.seq_value = True
	cls.__init__ = __init__
	return cls

# Value system
class Value(object):
	def __init__(self, **kwargs):
		self.seq_value = False # values for sequencer instructions
		self.do_not_remove = False # do not remove, although value is never used
		for attr in ['seq_value', 'do_not_remove']:
			if attr in kwargs:
				setattr(self, attr, kwargs[attr])
	def type(self):
		if hasattr(self, '_type'):
			return self._type
		else:
			return self.__class__.__name__.lower().replace('value','')
	def gen_str(self, v):
		return '<%s%s>'%(v, ' #seq' if self.seq_value else '')

class NamedValue(Value):
	def __init__(self, name, **kwargs):
		Value.__init__(self, **kwargs)
		self.name = name
	def __str__(self):
		return self.gen_str('%s'%self.name)

class PhiValue(Value):
	def __init__(self, cond, left, right, **kwargs):
		Value.__init__(self, **kwargs)
		self.cond = cond
		self.left = left
		self.right = right
	def __str__(self):
		return self.gen_str('phi %s %s %s'%(self.cond, self.left, self.right))

class EmitValue(Value):
	def __init__(self, value, **kwargs):
		Value.__init__(self, **kwargs)
		self.value = value
	def __str__(self):
		return self.gen_str('emit %s'%self.value)
	
class BinopValue(Value):
	def __init__(self, left, op, right, **kwargs):
		Value.__init__(self, **kwargs)
		self.left = left
		self.right = right
		self.op = op
		#propagate the seq type annotations
		if self.left.seq_value != self.right.seq_value:
			raise InternalCompilerException('can\'t mix seq type and pe types without explicit conversion')
		self.seq_value = self.left.seq_value and self.right.seq_value
	def __str__(self):
		return self.gen_str('binop %s %s %s'%(str(self.left), str(self.op), str(self.right)))

class ConstValue(Value):
	def __init__(self, value, **kwargs):
		Value.__init__(self, **kwargs)
		self.value = value
	def __str__(self):
		return self.gen_str('const %s'%(str(self.value)))

@seq_value
class LabelValue(Value):
	def __init__(self, **kwargs):
		Value.__init__(self, **kwargs)
	def __str__(self):
		return self.gen_str('label')

class CondValue(Value):
	def __init__(self, test, body, orelse, **kwargs):
		Value.__init__(self, **kwargs)
		self.test = test
		self.body = body
		self.orelse = orelse
	def __str__(self):
		return self.gen_str('condvalue %s:%s?%s'%(str(self.test), str(self.body), str(self.orelse)))

class MemDereference(Value):
	def __init__(self, ptr, index, **kwargs):
		Value.__init__(self, **kwargs)
		self.ptr = ptr
		self.index = index
	def __str__(self):
		return self.gen_str('memderef %s[%s]'%(str(self.ptr), str(self.index)))

@do_not_remove
class MemWriteValue(Value):
	def __init__(self, ptr, index, value, **kwargs):
		Value.__init__(self, **kwargs)
		self.ptr = ptr
		self.index = index
		self.value = value
	def __str__(self):
		return self.gen_str('memwrite %s[%s] <- %s'%(str(self.ptr), str(self.index), str(self.value)))

class ConstList(Value):
	def __init__(self, elements, **kwargs):
		Value.__init__(self, **kwargs)
		self.elements = elements
	def __str__(self):
		return self.gen_str('constlist (%s)'%', '.join(str(x) for x in self.elements))

@do_not_remove
class ArgumentPlaceholderValue(Value):
	def __init__(self, **kwargs):
		Value.__init__(self, **kwargs)
	def __str__(self):
		return self.gen_str('<insert arg here>')

@do_not_remove
class FunctionCallValue(Value):
	def __init__(self, func, args, **kwargs):
		Value.__init__(self, **kwargs)
		self.func = func
		self.args = args
	def __str__(self):
		return self.gen_str('call %s(%s)'%(str(self.func), ', '.join(str(x) for x in self.args)))

@do_not_remove
@seq_value
class JmpValue(Value):
	def __init__(self, cond, label, **kwargs):
		Value.__init__(self, **kwargs)
		self.cond = cond
		self.label = label
	def __str__(self):
		return self.gen_str('jmp %s, %s'%(str(self.cond), str(self.label)))

class Comparison(Value):
	def __init__(self, left, comparison, right, **kwargs):
		Value.__init__(self, **kwargs)
		self.left = left
		self.comparison = comparison
		self.right = right
		#propagate the seq type annotations
		if self.left.seq_value != self.right.seq_value:
			raise InternalCompilerException('can\'t mix seq type and pe types without explicit conversion')

		self.seq_value = self.left.seq_value and self.right.seq_value
		self.convert = {'Eq':'EQ', 'notEq':'NEQ', 'Lt':'LT', 'LtE':'LE', 'Gt':'GT', 'GtE': 'GE'}
		self.invert = {'EQ': 'NEQ', 'NEQ':'EQ', 'LT':'GE', 'GE':'LT', 'LE':'GT', 'GT':'LE'}
	def condition(self):
		comp_str = self.comparison
		if not comp_str in self.convert:
			raise InvalidCodeException('unsupported comparison operator %s'%comp_str)
		return self.convert[comp_str]
	def inv_condition(self):
		return self.invert[self.condition()]
	def __str__(self):
		return self.gen_str('compare %s %s %s'%(str(self.left), str(self.comparison), str(self.right)))

class BodyVisitor(ast.NodeVisitor):
	''' Convert the function body to the value representation. '''
	def __init__(self, compiler, kernel_object):
		self.compiler = compiler
		self.kernel_object = kernel_object
		self.code = kernel_object.code
		self.varcontext = kernel_object.varcontext
		self.function_name = kernel_object.function_name
		self.values = kernel_object.values

		self.last_return_value = False

	def add_value(self, name, value):
		''' Append a value to the value list. '''
		self.values.append((name, value))
	def find_label_pos(self, label_var):
		''' Return the position of the label value. '''
		for i, (n, v) in enumerate(self.values):
			if n == label_var: return i
		raise Exception('label %s not found'%str(label_var))
	def insert_after_label(self, label_var, namedvalue, value):
		''' Insert a value after the label label_var. '''
		pos = self.find_label_pos(label_var)
		self.values.insert(pos + 1, (namedvalue, value))
	def handle_value(self, value, seq_value = False):
		if isinstance(value, str):
			value = self.varcontext.use_var(value, seq_value)
		if not isinstance(value, NamedValue):
			name = self.varcontext.get_tmp_name()
			nvalue = value
			if isinstance(nvalue, ConstValue):
				nvalue.seq_value = seq_value
			value = self.varcontext.def_var(name, seq_value)
			self.add_value(value, nvalue)
		return value
	def resolve_seq_values(self, values):
		''' Add xemit instructions for seq->pe copies if needed. '''
		seq_res = all(v.seq_value for v in values)
		mixed_res = any(v.seq_value for v in values)
		# directly return if all values are either seq_value or not
		if seq_res or not mixed_res:
			return values
		
		resolved_values = []
		for value in values:
			if value.seq_value:
				old_value = value
				value = self.varcontext.def_var(self.varcontext.get_tmp_name())
				self.add_value(value,  EmitValue(old_value))
			resolved_values.append(value)
		return resolved_values
	def set_return_value(self, value):
		''' Set the function return value. '''
		if self.last_return_value:
			raise InvalidCodeException('there can be only a single return value in a function')
		self.last_return_value = value
		return_value = self.kernel_object.return_val()
		print 'returning', str(return_value)
		return return_value
	def new_label(self):
		return LabelValue()
	def unique_label(self, basename = ''):
		''' Calculate a new global unique label. '''
		num_id = self.compiler.get_unique_id()
		return '%s_%i'%(basename if basename else 'label', num_id)
	def visit_Num(self, node):
		return ConstValue(node.n)
	def visit_Add(self, node):
		return 'add'
	def visit_Sub(self, node):
		return 'sub'
	def visit_Mult(self, node):
		return 'mul'
	def visit_BitOr(self, node):
		return 'or'
	def visit_BitAnd(self, node):
		return 'and'
	def visit_BitXor(self, node):
		return 'xor'
	def visit_BinOp(self, node):
		left = self.visit(node.left)
		left_var = self.handle_value(left)
		right = self.visit(node.right)
		right_var = self.handle_value(right)
		left_var, right_var = self.resolve_seq_values([left_var, right_var])
		op = self.visit(node.op)
		assert left_var.seq_value == right_var.seq_value
		return BinopValue(left_var, op, right_var)
	def visit_Name(self, node):
		name = node.id
		if '@' in name:
			raise InvalidCodeException('illegal identifier %s: @ in identifiers is not allowed'%sname)
		return name
	def visit_Compare(self, node):
		if len(node.ops) > 1 or len(node.comparators) > 1:
			raise InvalidCodeException('only simple conditions are allowed')
		left = self.visit(node.left)
		left_var = self.handle_value(left)
		right = self.visit(node.comparators[0])
		right_var = self.handle_value(right)
		comparator = node.ops[0].__class__.__name__

		left_var, right_var = self.resolve_seq_values([left_var, right_var])
		assert left_var.seq_value == right_var.seq_value
		return Comparison(left_var, comparator, right_var)
	def visit_Assign(self, node):
		if len(node.targets) > 1: raise InvalidCodeException()
		value = self.visit(node.value)
		if isinstance(value, str):
			value = self.varcontext.use_var(value)
		target = self.visit(node.targets[0])
		var_target = None
		if isinstance(target, MemDereference):
			value = self.handle_value(value)
			value = MemWriteValue(target.ptr, target.index, value)
			var_target = self.varcontext.def_var(self.varcontext.get_tmp_name())
		else:
			var_target = self.varcontext.def_var(target)

		var_target.seq_value = value.seq_value
		self.add_value(var_target, value)
		return var_target
	def visit_AugAssign(self, node):
		value = self.visit(node.value)
		target = self.visit(node.target)
		var_value = self.handle_value(value)

		var_target = self.varcontext.use_var(target)
		op = self.visit(node.op)
		var_target, var_value = self.resolve_seq_values([var_target, var_value])
		binop = BinopValue(var_target, op, var_value)
		var_new_target = None
		if isinstance(target, MemDereference):
			binop = MemWriteValue(target.ptr, target.index, binop)
			binop = self.handle_value(binop)
			var_new_target = self.varcontext.def_var(self.varcontext.get_tmp_name())
		else:
			var_new_target = self.varcontext.def_var(target)
		self.add_value(var_new_target, binop)
		return var_new_target
	def visit_List(self, node):
		elements = [self.visit(x) for x in node.elts]
		if all(isinstance(x, ConstValue) for x in elements):
			return ConstList(elements)
		elif all(isinstance(x, ConstList) for x in elements):
			return ConstList(elements)
		else:
			raise InvalidCodeException('Only homogenous lists of numbers or Const Lists are allowed in Const Lists')
	def visit_Index(self, node):
		# don't visit the node, otherwise the code is generated too early
		return self.visit(node.value)
	def visit_Subscript(self, node):
		value = self.visit(node.value)
		var_value = self.handle_value(value)
		index = self.visit(node.slice)
		if not index:
			raise InvalidCodeException('only single value indexing is allowed')
		var_index = self.handle_value(index)
		# index will be visited when generating the memdereference code
		return MemDereference(var_value, var_index)
	def visit_IfExp(self, node):
		test = self.visit(node.test)
		var_test = self.handle_value(test)
		body = self.visit(node.body)
		var_body = self.handle_value(body)
		orelse = self.visit(node.orelse)
		var_orelse = self.handle_value(orelse)
		return CondValue(var_test, var_body, var_orelse)	


	def _visit_For_Seq(self, cnt, target, body):
		symbolic_name = self.varcontext.symbolic_name # shortcut for long name

		var_cnt = self.handle_value(cnt, seq_value=True)

		# inside loop intro
		self.varcontext.push()
		var_loop_intro_label = NamedValue(self.unique_label('loop_intro'), seq_value = True)
		self.add_value(var_loop_intro_label, self.new_label())

		var_target = self.varcontext.def_var(target, seq_value=True)
		self.add_value(var_target, ConstValue(0, seq_value=True))
		var_inc_1 = self.varcontext.def_var('inc_1', seq_value=True)
		self.add_value(var_inc_1, ConstValue(1, seq_value=True))

		var_loop_label = NamedValue(self.unique_label('for'), seq_value = True)
		self.add_value(var_loop_label, self.new_label())

		# inside loop
		self.varcontext.push()
		for b in body:
			self.visit(b)
		self.varcontext.pop()

		# outside loop
		old_target = self.varcontext.use_var(target, seq_value=True)
		new_cnt = self.varcontext.def_var(target, seq_value=True)
		self.add_value(new_cnt, BinopValue(old_target, 'add', var_inc_1))

		# phi node insertion
		# hack to find all variables used in the body
		def get_used_values(value):
			for attrname in dir(value):
				attr = getattr(value, attrname)
				if isinstance(attr, NamedValue):
					yield attr
				elif isinstance(attr, Value):
					for vuse in get_used_values(attr):
						yield vuse
		def get_all_used_values(values):
			for v in values:
				for u in get_used_values(v):
					yield u
		def get_use(symbol_name, values, first_use=True):
			''' Returns first or last use of a variable. '''
			matching_values = [x for x in values if self.varcontext.has_symbolic_name(x, symbol_name)]
			return sorted(matching_values, key=lambda x: self.varcontext.name_version(x.name))[0 if first_use else -1]
		def symbolic_replace_use(value, search_symbolic_name, replacement, skiplist = []):
			''' Replace all uses of a symbolic name with the replacement. '''
			new_value = copy.copy(value)
			if isinstance(value, NamedValue):
				if symbolic_name(value.name) == search_symbolic_name:
					new_value.name = replacement
			else:
				for attrname in dir(value):
					attr = getattr(value, attrname)
					if isinstance(attr, NamedValue):
						if symbolic_name(attr.name) == search_symbolic_name:
							setattr(new_value, attrname, NamedValue(replacement))
					elif isinstance(attr, Value):
						new_value = symbolic_replace_use(attr, search_symbolic_name, replacement, skiplist)
			return new_value

		loop_values_start = self.find_label_pos(var_loop_label) + 1
		body_values = self.values[loop_values_start:]

		assigned = [n for n, v in body_values]
		used = list(x for x in get_all_used_values(v for n, v in body_values))
		# candidates have a common symbolic name in both assigned and used values
		phi_candidates = set(symbolic_name(x.name) for x in assigned).intersection(symbolic_name(y.name) for y in used)

		#print 'body values'
		#print '\n'.join(' - %s: %s'%(str(x), str(y)) for x, y in body_values)
		#print 'assigned: ', ', '.join(str(x) for x in assigned)
		#print 'used: ', ', '.join(str(x) for x in used)
		#print 'phi_candidates', phi_candidates

		# add jump
		var_cnt_cond = self.varcontext.def_var(self.varcontext.get_tmp_name(), seq_value = True)

		new_cnt, var_cnt = self.resolve_seq_values([new_cnt, var_cnt])
		self.add_value(var_cnt_cond, Comparison(new_cnt, 'Lt', var_cnt))
		var_jump = self.varcontext.def_var(self.varcontext.get_tmp_name(), seq_value=True)
		self.add_value(var_jump, JmpValue(var_cnt_cond, var_loop_label, seq_value=True))

		phi_values = []
		# add phi condition
		var_0 = self.varcontext.def_var(self.varcontext.get_tmp_name(), seq_value = True)
		phi_values.append((var_0, ConstValue(0, seq_value = True)))

		# condition for sequencer values
		var_phi_cond_seq = self.varcontext.def_var(self.varcontext.get_tmp_name(), seq_value = True)
		phi_values.append((var_phi_cond_seq, Comparison(new_cnt, 'Eq', var_0, seq_value = True)))
		# condition for normal values
		new_cnt_noseq = self.varcontext.def_var(self.varcontext.get_tmp_name())
		phi_values.append((new_cnt_noseq, EmitValue(new_cnt)))
		var_0_noseq = self.varcontext.def_var(self.varcontext.get_tmp_name())
		phi_values.append((var_0_noseq, ConstValue(0)))
		var_phi_cond = self.varcontext.def_var(self.varcontext.get_tmp_name())
		phi_values.append((var_phi_cond, Comparison(new_cnt_noseq, 'Eq', var_0_noseq)))

		# add phi values
		replacements = {}
		# note: the following list needs to be verified manually!
		skipvarlist = [var_cnt_cond, var_phi_cond, var_0, var_0_noseq, new_cnt_noseq, var_phi_cond_seq]
		skiplist = [x.name for x in skipvarlist]
		for c in phi_candidates:
			phi_left = get_use(c, used, first_use=True)
			phi_right = get_use(c, assigned, first_use=False)
			if phi_left.name == phi_right.name: continue # no need to handle values that have no new assignment
			assert phi_left.seq_value == phi_right.seq_value
			seq_value = phi_right.seq_value
			# create phi node
			var_phi = self.varcontext.def_var(self.varcontext.get_tmp_name(), seq_value)
			cond = var_phi_cond_seq if seq_value else var_phi_cond
			phi_values.append((var_phi, PhiValue(cond, phi_left, phi_right, seq_value = seq_value)))
			replacements[symbolic_name(phi_left.name)] = var_phi
			skiplist.append(var_phi.name)
		print ', '.join(str(x) for x in skiplist)

		# now add all new values
		for name, value in reversed(phi_values):
			self.insert_after_label(var_loop_label, name, value)
		# rewrite all uses of entry phi_candidates
		for i in xrange(loop_values_start+1, len(self.values)):
			value = self.values[i][1]
			name = self.values[i][0].name
			if name in skiplist: continue
			new_value = value # if nothing is replaced, keep original value
			for search, replace in replacements.iteritems():
				new_value =  symbolic_replace_use(new_value, search, replace.name, skiplist)
				self.values[i] = (self.values[i][0], new_value)

		self.varcontext.pop()
		# outside loop
	def _visit_For_NoSeq(self, cnt, target, body):
		# inside loop intro
		self.varcontext.push()

		for i in xrange(cnt.value):
			var_target = self.varcontext.def_var(target)
			self.add_value(var_target, ConstValue(i))
			# inside loop
			self.varcontext.push()
			for b in body:
				self.visit(b)

			self.varcontext.pop()
	def visit_For(self, node):
		if node.orelse:
			raise InvalidCodeException('else statements after for loops are not supported')
		target = self.visit(node.target)

		# no visit here, otherwise a real call would be generated
		iter_n = node.iter
		if not (isinstance(iter_n, ast.Call) and iter_n.func.id in ['range', 'xrange']):
			raise InvalidCodeException('only calls to range and xrange are supported right now')
		if len(iter_n.args) > 1:
			raise InvalidCodeException('for now only a single argument call to (x)range is  supported')
		cnt = self.visit(iter_n.args[0])
		if self.compiler.config['no_sequencer']:
			self._visit_For_NoSeq(cnt, target, node.body)
		else:
			self._visit_For_Seq(cnt, target, node.body)

	def visit_Call(self, node):
		if node.keywords or node.starargs or node.kwargs:
			raise InvalidCodeException('no keyword/star/dict args allowed')
		args = [self.visit(a) for a in node.args]
		func = self.visit(node.func)
		# recursion is not allowed
		if func == self.function_name:
			raise InvalidCodeException('recursion is not allowed')

		# inline code right away
		# decide call semantics here: copy by value or ref?
		if self.compiler.is_builtin_function(func):
			builtin = self.compiler.get_builtin_function(func)
			args = [self.handle_value(a) for a in args]
			return builtin(args)
		else:
			# inline function call
			# first compile function
			func_ast = self.compiler.get_kernel_ast(func)
			if not func_ast:
				raise InvalidCodeException('function %s is not defined'%func)

			# XXX patch varcontext for this codeobject
			subCodeObject = KernelObject(func)
			subCodeObject.varcontext = self.varcontext

			# now add argument passing
			for arg, v in zip(subCodeObject.arguments, args):
				var_arg = self.varcontext.def_var(arg)
				var_v = self.varcontext.use_var(v)
				self.add_value(var_arg, var_v)
			subCodeObject = self.compiler.compile_kernel_to_values(func_ast, subCodeObject)

			# finally add the function code
			for c in subCodeObject.values:
				self.values.append(c)

			return subCodeObject.return_val()

		#return FunctionCallValue(func, var_args)
	def visit_Return(self, node):
		value = self.visit(node.value)
		var_value = self.handle_value(value)
		return_value = self.set_return_value(var_value)
		self.add_value(return_value, var_value)
		return return_value		

	def visit_Expr(self, node):
		value = self.visit(node.value)
		if value:
			if isinstance(value, BuiltinFunction):
				valuename = self.varcontext.def_var(self.varcontext.get_tmp_name())
				self.add_value(valuename, value)

	def generic_visit(self, node):
		raise InvalidCodeException('Invalid code: %s'%ast.dump(node))
		
class KernelObject(object):
	def __init__(self, function_name):
		self.varcontext = VariableContextManager()
		self.code = []
		self.function_name = function_name
		self.return_val_name = self.function_name + '___return'
		self._return_val = None
		self.arguments = []
		self.reg_mapping = {}
		self.values = []
	def return_val(self):
		if self._return_val: return self._return_val
		self._return_val = self.varcontext.def_var(self.return_val_name)
		return self._return_val

class RegisterManager(object):
	def __init__(self, nr_reg):
		self.nr_reg = nr_reg
		self.reg_free = [True for x in xrange(self.nr_reg)]
	def get_free_reg(self):
		for i, free in enumerate(self.reg_free):
			if free:
				self.reg_free[i] = False
				return 'r%i'%i
		raise Exception('No more free registers')
	def release_reg(self, reg):
		reg_nr = int(reg[1:])
		if reg_nr < 0 or reg_nr >= self.nr_reg:
			raise Exception('Invalid register id %i'%reg_nr)
		if self.reg_free[reg_nr]:
			raise Exception('Trying to free already freed register %i'%reg_nr)
		self.reg_free[reg_nr] = True

@do_not_remove
class BuiltinFunction(Value):
	def __init__(self, name, nr_args, func_args, **kwargs):
		Value.__init__(self, **kwargs)
		self.name = name
		self.nr_args = nr_args
		self.func_args = func_args
		self.check_args(func_args)
	def check_args(self, func_args):
		if len(func_args) != self.nr_args:
			msg = 'need %i args instead of %i for function %s'%(self.nr_args, len(func_args), self.name)
			raise InvalidArgsException(msg)
	def gen_code(self, codegen, res):
		raise InternalCompilerError('no codegen defined')
	@classmethod
	def is_valid_builtin(cls, candidate):
		try:
			return issubclass(candidate, cls) and not candidate == cls
		except:
			return False
	@classmethod
	def gather_valid_builtins(cls):
		def get_builtin_name(builtin_cls):
			# XXX hack, find all builtin names. Remove check while searching
			old_check_args = cls.check_args
			def fake_check(self, args): return True
			cls.check_args = fake_check
			name = builtin_cls([]).name
			cls.check_args = old_check_args
			return name
		valid_builtins = dict((get_builtin_name(c), c) for name, c in globals().items() if cls.is_valid_builtin(c))
		return valid_builtins

class GetGroupId(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		# actually get_group_id should have a parameters to select the dimension
		# for now, assume a single dimension
		BuiltinFunction.__init__(self, 'get_group_id', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('grid', None, dest.name)
class GetGlobalId(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'get_global_id', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		arg = self.func_args[0]
		yield ('glid', None, dest.name, arg.name)
class GetLocalId(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'get_local_id', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		arg = self.func_args[0]
		yield ('lid', None, dest.name, arg.name)
class LoadWest(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'loadWest', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('mov', None, dest.name, 'west')
class LoadEast(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'loadEast', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('mov', None, dest.name, 'east')
class LoadNorth(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'loadNorth', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('mov', None, dest.name, 'north')
class LoadSouth(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'loadSouth', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('mov', None, dest.name, 'south')
class SendToOutput(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'sendOut', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
# slightly more efficient implementation of communication, combine send/receive
class TransferFromWest(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'transferFromWest', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
		yield ('mov', None, dest.name, 'west')
class TransferFromEast(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'transferFromEast', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
		yield ('mov', None, dest.name, 'east')
class TransferFromNorth(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'transferFromNorth', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
		yield ('mov', None, dest.name, 'north')
class TransferFromSouth(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'transferFromSouth', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
		yield ('mov', None, dest.name, 'south')
class Get2D(BuiltinFunction):
	''' Function to get a value from a buffer, with neighbor value handling. '''
	src = '''
@kernel
def get2D(ptr, x, y, bwidth, bheight):
	# determine index/copy directions
	x_bl = x < 0
	x_br = x >= bwidth
	y_bl = y < 0
	y_br = y >= bheight

	x_lim = x + bwidth if x_bl else x
	x_lim = x_lim - bwidth if x_br else x_lim

	y_lim = y + bheight if y_bl else y
	y_lim = y_lim - bheight if y_br else y_lim

	# load value
	value = ptr[y_lim*bwidth + x_lim]

	# move value to correct block
	value = transferFromWest(value) if x_bl else value
	value = transferFromEast(value) if x_br else value
	value = transferFromNorth(value) if y_bl else value
	value = transferFromSouth(value) if y_br else value

	return value
'''
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'get2D', 5, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		buffer_ptr, x, y, bwidth, bheight = self.func_args
		# take  the meta approach: describe function in highlevel code
		# and compile with local compiler
		local_compiler = Compiler(no_sequencer=codegen.compiler.config['no_sequencer'])
		main_object = local_compiler.compile(self.src)[0]

		# find out original return value
		original_return_name = main_object.return_val_name

		# now mangle names to avoid conflicts with other functions
		exclude_mangling = [x for x in self.func_args if isinstance(x, NamedValue)]
		main_code_mangled, mangle_map = codegen.replace_regnames_by_tmp_names(main_object.code, exclude_mangling)
		main_object.code = main_code_mangled

		# now map original function arguments to new ones
		mapped_arguments = []
		for arg in main_object.arguments:
			versions = sorted(mangle_name for name, mangle_name in  mangle_map.iteritems() if name.rpartition('@')[0] == arg)
			first_version = versions[0] if len(versions) else arg
			mapped_arguments.append(first_version)
		main_object.arguments = mapped_arguments

		# patch arguments
		patched_object = Compiler.patch_arguments_before_run(main_object, self.func_args)

		# wire return value to destination value
		return_name_versions = sorted(mangle_name for name, mangle_name in  mangle_map.iteritems() if name.rpartition('@')[0] == original_return_name)
		return_name = return_name_versions[-1] if len(return_name_versions) else original_return_name

		# emit code
		for x in patched_object.code: yield x
		yield ('mov', None, dest.name, return_name)


class Codegenerator(object):
	def __init__(self, compiler):
		self.compiler = compiler
		self.code = None
		self.values = None
		self.tmp_name_cnt = 0
	def get_tmp_name(self):
		self.tmp_name_cnt+=1
		return 'cg_tmp_%i@0'%self.tmp_name_cnt
	def replace_regnames_by_tmp_names(self, instr_fragment, exclude_list=[]):
		''' Replace all register names by tmp names, except for values in the exclude list. '''
		new_instr = []
		reg_map = {}
		for instr in instr_fragment:
			op, cond = instr[:2]
			new_regs = []
			for i, reg in enumerate(instr[2:]):
				# exclude renaming for special registers, values of imm instrs
				# and exclude list entries
				if reg in exclude_list or reg in SPECIAL_REGS or op == 'imm' and i == 1:
					new_regs.append(reg)
					continue
				if not reg in reg_map:
					reg_map[reg] = self.get_tmp_name()
				new_regs.append(reg_map[reg])
			new_instr.append(tuple([op, cond] + new_regs))
		return new_instr, reg_map
	def get_value(self, name):
		found = [v for n, v in self.values if n.name == name]
		if len(found) > 1:
			raise InternalCompilerException('code is not in ssa form')
		elif len(found) == 1:
			return found[0]
		raise InternalCompilerException('could not find value %s'%name)
	def sequencer_instr(self, op):
		return tuple(['x' + op[0]] + list(op[1:]))
	def add_code(self, instr, seq_op = False):
		self.code.append((self.sequencer_instr(instr) if seq_op else instr))
	def gen_ConstValue(self, name, value):
		yield ('imm', None, name.name, value.value)
	def gen_BinopValue(self, name, value):
		yield (value.op, None, name.name, value.left.name, value.right.name)
	def gen_LabelValue(self, name, value):
		yield ('label', None, name.name)
	def gen_MemDereference(self, name, value):
		if isinstance(value.ptr, NamedValue):
			ptr = self.get_value(value.ptr.name)
		else:
			ptr = value.ptr
		ind = self.get_value(value.index.name)
		if isinstance(ptr, ConstList) and isinstance(ind, ConstValue):
			num_ind = ind.value
			element = ptr.elements[num_ind]
			if isinstance(element, ConstValue):
				yield ('imm', None, name.name, element.value)
			elif isinstance(element, ConstList):
				# skip first indexing of a nested list
				# this will be handled by the second index
				yield
			elif isinstance(element, NamedValue):
				yield ('mov', None, name.name, element.name)
		elif isinstance(ptr, MemDereference):
			# double memdereference, this is an access to a 2D const list
			# first find get the element from the first dereference
			num_ind = self.get_value(ptr.index.name)
			base_array_name = ptr.ptr.name
			base_array = self.get_value(base_array_name)
			element = base_array.elements[num_ind.value]
			# the problem is now reduced to the 1D case, use the existing code
			for x in self.gen_MemDereference(name, MemDereference(element, value.index)):
				yield x
		elif isinstance(ptr, ConstValue) and isinstance(ind, ConstValue):
			num_id = ptr.value + ind.value
			yield ('memr_imm', None, name.name, num_id)
		elif isinstance(value.ptr, NamedValue) and isinstance(value.index, NamedValue):
			# if no special case, use the values without looking into them
			tmp_name = self.get_tmp_name()
			yield ('add', None, tmp_name, value.ptr.name, value.index.name)
			yield ('memr', None, name.name, tmp_name)
		else:
			raise UnImplementedFeatureException('can\'t generate code for %s and %s'%(str(name), str(value)))
	def gen_MemWriteValue(self, name, lvalue):
		ptr = self.get_value(lvalue.ptr.name)
		ind = self.get_value(lvalue.index.name)
		value = lvalue.value
		if isinstance(ptr, ConstValue) and isinstance(ind, ConstValue):
			num_id = ptr.value + ind.value
			yield ('memw_imm', None, num_id, value.name)
		elif isinstance(lvalue.ptr, NamedValue) and isinstance(lvalue.index, NamedValue):
			# if no special case, use the values without looking into them
			tmp_name = self.get_tmp_name()
			yield ('add', None, tmp_name, lvalue.ptr.name, lvalue.index.name)
			yield ('memw', None, tmp_name, value.name)
		else:
			raise UnImplementedFeatureException('can\'t generate write code for %s and %s'%(str(name), str(value)))
	def gen_CondValue(self, name, value):
		test = self.get_value(value.test.name)
		yield ('cmp', None, test.left.name, test.right.name)
		yield ('phi', test.condition(), name.name, value.body.name, value.orelse.name)
	def gen_EmitValue(self, name, value):
		yield (('xemit', None, name.name, value.value.name))
	def gen_ConstList(self, name, value):
		# Do nothing, code is generated on value access
		yield
	def gen_Comparison(self, name, value):
		# Do nothing, code is generated when compare is used
		yield
	def gen_JmpValue(self, name, value):
		test = self.get_value(value.cond.name)
		yield ('cmp', None, test.left.name, test.right.name)
		yield ('jmp', test.condition(), value.label.name)
	def gen_NamedValue(self, name, value):
		yield ('mov', None, name.name, value.name)
	def gen_PhiValue(self, name, value):
		test = self.get_value(value.cond.name)
		yield ('cmp', None, test.left.name, test.right.name)
		yield ('phi', test.condition(), name.name, value.left.name, value.right.name)
	def gen_BuiltinValue(self, name, value):
		for instr in value.gen_code(self, name):
			yield instr
	def gen_generic(self, name, value):
		print 'not implemented: ', name,value
		yield
	def gen_ArgumentPlaceholderValue(self, name, value):
		yield
	def gen_code(self, kernel_object):
		values = kernel_object.values
		#value_dict = dict((x.name, v) for x, v in values.iter_items())
		self.code = kernel_object.code
		self.values = values
		for name, value in values:
			handler = self.gen_generic
			handler_name = 'gen_' + value.__class__.__name__
			if hasattr(self, handler_name):
				handler = getattr(self, handler_name)
			elif isinstance(value, BuiltinFunction):
				handler = self.gen_BuiltinValue
			for instr in handler(name, value):
				if name.seq_value and not value.seq_value:
					raise InvalidCodeException('can\'t copy from PE registers to sequencer')
				elif not name.seq_value and value.seq_value:
					raise InternalCompilerException('an emit instruction is needed for a copy from a seq register to a PE register')
				if instr: self.add_code(instr, name.seq_value)
		return kernel_object

class RegisterAllocator(object):
	''' Linear register allocator based on Linear Scan Register Allocation M. Poletto '''
	@classmethod
	def _sort_by_inc_starting_point(cls, intervals):
		return sorted(intervals, key=lambda interval: interval[1])
	@classmethod
	def _sort_by_inc_end_point(cls, intervals):
		return sorted(intervals, key=lambda interval: max(interval[2]))
	@classmethod
	def _expire_old_intervals(cls, interval, active, reg_manager):
		to_remove = []
		for active_interval in cls._sort_by_inc_end_point(active):
			# compare endpoint and start point
			if max(active_interval[2]) >= interval[1]:
				break #make sure that last loop is executed
			else:
				to_remove.append(active_interval)
		for x in to_remove:
			reg_manager.release_reg(x[3])
			for i, y in enumerate(active):
				if x == y: del active[i]
	@classmethod
	def _spill_at_interval(cls, interval):
		raise Exception('not implemented yet, see lineair scan register allocation, massimiliano poletto pseudocode')
	@classmethod
	def register_allocation(cls, code, liveness, nr_reg):
		''' Allocate registers, make sure the liveness analysis is up to date! '''
		reg_manager = RegisterManager(nr_reg)
		intervals = [(k, v[0], v[1], None) for k, v in liveness.iteritems()]
		intervals = cls._sort_by_inc_starting_point(intervals)

		active = []
		for i, interval in enumerate(intervals):
			cls._expire_old_intervals(interval, active, reg_manager)
			if len(active) >= nr_reg:
				cls._spill_at_interval(interval)
			else:
				name, start, ends, _ = interval
				interval = (name, start, ends, reg_manager.get_free_reg())
				active.append(interval)
				intervals[i] = interval
				active = cls._sort_by_inc_end_point(active)
		reg_mapping = dict((name, reg) for name, s, ends, reg in intervals)
		for i, c in enumerate(code):
			regs = c[2:]
			regs_mod = [reg_mapping[x] if x in reg_mapping else x for x in regs]
			code[i] = tuple(list(c[0:2]) + regs_mod)
		return code, reg_mapping

	@classmethod
	def liveness_analysis(cls, code):
		''' Analyse the def/use chains of the code. '''
		liveness = {}
		for i, c in enumerate(code):
			op = c[0]
			# skip these instructions, they don't use registers
			if op in ['nop']: continue
			regs = c[2:]
			# specialcase cmp etc., their first reg is not a def reg
			use_regs = []
			if op in ['cmp', 'memw']:
				use_regs = regs
			elif op in ['memw_imm']:
				use_regs = regs[1:]
			elif op in ['memr_imm', 'imm']:
				liveness[regs[0]] = (i, [])
			elif len(regs):
				# ignore out as a allocatable reg
				if regs[0] in liveness:
					raise RegisterAlreadyDefinedException('register %s already defined'%regs[0])
				if not regs[0] in ['out']:
					liveness[regs[0]] = (i,[])
				use_regs = regs[1:]

			for r in use_regs:
				if r in ['west', 'north', 'south', 'east']: continue
				if not r in liveness:
					print '[warning] register %s is used before definition, setting def to start'%r
					liveness[r] = (0, [])
				liveness[r][1].append(i)
		# fix empty endpoints by filling them with the code size
		# XXX not shure if this is a good idea, maybe code size +1 would be better
		# to distinguish between used at last line and not used at all?
		for k, liveint in liveness.iteritems():
			d, u = liveint
			if not u:
				liveness[k] = (d, [len(code)-1])
		return liveness

	@classmethod
	def print_liveness(cls, code, liveness):
		# max interval endpoint
		max_var_len = max(len(x) for x in liveness.keys())
		max_instr_len = 50
		print ' '*max_instr_len + '|' + ''.join(x.center(max_var_len) for x in liveness.keys())
		for i, c in enumerate(code):
			code_str = str(c)
			res = code_str.ljust(max_instr_len) + '|'
			for k, v in liveness.iteritems():
				s, ends = v
				symbol = ' '
				if s == i:
					symbol = 'D'
				elif i in ends:
					symbol = 'U'
				elif i > s and i < max(ends):
					symbol = '|'
				res += symbol.center(max_var_len)
			print res

class Compiler(object):
	''' Simple compiler for subset of Python, all function calls are inlined. '''

	default_config = {'no_sequencer':False}
	def __init__(self, **kwargs):
		self.src = None
		self.kernel_asts = []
		self.config = dict(self.default_config)
		for k, v in kwargs.iteritems():
			if k in self.config:
				self.config[k] = v
		self.builtin_functions = BuiltinFunction.gather_valid_builtins()
		self.unique_id_cnt = 0
	def is_builtin_function(self, func_name):
		return func_name in self.builtin_functions
	def get_builtin_function(self, func_name):
		return self.builtin_functions[func_name] if (func_name in self.builtin_functions) else None
	def get_kernel_ast(self, name):
		for a in self.kernel_asts:
			if a.name == name: return a
		return None
	def get_unique_id(self):
		self.unique_id_cnt += 1
		return self.unique_id_cnt
	def compile_kernel_to_values(self, f, kernelObject):
		if f.args.kwarg or f.args.vararg or f.args.defaults:
			raise InvalidCodeException('kernels do not support varargs or kwargs')
		print 'compile function %s'%f.name

		varcontext = kernelObject.varcontext

		# Convert abstract syntax tree to Value representation
		varcontext.push()

		arguments = [x.id for x in f.args.args]
		print 'arguments: ', arguments
		kernelObject.arguments = arguments

		visitor = BodyVisitor(self, kernelObject)
		for arg in arguments: 
			arg_varname = varcontext.def_var(arg)
			visitor.add_value(arg_varname, ArgumentPlaceholderValue())

		for s in f.body:
			visitor.visit(s)

		varcontext.pop()

		#return self.remove_metavalues(kernelObject)
		return kernelObject

	def compile(self, src, kernel_functions=[]):
		self.src = src
		src_ast = ast.parse(src)
		print src

		functions = [x for x in src_ast.body if isinstance(x, ast.FunctionDef)]
		for func in kernel_functions:
			functions.append(self.load_ast_from_function(func))
		
		self.kernel_asts = [x for x in functions if ast_is_kernel(x)]

		kernel_objects = []
		for f in self.kernel_asts:
			if f.args.kwarg or f.args.vararg or f.args.defaults:
				raise InvalidCodeException()

			kernelObject = KernelObject(f.name)
			kernelObject = self.compile_kernel_to_values(f, kernelObject)
			
			# Convert Value representation to code
			print 'values'
			print '\n'.join('%s: %s'%(str(var), str(value)) for var, value in kernelObject.values)
			print 
			codegen = Codegenerator(self)
			kernelObject = codegen.gen_code(kernelObject)
			kernel_objects.append(kernelObject)

		return kernel_objects

	@classmethod
	def load_ast_from_function(cls, func):
		func_name = func.func_name
		module = func.__module__
		if module == '__main__':
			module = __file__.replace('.py','')

		all_file = None
		try:
			f = open(module + '.py')
			all_file = ast.parse(f.read())
			f.close()
		except Exception, e:
			print 'could not find module %s'%module
			print 'error: %s'%str(e)
			return

		func_defs = [x for x in all_file.body if isinstance(x, ast.FunctionDef)]

		try:
			current_ast = [x for x in func_defs if x.name == func_name][0]
		except Exception, e:
			print 'could not find function with name %s'%func_name
			print 'error: %s'%str(e)
			return
		return current_ast


	def register_allocation(self, kernelObject, nr_reg):
		''' Wrap register allocator for compiler. '''
		code, reg_mapping = RegisterAllocator.register_allocation(kernelObject.code, kernelObject.liveness, nr_reg)
		kernelObject.code, kernelObject.reg_mapping = code, reg_mapping
		return kernelObject

	def liveness_analysis(self, kernelObject):
		code = kernelObject.code
		kernelObject.liveness = RegisterAllocator.liveness_analysis(code)
		return kernelObject

	def replace_phi_nodes(self, kernel_object):
		new_code = []
		code = kernel_object.code
		for instr in code:
			if instr[0] == 'phi':
				_, cond, res, left, right = instr
				new_code.append(('mov', None, res, right))
				new_code.append(('mov', cond, res, left))
			else:
				new_code.append(instr)
		kernel_object.code = new_code
		return kernel_object

	def opt_peephole(self, kernel_object):
		# we need code in SSA form, before register allocation!!!
		try:
			kernel_object = self.liveness_analysis(kernel_object)
		except RegisterAlreadyDefinedException:
			print 'warning: code not in SSA, skipping peephole opt'
			return kernel_object

		def get_def_instr(reg):
			return kernel_object.liveness[reg][0] if not reg.split('@')[0] in kernel_object.arguments else -1
		def is_unconditional_imm_instr(instr):
			return instr[0] == 'imm' and not instr[1]

		code = kernel_object.code
		code = [x for x in code if not (x[0] == 'mov' and x[2] == x[3])]
		for i, instr in enumerate(code):
			if instr[0] in ['add', 'sub', 'mul', 'or', 'xor', 'and']:
				op, cond, dest, src1, src2 = instr
				def_location = [get_def_instr(x) for x in [src1, src2]]
				def_srcs = [code[x] if x >= 0 else None for x in def_location] 
				uncond_imm_instrs = [is_unconditional_imm_instr(x) if x else False for x in def_srcs]
				other_srcs = (src2, src1)
				if all(uncond_imm_instrs):
					# if both instructions are unconditional, just calculate them and replace by constant
					op_map = {'or': 'or_', 'and': 'and_'}
					opname = op if not op in op_map else op_map[op]
					func = getattr(operator, opname)
					res = func(def_srcs[0][3], def_srcs[1][3])
					code[i] = ('imm', cond, dest, res)
				elif instr[0] in ['add', 'or']:
					# simplify add/or if one of the operands is zero
					for def_src, other_src in zip(def_srcs, other_srcs):
						if def_src and def_src[0] == 'imm' and not def_src[1] and def_src[3] == 0.:
							code[i] = ('mov', cond, dest, other_src)
				elif instr[0] == 'sub':
					def_src = def_srcs[1]
					other_src = other_srcs[1]
					if def_src and def_src[0] == 'imm' and not def_src[1] and def_src[3] == 0.:
						code[i] = ('mov', cond, dest, other_src)
				
		kernel_object.code = code
		return kernel_object

	def opt_copy_propagation(self, kernel_object):
		# we need code in SSA form, before register allocation!!!
		try:
			kernel_object = self.liveness_analysis(kernel_object)
		except RegisterAlreadyDefinedException:
			print 'warning: code not in SSA, skipping copy propagation opt'
			return kernel_object

		# find candidates for copy propagation, for now we only consider
		# cases with one def/ on use, with the def in a mov statement
		def valid_src(src):
			dirs = ['west', 'north', 'east', 'south']
			return src not in dirs
		def valid_dest(dest):
			if '@' in dest: # find symbolic name if name has a version part
				sym_dest = VariableContextManager.symbolic_name(dest)
			else:
				sym_dest = dest
			valid = not '__return' in dest and sym_dest != 'out'
			return valid

		def valid_candidate(instr):
			if not instr[0] == 'mov': return False
			return valid_src(instr[3]) and valid_dest(instr[2])
		def find_candidate(code, start_line = 0):
			for i, c in enumerate(code):
				if i < start_line: continue
				if valid_candidate(c): return i
			return None
		new_code = kernel_object.code[:]
		cind = find_candidate(new_code)
		while cind:
			instr = new_code[cind]
			dest = instr[2]
			src = instr[3]
			for i in xrange(cind, len(new_code)):
				c = new_code[i]
				regs = c[2:]
				regs_mod = [src if r == dest else r for r in regs]
				new_code[i] = tuple(list(c[0:2]) + regs_mod)
			
			cind = find_candidate(new_code, cind+1)
			
		kernel_object.code =  [x for x in new_code if not (x[0] == 'mov' and x[2] == x[3])]
		return kernel_object

	@classmethod
	def remove_metavalues(cls, kernel_object):
		# XXX note that this should be generalised to dead code elimination
		# this is an easy case since metavalues should only be defined and not used
		# after full processing
		def is_metavalue(instr):
			return any(x for x in instr[2:] if MetaValue.is_instance(x))
		kernel_object.code = [x for x in kernel_object.code if not is_metavalue(x)]
		return kernel_object

	@classmethod
	def print_object_code(cls, kernel_object):
		print '\n'.join(str(x) for x in kernel_object.code)

	@classmethod
	def patch_arguments_before_run(cls, kernel_object, args):
		''' Fill in kernel call arguments before execution.
		
		This works for code with or without the regalloc pass already executed. '''
		patched_object = copy.copy(kernel_object)
		kernel_args = patched_object.arguments
		reg_mapping = patched_object.reg_mapping
		varcontext = patched_object.varcontext
		code = patched_object.code
		if len(kernel_args) != len(args):
			raise Exception('invalid number of arguments for call, %i required, got %i'%(len(kernel_args),len(args)))

		new_code = []
		for argname, arg in zip(kernel_args, args):
			if not '@' in argname: # make this work for symbolic names also
				var_arg = VariableContextManager.gen_variable_name(argname, 0)
			else:
				var_arg = argname
			if reg_mapping.keys() != []:
				# non-empty means that register allocation was done already
				var_arg = reg_mapping[var_arg]
			if isinstance(arg, NamedValue):
				# symbolic/register arguments are copied
				new_code.append(('mov', None, var_arg, arg.name))
			else:
				# literal constants are implemented as immediates
				new_code.append(('imm', None, var_arg, arg))
		for c in code: new_code.append(c)
		patched_object.code = new_code
		return patched_object

class CompilerDriver(object):
	def __init__(self, nr_registers, **kwargs):
		self.nr_registers = nr_registers
		self.compiler = Compiler(**kwargs)
	def run(self, src, verbose = False):
		kernel_objects = self.compiler.compile(src)
		main_object = [x for x in kernel_objects if x.function_name == 'main'][0]
		self.compiler.print_object_code(main_object)
		
		main_object = self.compiler.opt_copy_propagation(main_object)
		if verbose:
			print '#'*100
			self.compiler.print_object_code(main_object)

		main_object = self.compiler.liveness_analysis(main_object)
		if verbose:
			print '#'*100
			RegisterAllocator.print_liveness(main_object.code, main_object.liveness)

		main_object = self.compiler.register_allocation(main_object, self.nr_registers)
		if verbose:
			print '#'*100
			self.compiler.print_object_code(main_object)
		
		main_object = self.compiler.opt_peephole(main_object)
		if verbose:
			print '#'*100
			self.compiler.print_object_code(main_object)

		main_object = self.compiler.replace_phi_nodes(main_object)
		if verbose:
			print '#'*100
			self.compiler.print_object_code(main_object)

		return main_object

@kernel
def inlinekerneltest():
	x = 3

if __name__ == '__main__':
	src = '''
@kernel
def add1(x):
	return x + 1

@kernel
def main():
	d = [1, 3, 4]
	a = 1
	b = a + 2
	b += 3
	q = d[2]
	z = a[3]

	a[2] = 3

	p = b if q > 0 else a
	t = add1(b)

	for i in xrange(3):
		a += i
'''
	src2 = '''
@kernel
def add1(x):
	q = x + 1
	return q
#	return x + 1

@kernel
def main():
	b = 3
	t = add1(b)
'''
	src3 = '''
@kernel
def main():
	b = loadWest()
'''
	compiler = Compiler()

	kernel_objects = None 
	try:
		kernel_objects = compiler.compile(src3, [inlinekerneltest])
	except Exception, e:
		print str(e)
		traceback.print_tb(sys.exc_traceback)
		pdb.post_mortem(sys.exc_traceback)
		exit(1)
		
	main_object = [x for x in kernel_objects if x.function_name == 'main'][0]

	print '\n'.join(str(x) for x in main_object.code)


