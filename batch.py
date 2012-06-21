import blip.simulator.interpreter
from blip.code.codegen import load_codegen, get_codegen_parameters # wrap_codegen commented, not in that file!
from blip.code.codegen import InvalidCodegenArgumentsException
    
import pickle

b_s = (32,64,128)
results = [[] for i in xrange(len(b_s))]
output_buffers = [[] for i in xrange(len(b_s))]
k = 0

#image = ('data/worms2.png',42)
image = ('spiral2',1)

for blsize in b_s:
    t = int(0);
    stop = False
    while not stop:
        codegen_impl = 'blip.blipcode.gen_conn_comp_lbl3'
        module_name, _, codegen_name = codegen_impl.rpartition('.')
        codegen_impl = load_codegen(module_name, codegen_name)
        interpreter = blip.simulator.interpreter.main(block_size=(blsize,blsize), code_gen = codegen_impl, args = {'number_of_runs':int(t)}, image_filename = 'data/'+image[0]+'.png', output_filename = 'out/'+image[0]+'_' + str(blsize) + '_' + str(t) + '.png')
        results[k].append(interpreter.temp)
        output_buffers[k].append(interpreter.get_output_buffer())
        stop = int(interpreter.temp) <= image[1]
        t+=1
    k += 1

for i in xrange(len(results)):
    print 'result ' + str(b_s[i])
    for j in xrange(len(results[i])):
        print results[i][j]
        
pickle.dump( results, open( "resultscolor_"+ str(image[0]) + ".p", "wb" ) )
pickle.dump( output_buffers, open( "output_buffers_"+ str(image[0]) + "p", "wb" ) )


