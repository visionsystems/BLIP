import parse_haar
import sys

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print 'usage: %s cascade.xml outfile.txt'
		exit(1)

	in_filename, out_filename = sys.argv[1:3]
	cascade = parse_haar.parse_haar_xml(in_filename)
	
	with open(out_filename, 'w') as outf:
		for i, stage in enumerate(cascade.stages):
			outf.write('stage %i\n'%i)
			for j, feature in enumerate(stage.features):
				outf.write('  feature %i\n'%j)
				for k, shape in enumerate(feature.shapes):
					outf.write('    %s\n'%str(shape[0]))
