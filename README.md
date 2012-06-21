#BLIP Simulator

##Requirements
The library depends on python >= 2.5 or pypy >= 1.5
It is adviced to use pypy instead of python, as this speeds up simulation x4!

Please note that the repo dir should be added to PYTHONPATH before executing the code:
```bash
export PYTHONPATH=$PYTHONPATH:path/to/repo
```

##Documentation

In order to generate the documentation, install Sphinx and execute the following commands:
```bash
cd doc
make html
```
the files are now in doc/_build/html

The documentation can also be found online: http://telin.ugent.be/~mslembro/BLIP


##Main entry points

 * Pure Python Viola Jones implementation
```bash
python violajones/reference.py
```

 * BLIP interpreter running default program
```bash
python blip/simulator/interpreter.py
```

 * BLIP interpreter running Viola Jones codegen
```bash
python violajones/gen_code.py
```

 * BLIP interpreter running planarity filters
```bash
python planarity/gen_code.py
```


##Tests

Most parts of this projects are covered with unit and integration tests.
These can be executed using the buildin testing tool (tools/tester.py):

```bash
cd test
{pypy/python} test_all.py
```

It is adviced however to install and use py.test instead
```bash
cd test
py.test .
```

##File guide:

###Blip interpreter

 * blip/simulator/interpreter.py BLIP high-level simulator

 * blip/simulator/opcodes.py
   Description of BLIP instruction format and Code object

###Code generation and compilers

 * blip/code/codegen.py
   Code object, utility functions

 * blip/code/trace_optimiser.py
   Instruction level binary runtime code optimiser

 * blip/code/BlipCompiler.py
   Experimental compiler for an OpenCl dialect executable on the Blip array

 * blip/code/skeletons.py
   Experiment with algebraic skeletons for the blip code generator

###Example code

 * blip/blipcode.py
   Various code generators, mostly for image processing

###Viola Jones specific:

 * violajones/parse_haar.py
   Parsing of OpenCV haar cascade XML files

 * violajones/gen_code.py
   Main entry point for running VJ on top of BLIP

 * violajones/draw.py
   Draw output of VJ, mostly used for debugging shape splitting

 * violajones/reference.py
   Pure Python reference implementation

###Planarity filter specific:

 * planarity/filters.py
   Parsing of OpenCV filterbank XML files

 * planarity/gen_code.py
   Main entry point for running planarity filters on top of BLIP

 * planarity/reference.py
   Pure Python reference implementation

###Analysis:

 * blip/analysis/analysis.py
   Main entry point for BLIP analysis

 * blip/analysis/sweep_analysis.py
   Batch processing of BLIP analysis, sweep across parameters

###Support library:

 * blip/support/png.py
   Pure Python PNG image handling

 * blip/support/svgfig.py
   Pure Python SVG image generation
   
 * blip/support/imageio.py
   Wrapper around OpenCV or png.py image handling

