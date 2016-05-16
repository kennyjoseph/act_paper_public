Intro
======

This repository is the public repository for the ACT-based work that we have been doing with respect to analyzing newspaper data. 

This code base and the documentation will evolve over time, but right now you need to do at least the following to get it into shape:

1. For the dependency_parsing Java code, you'll need to get the following jars from the stanford website and put them in the libs folder in that directory:

	- stanford-corenlp-3.4.1-models.jar 
	- stanford-srparser-2014-08-28-models.jar  
 
2. Change the value of the variable ```top_dir``` in the files ```results.py```,  ```run_best_full_results.py``` and ```handle_dep_parse_data.R``` and change the variable ```wd``` in ```handle_raw_act_data.R``` to whatever absolute path you put this code on.

3. Extract k_fold_tmp_test

4. In the python/src directory, you have to build the cython extensions using:
```python setup.py build_ext --inplace```

The full dependency parse data is too big to put on github, but its available [here](https://www.dropbox.com/s/tbzrgwrvqcdx9nz/dep_parse_all.tsv?dl=0).  This is essentially the output of the java code (after cat'ing all of the results for each article together). Its then handled by ```handle_dep_parse.R``` to generate the data used by the python model. To run the python models, simply go into the python/src directory and run ```python results.py``` (after doing the steps above) and then ```python run_best_model.py```. If that doesn't work, check all paths are correct. If THAT doesn't work, email me!

All plots generated in the paper are created in ```gen_results.R```.

Citation
========

If you use this code, please cite our JMS article!

```
@article{joseph2016social,
  title={A social-event based approach to sentiment analysis of identities and behaviors in text},
  author={Joseph, Kenneth and Wei, Wei and Benigni, Matthew and Carley, Kathleen M},
  journal={The Journal of Mathematical Sociology},
  pages={1--30},
  year={2016},
  publisher={Routledge}
}

```

License
=========
All code in this repository follows the MIT License agreement.

```
The MIT License (MIT)

Copyright (c) 2014 Kenneth Joseph, Peter M. Landwehr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
