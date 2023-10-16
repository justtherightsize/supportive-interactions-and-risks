README
======================
This is a collection of scripts for training the models. Follow these instructions. You will need your own data - see 6.

1. Dependencies for the env: requirements.txt

2. Scripts for classification with previous context are in: the folder previous_context, scripts for seq2seq are in the folder seq2seq.

3. The scripts from are setup to run without cross-validation but can be run with it. To run them in terminal one must do:
	a. create a new config file in the configs dir. IMPORTANT: change the config's id inside the config too!!, not just the filename. See existing configs for reference.
	b. in run_telamon.sh change: filename of the script to run, config id. Now it is setup to only run the classification once. The rest of the script is for crossvalidation. Otherwise, the splits are coded with the prefix split-1...N.
	c. to run the classification, run from the irtisdb dir: run_telamon.sh

4. If you want to run the script directly {eacl_bert_coarse.py, eacl_bert_multi.py}, see the args in run_telamon.sh, e.g.:
    a. cd irtisdb
    b. <env>/python bert_cls/eacl_bert_coarse.py <path_to_config> <split_n is 1 without crossvalidation> <GPU>

4. The script eacl_bert_coarse.py contains all the code for loading the data, resampling, classification, evaluation, and plots so it can be easily copied beween servers. It uses arguments from the global dictionary ARGS throughout the script. ARGS are loaded from config files (see 3a.)

5. the run() method of eacl_bert_coarse.py (and variants) first loads the data from the dataset files. They are not versioned. The dataset files are coded with 'tag' and 'dir'. e.g. the dataset filename will look like this: outputs/split-1_eacl.train where 'tag'='eacl', 'dir'='outputs'

6. The data are split into three dataset files (e.g. split-1_eacl.train) in 80:5:15 train/dev/test. Dev is for model selection. The dataset files are json files with this structure:
	{	'coarse' = [],
		'multi' = []
	}
Each of the root-level keys contains the same training examples as tuples with these label types:
	coarse = [('text', 0 or 1, _, _), ...] 

	multi = [('text', [1, 0, 0, 0, 0, 0], _, _), ...] # the vector is an n-hot vector of length 6. 	a special case is [1, 0, 0, 0, 0, 0] which is the 0 class which is exclusive with others such as [0, 0, 1, 0, 0, 1] which would mean that the exampl has labels 2 and 5.
