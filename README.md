# A Comparative Survey: Benchmarking for Pool-based Active Learning

This is an implementation of our IJCAI 2021 Survey Track paper  "[A Comparative Survey: Benchmarking for Pool-based Active Learning](https://www.ijcai.org/proceedings/2021/0634.pdf)"

***
## Code
We utilized three existing libraries here: [libact](https://github.com/ntucllab/libact), [google active learning toolbox](https://github.com/google/active-learning) and [ALiPy](https://github.com/NUAA-AL/ALiPy).

- libact: US, QBC, HintSVM, QUIRE, VR, ALBL, DWUS
- Google active learning: Random/Uniform sampling, KCenter, Graph, Margin, Hier, MCM
- ALiPy:  EER, LAL, BMDR, SPAL
***

## Prerequisites 
Major libraries are listed here:

- libact
- ALiPy
- scikit-learn
- cvxpy
- numpy
- scipy
- tensorflow


You can also use the following command to install conda environment

```
conda env create -f environment.yml
```

***

## Data

We employed 35 public datasets, which are uniformly dealed with libsvm style, as shown in `Dataset/dealeddata/` file folder.

*** 

## Demo
Here we take `appendicitis` dataset as binary-class classification task example and `thyroid` dataset as multi-class classification task example, with batch size 5, repeat 100 trials, initial labeled set with size 20. 

General format: python baseline-[library name]-[binary/mulitple].py [dataset name] [AL model name] [AL batch size] [number of repeat trials] [initial labeled set size] 

- libact: take AL algorithm QBC
    1. binary-class classification: `python baseline-libact-binary.py appendicitis-svmstyle QBC 5 100 20`
    2. multi-class classification: `python baseline-libact-multiple.py thyroid-svmstyle QBC 5 100 20`
- Google active learning: take AL algorithm Hier as example
    1. binary-class classification: `python baseline-google-binary.py appendicitis-svmstyle Hier 5 100 20`
    2. multi-class classification: `python baseline-google-multiple.py thyroid-svmstyle Hier 5 100 20`
- ALiPy: take AL algorithm EER as example
    1. binary-class classification: `python baseline-alipy-binary.py appendicitis-svmstyle QueryExpectedErrorReduction 5 100 20`
    2. multi-class classification: `python baseline-alipy-multiple.py thyroid-svmstyle QueryExpectedErrorReduction 5 100 20`

**Please read `notes.txt` before running the code, especially for BMDR, SPAL and LAL!!!**

***

## Citing
If you use our code in your research or applications, please consider citing our paper.

```
@inproceedings{zhan2021comparative,
  title={A Comparative Survey: Benchmarking for Pool-based Active Learning.},
  author={Zhan, Xueying and Liu, Huan and Li, Qing and Chan, Antoni B},
  booktitle={IJCAI},
  pages={4679--4686},
  year={2021}
}
```

***

## Contact
If you have any further questions or want to discuss Active Learning with me, please contact <xyzhan2-c@my.cityu.edu.hk> (my spare email is <sinezhan17@gmail.com>).

