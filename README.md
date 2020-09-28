# weka2json
A utility to convert a pretrained Weka REPtree bagging model to JSON format, as used in [ObfusX](https://github.com/wei-zeng/obfusX) project.

Currently it only supports Weka bagging model with REPtree as the base learner.

## Requirements
- Weka >= 3.8
- Java >= 1.4

## Compilation
Assuming `weka.jar` from Weka installation is available in `/path/to/weka.jar`: 

`javac -cp ".:/path/to/weka.jar" REPTreeBagging2JSON.java`

This will yield a compiled `REPTreeBagging2JSON.class`.

## Usage
Assuming a Weka REPTree bagging model is saved as `bg.model`:

`java REPTreeBagging2JSON bg` (without `.model`)

This will yield a converted JSON file `bg.json`.
