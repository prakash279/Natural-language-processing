# Text Classification Models

This repository contains two different approaches to text classification:
* Spacy Text Classification
* Support Vector Machine (SVM)

## Spacy

Spacy is an industrial grade Natural Language Processing (NLP) library created in python which can be used to create ML models.

To create a valid output model for us to use in our projects we need to have a *dataset* (in most cases a csv file) and we need to create a *python file* to write those models with the correct algorithm.
We also need to write a small code to convert and split the data so that we can work with it efficiently. The code will split the dataset into a training part and a testing part. The testing data is also known as Validation data. Remember to set a path for train.spacy and valid.spacy in the above code.

Once the data has been modified to our use, we can create a base_config.cfg file. Spacy offers a base config file an it can be later modified for the use of different components. The config.cfg can be downloaded [here](https://spacy.io/usage/training#quickstart). For training we only need a single configuration file that includes all settings and hyperparameters.
 
After you’ve saved the starter config to a file base_config.cfg, you can use the init fill-config command to fill in the remaining defaults. 

```
python -m spacy init fill-config base_config.cfg config.cfg
```

This command will fill in the defaults for you and create config.cfg file 

Now you can start training with spacy!

```
python -m spacy train config.cfg --output ./output
```

Once the output model is created it will tell you it's accuracy and the model can be tested on new data by using the output_test.py file.


## Support Vector Machine

It as an algorithm used in machine learning which categorizes two different data in two different modules.

To run this code you need to provide a dataset you can use the one already uploaded(stress.csv)

Just like spacy model throws accuracy percentage, we can configure our code to use the inbuilt function to check accuracy and throw it on the console. 



At the end we can see a line to test the model where we can give it a line that is not present in dataset for the classification it was trained for and it provides the label that we need.
