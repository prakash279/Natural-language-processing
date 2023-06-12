--------------------------------------------------------------------------------------------------
Text Classification Models.
--------------------------------------------------------------------------------------------------
This repository contains two different approaches to text classification: Spacy and Support Vector Machine (SVM).

--------------------------------------------------------------------------------------------------
Spacy
--------------------------------------------------------------------------------------------------
Spacy is a easy and simple to use library created in python which helps in machine learning.

To Create the output model on your computer you need to have the data csv file and the code.

Then the code converts the data from .csv to .spacy in two different parts one being training while other being validation data.

Once the data is created in .spacy format you need to create a base_config.cfg.

which can either created from the website below with your own speicfications.

You need to set the path for train.spacy and valid.spacy the one created above.

https://spacy.io/usage/training
once you create the base_config.cfg you are supposed to run the following command in command prompt

python -m spacy init fill-config base_config.cfg config.cfg
this command will create a config.cfg file 

now you can start creating an output model using the command
python -m spacy train config.cfg --output ./output
(Specifications like using GPU or any another vectors library can be done from the above website as it offers a lot of specifications)

Once the output model is created it will tell you it's accuracy and the model can be tested on new data by using the output_test.py file.

--------------------------------------------------------------------------------------------------
Support Vector Machine
--------------------------------------------------------------------------------------------------

It as an algorithm used in machine learning which categorizes two different data in two different modules.

To run this code you need to provide a dataset you can use the one already uploaded(stress.csv)

Then there are parameters for accuracy,for testing and training which can be changed and can also work without being changed.

At the end we can see a line to test the model where we can give it a line that is not present in dataset for the classification it was trained for and it provides the label that we need.
