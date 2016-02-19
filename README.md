# Decision-Trees
Creates a decision tree with the training data and then predicts on the test data using the created tree. The file dt-learn needs to be 
invoked in the following way
dt-learn train-set-file test-set-file m.
Here m is the upper bound for the number of training instances in a leaf node i.e. reaching a node with fewer nodes than m is a termination
condition. Information gain is used while chosing between features.
