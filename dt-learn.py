import sys
import math
import re
from collections import OrderedDict
import random
import numpy
import matplotlib.pyplot as plt
features = OrderedDict({})
trainingData = []
testFeatures = OrderedDict({})
testData = []
class_attribute = []
def parse_file(filename):
    global lines, f, lines_read, l, m, fvalues, values, index, value
    data = []
    lines = None
    with open((filename), 'r')as f:
        lines = f.read().splitlines()
    lines_read = 1  # @relation line ignored
    for l in lines[lines_read:]:
        m = re.search(r'@attribute', l)
        if m:
            # @attribute 'age' real
            m = re.search(r'@attribute \'([a-zA-Z0-9]+)\' {\s*(.*)\s*}', l)
            if m:
                # @attribute 'sex' { female, male}
                features[m.group(1)] = m.group(2).replace(' ', "").split(',')
            else:
                m = re.search(r'@attribute \'([a-zA-Z0-9]+)\'', l)
                features[m.group(1)] = 'real'
        else:
            break  # assumption, all attributes are declared in order at the beginning of the file
    lines_read += len(features)
    lines_read += 1  # @data line ignored
    print features
    fvalues = features.values()
    for l in lines[lines_read:]:
        # 63,male,typ_angina,145,233,t,left_vent_hyper,150,no,2.3,down,0,fixed_defect,negative
        values = l.split(',')
        for index, value in enumerate(values):
            if fvalues[index] == 'real':
                values[index] = float(value)
        data.append(values)
    class_attribute = features[features.keys()[-1]]
    del features[features.keys()[-1]]  # class attribute ignored; assuming it is the last element
    return features, data, class_attribute


def splitData(data, element, feature_index):
    elementInstances = []
    for d in data:
        if d[feature_index] == element:
            elementInstances.append(d)
    return elementInstances


def findPosNegCounts(data):
    negativeCount = 0
    positiveCount = 0
    for d in data:
        if d[-1] == 'negative':
            negativeCount = negativeCount + 1
        else:
            positiveCount = positiveCount + 1
    return negativeCount, positiveCount


def findEntropy(data):
    posNegCounts = findPosNegCounts(data)
    totalCount = len(data)
    if totalCount == 0:
        return 0
    # # TODO: validate this assumption
    if posNegCounts[0] * posNegCounts[1] == 0:
        return 0
    negFraction = (posNegCounts[0] / float(totalCount)) or 1
    posFraction = (posNegCounts[1] / float(totalCount)) or 1
    return -1 * ((posFraction * (math.log(posFraction, 2))) + (negFraction * (math.log(negFraction, 2))))


def findInfoGainForThreshold(data, Left, Right):
    parentEntropy = findEntropy(data)
    parentTotal = len(data)
    LTotal = len(Left)
    RTotal = len(Right)
    return parentEntropy - (
        ((LTotal / float(parentTotal)) * findEntropy(Left)) + ((RTotal / float(parentTotal)) * findEntropy(Right)))


def findThreshold(data, position):
    keyList = []
    infogains = []
    for d in data:
        if d[position] not in keyList:
            keyList.append(d[position])
    i = 0
    keyList = sorted(keyList)
    if len(keyList) == 1:
        return (keyList[0], 0)
    while i < (len(keyList) - 1):
        threshold = (keyList[i] + keyList[i + 1]) / 2.0
        L = []
        R = []
        for d in data:
            if d[position] <= threshold:
                L.append(d)
            else:
                R.append(d)
        i = i + 1
        infoTuple = (threshold, findInfoGainForThreshold(data, L, R))
        infogains.append(infoTuple)
    return sorted(infogains, key=lambda x: x[1], reverse=True)[0]


def numericSplit(data, feature_index):
    threshold = findThreshold(data, feature_index)[0]
    leftArray = []
    rightArray = []
    for d in data:
        # print "Culprit-------------"
        # print 'Threshold', threshold
        # print 'feature_values',float(d[feature_index])
        # print 'Is',d[feature_index],'>',threshold,':',(float(d[feature_index])>threshold)
        if d[feature_index] > threshold:
            rightArray.append(d)
        else:
            leftArray.append(d)
    return leftArray, rightArray


def findInformationGain(data, feature, feature_index):
    parentEntropy = findEntropy(data)
    # print 'Parent Entropy %f' % parentEntropy
    parentTotal = len(data)
    if features[feature] == 'real':
        # print 'Feature is numerical'
        L, R = numericSplit(data, feature_index)
        LTotal = len(L)
        RTotal = len(R)
        return parentEntropy - (
            ((LTotal / float(parentTotal)) * findEntropy(L)) + ((RTotal / float(parentTotal)) * findEntropy(R)))
    else:
        # print 'feature is nominal'
        s = 0
        for element in features[feature]:
            matchingData = splitData(data, element, feature_index)
            # if len(matchingData) == 0:
            # continue
            subArrayTotal = len(matchingData)
            s = s + ((subArrayTotal / float(parentTotal)) * findEntropy(matchingData))
        return parentEntropy - s


# returns
# feature index of the best feature
# information gain by that feature in data
def findBestCandidate(data, features):
    i = 0
    infoGains = []
    for f in features.keys():
        # print "\n\n**Feature", f, "\nIndex", i
        infoTuple = (i, findInformationGain(data, f, i))
        infoGains.append(infoTuple)
        i = i + 1
    sortedList = sorted(infoGains, key=lambda x: x[0])
    return sorted(sortedList, key=lambda x: x[1], reverse=True)[0]


# ----------------------------------------------------------------------------------------

def predict_class(head, instance):
    # if head.class_type is not None:
    # return head.class_type
    if head.feature is None:
        return head.class_type  # decision tree prediction

    keyList = features.keys()
    feature_value = instance[keyList.index(head.feature)]

    if features[head.feature] == 'real':
        if feature_value <= head.children[0].feature_values:
            return predict_class(head.children[0], instance)
        else:
            return predict_class(head.children[1], instance)
    else:
        matching_child = None
        for c in head.children:
            if c.feature_values == feature_value:
                matching_child = c
                break
        return predict_class(matching_child, instance)


class Node:
    # feature = None  # slope
    # parent = None  # anotehr node
    # children = []  # some nodes
    # feature_values = []  # [up, down, flat] # TODO think this way or store conditions
    # class_type = None  # negative or positive
    # negPosCount = None  # number of +ves and -ves at this node

    def __init__(self, feature=None, parent=None, children=[], feature_values=[], class_type=None,
                 negPosCount=None):
        self.feature = feature
        self.parent = parent
        self.children = []
        self.feature_values = feature_values
        self.class_type = class_type
        self.negPosCount = negPosCount

    def add_child(self, node):
        self.children.append(node)


class Tree:
    # head = None  # head node for the tree

    def __init__(self):
        self.head = Node()

    # def add_node(self, node_to_add, nodes_parent):
    # nodes_parent.

    def createTree(self, data, head, depth, n):
        s = get_class_type(data)
        if s:
            # base case
            head.class_type = s  # positive or negative
            print ':', head.class_type
            return head
        if len(data) < n:  # TODO this should check if it is less than n and should not be hardcoded
            head.class_type = 'negative'
            return head
        best_feature_index, info_gain = findBestCandidate(data, features)
        if info_gain == 0:
            negPosCount = findPosNegCounts(data)
            if negPosCount[0] < negPosCount[1]:
                head.class_type = class_attribute[1]
            else:
                head.class_type = class_attribute[0]

            print ':', head.class_type
            return head
        else:
            print

        f = features.keys()[best_feature_index]
        head.feature = f
        if features[f] == 'real':
            for i, filteredData in enumerate(numericSplit(data, best_feature_index)):
                child = Node(negPosCount=findPosNegCounts(filteredData))
                child.feature_values = findThreshold(data, best_feature_index)[0]

                for x in range(0, depth): print '|\t',
                if i == 0:
                    print head.feature, '<=', findThreshold(data, best_feature_index)[0], '[', child.negPosCount[
                        0], \
                        child.negPosCount[1], ']',
                else:
                    print head.feature, '>', findThreshold(data, best_feature_index)[0], '[', child.negPosCount[0], \
                        child.negPosCount[1], ']',
                remainingfeatures = features.copy()
                del remainingfeatures[f]

                head.children.append(child)
                self.createTree(filteredData, child, depth + 1, n)

        else:
            for element in features[f]:
                filteredData = splitData(data, element, best_feature_index)
                child = Node(negPosCount=findPosNegCounts(filteredData))
                child.feature_values = element

                for i in range(0, depth): print '|\t',
                print head.feature, '=', element, '[', child.negPosCount[0], child.negPosCount[1], ']',
                remainingfeatures = features.copy()
                del remainingfeatures[f]

                head.children.append(child)
                self.createTree(filteredData, child, depth + 1, n)

        return head

    # eg: (56,'male','atyp_angina',120,236,'f','normal',178,'no',0.8,'up',0,'normal') => (negative)
    def find_class(data):
        # traverses the tree and finds the class
        pass


def get_class_type(data, features=None):
    '''
    gets the class type if this is a stopping phase of data
    :param data:
    :param features:
    :return: positive or negative class if this is a stopping phase, else None
    '''
    posNegCounts = findPosNegCounts(data)
    if posNegCounts[0] == 0:
        return 'positive'
    elif posNegCounts[1] == 0:
        return 'negative'
    return None

def main():
    argumentList=(sys.argv)
    n = int(argumentList[3])
    trainingFileName = argumentList[1]
    testFileName = argumentList[2]
    features,trainingData,class_attribute=parse_file(trainingFileName)
    testFeatures,testData,class_attribute=parse_file(testFileName)
    t = Tree()
    t.createTree(trainingData, t.head, 0, n)
    i = 1
    sum = 0
    print '<Predictions for the Test Set Instances>'
    for d in testData:
        prediction = predict_class(t.head,d)
        realClass = d[-1]
        print i,': Actual: ', realClass,' Predicted: ',prediction
        if prediction==realClass:
            sum = sum+1
    print 'Number of correctly classified: ',sum,'Total number of test instances: ',len(testData)


#------------------------------Code for plotting various graphs based on data size-accuracy and tree size-accuracy-------------------------

    #newDataSetSizeList = [int(.05*len(trainingData)),int(.1*len(trainingData)),int(.2*len(trainingData)),int(.5*len(trainingData)),len(trainingData)]
    #plotForSubset(newDataSetSizeList,trainingData,testData)
    '''
    mvalues = [2,5,10,20]
    accuracyList=[]
    for m in mvalues:
        t = Tree()
        t.createTree(trainingData, t.head, 0, m)
        sum = 0
        for d in testData:
            prediction = predict_class(t.head,d)
            realClass = d[-1]
            #print i,': Actual: ', realClass,' Predicted: ',prediction
            if prediction==realClass:
                sum = sum+1
        accuracy = sum/float(len(testData))
        accuracyList.append(accuracy)
    plt.plot(mvalues,accuracyList,'r')
    plt.axis([0, 25, 0, 1])
    plt.xlabel('Tree Size')
    plt.ylabel('Accuracy')
    plt.show()
    '''
def plotForSubset(newDataSetSizeList,trainingData,testData):
    xvalues=[]
    ymaxvalues=[]
    yminvalues=[]
    ymeanvalues=[]

    for newDataSetSize in newDataSetSizeList:
        accuracyList =get_accuracyList(trainingData,newDataSetSize,testData)
        max = sorted(accuracyList)[-1]
        min = sorted(accuracyList)[0]
        mean = numpy.mean(accuracyList)
        xvalues.append(newDataSetSize)
        ymaxvalues.append(max)
        yminvalues.append(min)
        ymeanvalues.append(mean)
    plt.plot(xvalues,ymaxvalues,'r')
    plt.plot(xvalues,yminvalues,'b')
    plt.plot(xvalues,ymeanvalues,'g')
    plt.axis([0, len(trainingData), 0, 1])
    plt.xlabel('Data Size')
    plt.ylabel('Accuracy')
    plt.show()

def get_accuracyList(trainingData,newDataSetSize,testData):
    j=0
    accuracyList = []
    while j<10:
        t = Tree()
        t.createTree(get_subset(trainingData,newDataSetSize), t.head, 0, 4)
        i = 1
        sum = 0
        for d in testData:
            prediction = predict_class(t.head,d)
            realClass = d[-1]
            #print i,': Actual: ', realClass,' Predicted: ',prediction
            if prediction==realClass:
                sum = sum+1
        accuracy = sum/float(len(testData))
        accuracyList.append(accuracy)
        j=j+1
    return accuracyList

def get_subset(trainingData,newDatasetSize):
    rand_smpl = []
    rand_smpl_indices = random.sample(xrange(len(trainingData)), newDatasetSize)
    for i in rand_smpl_indices:
        rand_smpl.append(trainingData[i])
    #print rand_smpl
    return rand_smpl

if __name__ == '__main__':
    main()