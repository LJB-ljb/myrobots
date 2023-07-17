import csv
import numpy as np
import matplotlib.pyplot as plt



def read_data(csvfile):
    """read datas from file using csv format"""
    data = []
    results = []
    with open(csvfile) as f:
        csv_reader = csv.reader(f)
        [data.append(row) for row in csv_reader]

    variable_name = data[0]
    data.remove(data[0])
    for d in data:
        for i in range(0,5):
            if i == 4:
                if d[i] == "setosa":
                    results.append(0)
                elif d[i] == "versicolor":
                    results.append(1)
                elif d[i] == "virginica":
                    results.append(2) 
            else:
                d[i] = float(d[i])
        d.remove(d[4])         
    return np.array(data), results, variable_name


def plot(training_data, training_results, labels, variable_name):
    """plot data"""    
    
    # plot data using first two lines
    for i in range(len(training_results)):
        data = training_data[i]
        if training_results[i] == 0:
            type1 = plt.scatter(data[0], data[1], label=labels[0], c='b')
        
        elif training_results[i] == 1:
            type2 = plt.scatter(data[0], data[1], label=labels[1], c='r')
        
        elif training_results[i] == 2:
            type3 = plt.scatter(data[0], data[1], label=labels[2], c='g')
    plt.xlabel(variable_name[0])
    plt.ylabel(variable_name[1])
    plt.legend((type1, type2, type3), labels, loc = 0)
    plt.savefig("fig1.png")
    

    # plot data using first line and third line
    for i in range(len(training_results)):
        data = training_data[i]
        if training_results[i] == 0:
            type1 = plt.scatter(data[0], data[2], label=labels[0], c='b')
        
        elif training_results[i] == 1:
            type2 = plt.scatter(data[0], data[2], label=labels[1], c='r')
        
        elif training_results[i] == 2:
            type3 = plt.scatter(data[0], data[2], label=labels[2], c='g')
    plt.xlabel(variable_name[0])
    plt.ylabel(variable_name[1])
    plt.legend((type1, type2, type3), labels, loc = 0)
    plt.savefig("fig2.png")
    

    # plot data using first line and fourth line
    for i in range(len(training_results)):
        data = training_data[i]
        if training_results[i] == 0:
            type1 = plt.scatter(data[0], data[3], label=labels[0], c='b')
        
        elif training_results[i] == 1:
            type2 = plt.scatter(data[0], data[3], label=labels[1], c='r')
        
        elif training_results[i] == 2:
            type3 = plt.scatter(data[0], data[3], label=labels[2], c='g')
    plt.xlabel(variable_name[0])
    plt.ylabel(variable_name[1])
    plt.legend((type1, type2, type3), labels, loc = 0)
    plt.savefig("fig3.png")
    


def perception_training(data, training_results, labels):
    """training process for perception"""
    class_number = len(labels)
    row_len, conlumn_len = data.shape
    lr = 0.01 # learning rate
    w = np.zeros((class_number, conlumn_len))   #   weight matrix's row lenth equals to class numbers, column lenth equals to variable numbers

    # perception process
    
    while True:
        Measure = 0

        for i in range(row_len):
            i_class = training_results[i]
            w_i = np.zeros(class_number)    #   weight of the row i
            
            for j in range(class_number):
                # caculate weight
                w_i[j] = w[j].dot(data[i].T)
            

            for j in range(class_number):
                if j != i_class and w_i[i_class] <= w_i[j]:
                    Measure = 1

                    # update weight matrix
                    for j in range(class_number):
                        if j == i_class:
                            w[j] = w[j] + lr * data[i]
                        else:
                            w[j] = w[j] - lr * data[i]
            
            
        if Measure == 0:
            break

    return w

    
def predict(testing_data, test_results, weight, labels):
    """predict process for perception"""
    correct_number = 0
    for i in range(testing_data.shape[0]):
        results = []    
        [results.append(testing_data[i].dot(weight[j])) for j in range(weight.shape[0])]
        print("predict result is:" + labels[results.index(max(results))] + "\treal result is:" + labels[test_results[i]])
        if results.index(max(results)) == test_results[i]: correct_number = correct_number + 1

    accuracy_rate = correct_number/len(test_results)
    print("The accuracy rate is {}.".format(accuracy_rate))
    return accuracy_rate

def main():
    # load data
    training_data, training_results, variable_name = read_data("/home/ljb/Machine_Learning/D01-perceptron/training_data.csv")
    testing_data, test_results, _ = read_data("/home/ljb/Machine_Learning/D01-perceptron/testing_data.csv")

    # define labels
    labels = ["setosa", "versicolor", "virginica"]
    plot(training_data, training_results, labels, variable_name)

    weight = perception_training(training_data, training_results, labels)
    predict(testing_data, test_results, weight, labels)

    

main()

