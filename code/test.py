import numpy as np
import pandas as pd

def remove_carriage_return(filename):
    with open(filename) as file:
        txt = file.read()

    txt = txt.replace(r'\r', '')
    with open(filename+"_fixed", 'w') as file:
        file.write(txt)

def test1():
    act = 'Cook'
    try:
        if np.isnan(act):
            print("NAN")
        else:
            print(act)
    except TypeError:
        print(act)

def test2():
    grade = 'F'
    match grade:
        case 'A':
            print("90-100")
        case 'B':
            print("80-99")
        case 'C':
            print("70-79")
        case _:
            print("Fail")

def onehot_encode_test():
    labels = [0, 0, 4, 2, 3, 1, 5, 2, 2, 0]
    labels = pd.get_dummies(labels).values.astype(int)
    print(labels)

def test3():
    import os
    print(os.listdir())
    #remove_carriage_return('datamanager/hfailure/data_visualization.py')
    met_vals = met_dict = {
    'Bathe': 1.5,
    'Beach': 1.8,  # relaxing
    'Clean': 3.3,
    'Church': 1.8,
    'Church Services': 1.8,
    'Computer Work':1.5,
    'Dress': 2.5,
    'Drive': 2.5,
    'Driving': 2.5,
    'Eat': 1.8,
    'Exercise': 5.0,
    'Meeting': 1.5,
    'Play':3.5,
    'Shop':2.5,
    'Shopping': 2.5,
    'Test': 1.3,  # writing
    'Walk': 3.5,
    'Watch TV':1.0,
    'Work': 1.5,  # office work
    'Sleep':1.0,
    'Read':1.3,
    'Cook':3.3,
    'Garden':3.8,
    'Take Medicine':1.5,
    'Pray ROSARY':1.3,
    'Laundry':2.0,
    'Socialize':1.8,#family reunion retreat
    'Dog Walk':3.0,
    'Travel':2.0,
    'Computer':1.5,
    'Websurf':1.5, #computer work manually added
    'Relax':1.8,
    'Cycle': 8.0,
    }

    all_words = list(met_vals.keys())
    all_words.sort()
    print(len(all_words))
    for word in all_words:
        print(f'{word},{met_vals[word]}')


def main():
    onehot_encode_test()

if __name__=="__main__":
	main()


