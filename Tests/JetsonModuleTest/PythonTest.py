import numpy as np

def myFunc(count1,count2,count3, count4, count5):
    count = np.array([count1,count2, count3, count4, count5],dtype=np.uint8)
    counter = np.max(count) - np.min(count)

    for i in range(counter):
        print(f"{i+1}: Hello World")

if __name__ == "__main__":
    myFunc(5,10, 3, 7, 13)