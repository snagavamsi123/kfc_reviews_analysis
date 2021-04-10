import pandas as pd

def label_data():
    rows=pd.read_csv('C:/Users/Ganesh vamsi/Desktop/ninc/kfc/owndataset2 - owndataset.csv',header=0,index_col=False,delimiter=',')
    labels=[]
    for cell in rows['stars']:
        if cell>=4:
            labels.append('2')
        elif cell==3:
            labels.append('1')
        else:
            labels.append('0')
    rows['new_labels']=labels
    return rows