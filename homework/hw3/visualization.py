from matplotlib import pyplot as plt

# recommended color for different digits
color_mapping = {0:'red',1:'green',2:'blue',3:'yellow',4:'magenta',5:'orangered',
                6:'cyan',7:'purple',8:'gold',9:'pink'}

def plot2d(data,label,split='train'):
    # 2d scatter plot of the hidden features
    fig = plt.figure()
    plt.scatter(data[:,0],data[:,1],c=[color_mapping[cur] for cur in label])
    fig.savefig('hidden_2d_'+split+'.png')
    plt.show()

def plot3d(data,label,split='train'):
    # 3d scatter plot of the hidden features
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2],c=[color_mapping[cur] for cur in label])
    fig.savefig('hidden_3d_'+split+'.png')
    plt.show()
