from matplotlib import pyplot as plt
def mysubplot(imgs, titles, row, col):
    '''
        Subplots a set of images in grayscale
    '''
    for i in range(row*col):
        plt.subplot(row,col,i+1),plt.imshow(imgs[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.tight_layout()
    plt.show()        
