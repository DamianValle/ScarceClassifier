import matplotlib.pyplot as plt
import numpy as np

def show_img(img):
    img = img / 5 + 0.47     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_grid(imgs):
    
    plt.figure()
    plt.axis('off')

    f, axarr = plt.subplots(1,5)
    
    for idx, img in enumerate(imgs):
        img = img / 5 + 0.47
        img = np.transpose(img.numpy(), (1, 2, 0))
        axarr[idx].axis('off')
        axarr[idx].imshow(img)
    
    plt.show()

