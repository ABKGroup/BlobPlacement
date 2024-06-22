## Python code that read all the images in the images folder and generate a gif file
## The gif file is saved in the same folder as the images folder
## The gif file is named as 'output.gif'

import os
import sys
import imageio.v2 as imageio

def gen_gif(dir:str, output_gif_name='output.gif'):
    images = []
    folder = dir
    ## List the files in the folder in the sorted order
    file_names = []
    for filename in os.listdir(folder):
        file_names.append(filename.replace('.png',''))
    file_names.sort()
    
    for filename in file_names:
        filename += '.png'
        images.append(imageio.imread(os.path.join(folder, filename)))
    imageio.mimsave(output_gif_name, images, duration=1.0, loop = 1)

if __name__ == '__main__':
    image_dir=sys.argv[1]
    output_gif=sys.argv[2]
    gen_gif(image_dir, output_gif)
    
