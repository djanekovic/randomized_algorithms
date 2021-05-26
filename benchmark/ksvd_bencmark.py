from time import time

import numpy as np
import scipy as sp

from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from matplotlib import pyplot as plt

from randomized_algorithms import RandomizedKSVD, ApproximateKSVD, ClassicKSVD

def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference

    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)

def compute_norm(image, reference):
    return np.linalg.norm(image - reference)


def reconstruct(dico, V, distorted, patch_size):
    t0 = time()
    data = extract_patches_2d(distorted, patch_size)
    data = data.reshape(data.shape[0], -1)
    intercept = np.mean(data, axis=0)
    data -= intercept
    print('done in %.2fs.' % (time() - t0))

    reconstruction = distorted.copy()
    t0 = time()
    code = dico.transform(data)
    patches = np.dot(code, V)

    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    reconstruction = reconstruct_from_patches_2d(patches, distorted.shape)
    dt = time() - t0

    return reconstruction

def get_training_data(patch_size):
    t0 = time()
    data = np.array([])
    for filename in os.listdir('/home/darko/Code/gray_test_images'):
        image = plt.imread(filename)
        data = extract_patches_2d(lena, patch_size)
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        print('done in %.2fs.' % (time() - t0))


def plot_dictionary(V, x_num, y_num, patch_size):
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(V):
        plt.subplot(x_num, y_num, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Learned dictionary', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


if __name__ == "__main__":
    from scipy.misc import face

    np.random.seed(0)
    face = face(gray=True)

    with open('/home/darko/Downloads/lena512.pgm', 'rb') as f:
        lena = np.fromfile(f, dtype=np.uint8, offset=34).reshape((512, 512))


    # Convert from uint8 representation with values between 0 and 255 to
    # a floating point representation with values between 0 and 1.
    face = face / 255.
    lena = lena / 255.

    # downsample for higher speed
    #face = face[::4, ::4] + face[1::4, ::4] + face[::4, 1::4] + face[1::4, 1::4]
    #face /= 4.0
    patch_size = (8, 8)
    height, width = face.shape

    # Distort the right half of the image
    print('Distorting image...')
    distorted_face = face.copy()
    distorted_face += 0.075 * np.random.randn(*face.shape)

    distorted_lena = lena.copy()
    distorted_lena += 0.075 * np.random.randn(*lena.shape)

    print('Baseline norm: ', compute_norm(distorted_face, face))
    print('Baseline lena norm: ', compute_norm(distorted_lena, lena))

    show_with_diff(distorted_face, face, "Baseline distorted face")
    show_with_diff(distorted_lena, lena, "Baseline distorted lena")

    # #############################################################################
    # Learn the dictionary from reference patches

    error_norms = []
    for algo in ApproximateKSVD, RandomizedKSVD, ClassicKSVD:
        #data = get_training_data()
        data = extract_patches_2d(lena, patch_size)
        data = data.reshape(data.shape[0], -1)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)

        print('Learning the dictionary...')
        t0 = time()
        dico = algo(n_components=49, max_iter=25)
        V = dico.fit(data).components_
        dt = time() - t0
        print('done in %.2fs.' % dt)

        plot_dictionary(V, 7, 7, patch_size)

        error_norms.append(dico.error_norm)

        # #############################################################################
        # Extract noisy patches and reconstruct them using the dictionary
        #print('Extracting noisy patches... ')
        #t0 = time()
        #data = extract_patches_2d(distorted, patch_size)
        #data = data.reshape(data.shape[0], -1)
        #intercept = np.mean(data, axis=0)
        #data -= intercept
        #print('done in %.2fs.' % (time() - t0))

        #reconstruction = face.copy()
        #t0 = time()
        #code = dico.transform(data)
        #patches = np.dot(code, V)

        #patches += intercept
        #patches = patches.reshape(len(data), *patch_size)
        #reconstruction[:, width // 2:] = reconstruct_from_patches_2d(patches, (height, width // 2))
        #dt = time() - t0
        #print('done in %.2fs.' % dt)
        #print('Reconstruction norm: ', compute_norm(reconstruction, face))
        #show_with_diff(reconstruction, face, algo.__class__.__name__)

        print('Extracting noisy patches... ')
        reconstruction_face = reconstruct(dico, V, distorted_face, patch_size)
        reconstruction_lena = reconstruct(dico, V, distorted_lena, patch_size)
        print('Reconstruction norm: ', compute_norm(reconstruction_face, face))
        print('Reconstruction norm lena: ', compute_norm(reconstruction_lena, lena))
        show_with_diff(reconstruction_face, face, algo.__class__.__name__)
        show_with_diff(reconstruction_lena, lena, algo.__class__.__name__)

        plt.show()
