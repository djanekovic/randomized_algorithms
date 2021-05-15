from time import time

import numpy as np
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

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

if __name__ == "__main__":
    np.random.seed(0)
    from scipy.misc import face
    face = face(gray=True)

    # Convert from uint8 representation with values between 0 and 255 to
    # a floating point representation with values between 0 and 1.
    face = face / 255.

    # downsample for higher speed
    face = face[::4, ::4] + face[1::4, ::4] + face[::4, 1::4] + face[1::4, 1::4]
    face /= 4.0
    height, width = face.shape

    # Distort the right half of the image
    print('Distorting image...')
    distorted = face.copy()
    distorted[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)

    # Extract all reference patches from the left half of the image
    print('Extracting reference patches...')
    t0 = time()
    patch_size = (7, 7)
    data = extract_patches_2d(distorted[:, :width // 2], patch_size)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    print('done in %.2fs.' % (time() - t0))
    print('Baseline norm: ', compute_norm(distorted, face))

    # #############################################################################
    # Learn the dictionary from reference patches

    error_norms = []
    for algo in ApproximateKSVD, RandomizedKSVD, ClassicKSVD:
        print('Learning the dictionary...')
        t0 = time()
        dico = algo(n_components=100, max_iter=20)
        V = dico.fit(data).components_
        dt = time() - t0
        print('done in %.2fs.' % dt)

        error_norms.append(dico.error_norm)

        # #############################################################################
        # Extract noisy patches and reconstruct them using the dictionary
        print('Extracting noisy patches... ')
        t0 = time()
        data = extract_patches_2d(distorted[:, width // 2:], patch_size)
        data = data.reshape(data.shape[0], -1)
        intercept = np.mean(data, axis=0)
        data -= intercept
        print('done in %.2fs.' % (time() - t0))

        reconstruction = face.copy()
        t0 = time()
        code = dico.transform(data)
        patches = np.dot(code, V)

        patches += intercept
        patches = patches.reshape(len(data), *patch_size)
        reconstruction[:, width // 2:] = reconstruct_from_patches_2d(patches, (height, width // 2))
        dt = time() - t0
        print('done in %.2fs.' % dt)
        print('Reconstruction norm: ', compute_norm(reconstruction, face))


