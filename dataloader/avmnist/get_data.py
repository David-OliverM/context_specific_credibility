"""Implements dataloaders for the AVMNIST dataset.

Here, the data is assumed to be in a folder titled "avmnist".
"""
import numpy as np
from torch.utils.data import DataLoader



def get_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True, flatten_audio=False, flatten_image=False, unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):
    """Get dataloaders for AVMNIST.

    Args:
        data_dir (str): Directory of data.
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        flatten_audio (bool, optional): Whether to flatten audio data or not. Defaults to False.
        flatten_image (bool, optional): Whether to flatten image data or not. Defaults to False.
        unsqueeze_channel (bool, optional): Whether to unsqueeze any channels or not. Defaults to True.
        generate_sample (bool, optional): Whether to generate a sample and save it to file or not. Defaults to False.
        normalize_image (bool, optional): Whether to normalize the images before returning. Defaults to True.
        normalize_audio (bool, optional): Whether to normalize the audio before returning. Defaults to True.

    Returns:
        tuple: Tuple of (training dataloader, validation dataloader, test dataloader)
    """
    swap = 0.5
    #indicator array of shape [1, 128] with all values as 0
    noisy_ind = np.zeros((1, 128), dtype=np.float32)
    non_noisy_ind = np.ones((1, 128), dtype=np.float32)
    print(f"Loading the corrupted AVMNIST dataset...Swapped {swap} samples")
    trains = [np.load(data_dir+"/image/train_data.npy"), np.load(data_dir +
                                                                 "/audio/train_data.npy"), np.load(data_dir+"/train_labels.npy")]
    tests = [np.load(data_dir+"/image/test_data.npy"), np.load(data_dir +
                                                               "/audio/test_data.npy"), np.load(data_dir+"/test_labels.npy")]
    # Define the classes to swap
    class_a = 5  
    class_b = 9  


    #Adding the noise in image modality via swapping data
    labels_train = np.load(data_dir + "/train_labels.npy")
    labels_test = np.load(data_dir + "/test_labels.npy")

    

    # Get indices of the samples corresponding to each class
    train_indices_a = np.where(labels_train == class_a)[0]
    train_indices_b = np.where(labels_train == class_b)[0]

    test_indices_a = np.where(labels_test == class_a)[0]
    test_indices_b = np.where(labels_test == class_b)[0]


    #Add an indicator if the data is swapped or not
    indicator_image_train = [noisy_ind if idx in train_indices_a or idx in train_indices_b else non_noisy_ind for idx in range(len(trains[2]))]
    indicator_image_test = [noisy_ind if idx in test_indices_a or idx in test_indices_b else non_noisy_ind for idx in range(len(tests[2]))]


    # Decide how many samples to swap from each class
    num_train_to_swap = int(swap*min(len(train_indices_a), len(train_indices_b)))
    num_test_to_swap = int(swap*min(len(test_indices_a), len(test_indices_b)))

    # Select the indices to swap
    swap_indices_train_a = train_indices_a[:num_train_to_swap]
    swap_indices_train_b = train_indices_b[:num_train_to_swap]
    swap_indices_test_a = test_indices_a[:num_test_to_swap]
    swap_indices_test_b = test_indices_b[:num_test_to_swap]

    # Perform the swap for image data
    trains[0][swap_indices_train_a], trains[0][swap_indices_train_b] = trains[0][swap_indices_train_b], trains[0][swap_indices_train_a]
    tests[0][swap_indices_test_a], tests[0][swap_indices_test_b] = tests[0][swap_indices_test_b], tests[0][swap_indices_test_a]


    if flatten_audio:
        trains[1] = trains[1].reshape(60000, 112*112)
        tests[1] = tests[1].reshape(10000, 112*112)
    if generate_sample:
        _saveimg(trains[0][0:100])
        _saveaudio(trains[1][0:9].reshape(9, 112*112))
    if normalize_image:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_audio:
        trains[1] = trains[1]/255.0
        tests[1] = tests[1]/255.0
    if not flatten_image:
        trains[0] = trains[0].reshape(60000, 28, 28)
        tests[0] = tests[0].reshape(10000, 28, 28)
    if unsqueeze_channel:
        trains[0] = np.expand_dims(trains[0], 1)
        tests[0] = np.expand_dims(tests[0], 1)
        trains[1] = np.expand_dims(trains[1], 1)
        tests[1] = np.expand_dims(tests[1], 1)
    trains[2] = trains[2].astype(int)
    tests[2] = tests[2].astype(int)
    trainlist = [[[trains[j][i] for j in range(3)], indicator_image_train[i]] for i in range(60000)] 
    testlist = [[[tests[j][i] for j in range(3)], indicator_image_test[i]] for i in range(10000)]
    valids = DataLoader(trainlist[55000:60000], shuffle=False,
                        num_workers=num_workers, batch_size=batch_size)
    tests = DataLoader(testlist, shuffle=False,
                       num_workers=num_workers, batch_size=batch_size)
    trains = DataLoader(trainlist[0:55000], shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size)
    return trains, valids, tests

# this function creates an image of 100 numbers in avmnist


def _saveimg(outa):
    from PIL import Image
    t = np.zeros((300, 300))
    for i in range(0, 100):
        for j in range(0, 784):
            imrow = i // 10
            imcol = i % 10
            pixrow = j // 28
            pixcol = j % 28
            t[imrow*30+pixrow][imcol*30+pixcol] = outa[i][j]
    newimage = Image.new('L', (300, 300))  # type, size
    
    newimage.putdata(t.reshape((90000,)))
    newimage.save("samples.png")


def _saveaudio(outa):
    
    from PIL import Image
    t = np.zeros((340, 340))
    for i in range(0, 9):
        for j in range(0, 112*112):
            imrow = i // 3
            imcol = i % 3
            pixrow = j // 112
            pixcol = j % 112
            t[imrow*114+pixrow][imcol*114+pixcol] = outa[i][j]
    newimage = Image.new('L', (340, 340))  # type, size
    
    newimage.putdata(t.reshape((340*340,)))
    newimage.save("samples2.png")


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader = get_dataloader("/home/pxt220000/Projects/datasets/avmnist")
    for batch in train_loader:
        (images, audios, labels), indicators = batch
        print(f"Images shape: {images.shape}, Audios shape: {audios.shape}, Labels shape: {labels.shape}, Indicators shape: {indicators.shape}")
        break