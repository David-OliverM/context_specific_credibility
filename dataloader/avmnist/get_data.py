"""Implements dataloaders for the AVMNIST dataset.

Here, the data is assumed to be in a folder titled "avmnist".
"""
import numpy as np
from torch.utils.data import DataLoader
import torch


def get_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True, flatten_audio=False, flatten_image=False, unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True, exp_setup="", noise_severity=None):
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
    # # Define the classes to swap
    class_visual = [5, 8, 1] 
    class_audio = [6, 0, 7]
    if not noise_severity:
        print("Noise Severity for training is set to None - No swapping on the train data")
        swap = 0.0
    else:
        swap = 0.7
        print(f"Noise Severity is not None - Swapping {swap*100}% of the images for classes {class_visual} and {swap*100}% of the audio for classes {class_audio}")
    swap_test = 0.75 # Change this for setting test noise
    print(f"Swapping {swap_test*100}% of the images for classes {class_visual} and {swap_test*100}% of the audio for classes {class_audio} on test")
    #indicator array of shape [1, 128] with all values as 0
    noisy_ind = np.zeros((1, 128), dtype=np.float32)
    non_noisy_ind = np.ones((1, 128), dtype=np.float32)
    # print(f"Loading the corrupted AVMNIST dataset...Swapped {swap} train samples and {swap_test} test samples")
    trains = [np.load(data_dir+"/image/train_data.npy"), np.load(data_dir +
                                                                 "/audio/train_data.npy"), np.load(data_dir+"/train_labels.npy")]
    tests = [np.load(data_dir+"/image/test_data.npy"), np.load(data_dir +
                                                               "/audio/test_data.npy"), np.load(data_dir+"/test_labels.npy")]
    labels_train = np.load(data_dir + "/train_labels.npy")
    labels_test = np.load(data_dir + "/test_labels.npy")
    
    corruption_masks_train = []
    corruption_train = []
    corrupted_modalities_train = []
    corruption_masks_test = []
    corruption_test = []
    corrupted_modalities_test = []
    for i in range(len(trains[0])):
        img_noise_mask = np.zeros_like(trains[0][i])
        audio_noise_mask = np.zeros_like(trains[1][i])
        img_corr = audio_corr = 'none'
        corrupted_img = corrupted_audio = False
        if trains[2][i] in class_visual:
            corrupt_img = np.random.rand()
            if corrupt_img < swap:
                tmp = trains[0][i].copy()
                # corruption = np.random.choice(np.where(np.isin(labels_train, class_audio))[0])
                corruption = np.random.choice(np.where(labels_train == class_audio[class_visual.index(trains[2][i])])[0])
                # trains[0][i][392:] =  trains[0][corruption][392:]
                trains[0][i] = trains[0][corruption]
                img_noise_mask = trains[0][i] - tmp
                img_corr = "swap"
                corrupted_img = True
        
        if trains[2][i] in class_audio:
            corrupt_audio = np.random.rand()
            if corrupt_audio < swap:
                tmp = trains[1][i].copy()
                # corruption = np.random.choice(np.where(np.isin(labels_train, class_visual))[0])
                corruption = np.random.choice(np.where(labels_train == class_visual[class_audio.index(trains[2][i])])[0])
                # trains[1][i][56:, :] = trains[1][corruption][56:, :]
                trains[1][i] = trains[1][corruption]
                audio_noise_mask = trains[1][i] - tmp
                audio_corr = "swap"
                corrupted_audio = True

        corruption_masks_train.append((img_noise_mask, audio_noise_mask))
        corruption_train.append([img_corr, audio_corr])
        corrupted_modalities_train.append([corrupted_img, corrupted_audio])

    
    for i in range(len(tests[0])):
        img_noise_mask = np.zeros_like(tests[0][i])
        audio_noise_mask = np.zeros_like(tests[1][i])
        img_corr = audio_corr = 'none'
        if tests[2][i] in class_visual:
            corrupt_img = np.random.rand()
            if corrupt_img < swap_test:
                tmp = tests[0][i].copy()
                # corruption = np.random.choice(np.where(np.isin(labels_test, class_audio))[0])
                corruption = np.random.choice(np.where(labels_test == class_audio[class_visual.index(tests[2][i])])[0])
                tests[0][i] = tests[0][corruption]
                # tmp = tests[0][i].copy()
                # corruption = np.random.choice(np.where(np.isin(labels_test, class_audio))[0])
                # tests[0][i][392:] =  tests[0][corruption][392:]
                img_noise_mask = tests[0][i] - tmp
                img_corr = "swap"
                corrupted_img = True
        
        if tests[2][i] in class_audio:
            corrupt_audio = np.random.rand()
            if corrupt_audio < swap_test:
                tmp = tests[1][i].copy()
                # corruption = np.random.choice(np.where(np.isin(labels_test, class_visual))[0])
                corruption = np.random.choice(np.where(labels_test == class_visual[class_audio.index(tests[2][i])])[0])
                tests[1][i] = tests[1][corruption]
                # tests[1][i][56:, :] = tests[1][corruption][56:, :]
                audio_noise_mask = tests[1][i] - tmp
                audio_corr = "swap"
                corrupted_audio = True

        corruption_masks_test.append((img_noise_mask, audio_noise_mask))
        corruption_test.append([img_corr, audio_corr])
        corrupted_modalities_test.append([corrupted_img, corrupted_audio])

        


    if flatten_audio:
        trains[1] = trains[1].reshape(60000, 112*112)
        tests[1] = tests[1].reshape(10000, 112*112)
        corruption_masks_train[1] = corruption_masks_train[1].reshape(60000, 112*112)
        corruption_masks_test[1] = corruption_masks_test[1].reshape(10000, 112*112)
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
        corruption_masks_train = [(corruption_masks_train[i][0].reshape(28, 28), corruption_masks_train[i][1]) for i in range(60000)]
        corruption_masks_test = [(corruption_masks_test[i][0].reshape(28, 28), corruption_masks_test[i][1]) for i in range(10000)]
    if unsqueeze_channel:
        trains[0] = np.expand_dims(trains[0], 1)
        tests[0] = np.expand_dims(tests[0], 1)
        trains[1] = np.expand_dims(trains[1], 1)
        tests[1] = np.expand_dims(tests[1], 1)
        corruption_masks_train = [(np.expand_dims(corruption_masks_train[i][0], 0), np.expand_dims(corruption_masks_train[i][1], 0)) for i in range(60000)]
        corruption_masks_test = [(np.expand_dims(corruption_masks_test[i][0], 0), np.expand_dims(corruption_masks_test[i][1], 0)) for i in range(10000)]
    trains[2] = trains[2].astype(int)
    tests[2] = tests[2].astype(int)
    trainlist = [[tuple([trains[j][i] for j in range(3)]), (i, (corruption_train[i][0], corruption_train[i][1]), torch.tensor([corrupted_modalities_train[i][0], corrupted_modalities_train[i][1]])), (corruption_masks_train[i][0], corruption_masks_train[i][1].astype(np.float32))] for i in range(60000)] 
    testlist = [[tuple([tests[j][i] for j in range(3)]), (i, (corruption_test[i][0], corruption_test[i][1]),torch.tensor([corrupted_modalities_test[i][0], corrupted_modalities_test[i][1]])), (corruption_masks_test[i][0], corruption_masks_test[i][1].astype(np.float32))] for i in range(10000)]
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
    train_loader, val_loader, test_loader = get_dataloader("/nas1-nfs1/home/pxt220000/projects/datasets/avmnist", noise_severity=1)
    for batch in train_loader:
        (images, audios, labels), idx, corruptions = batch
        print(f"Images shape: {images.shape}, Audios shape: {audios.shape}, Labels shape: {labels.shape}, Indicators shape: {len(corruptions)}")
        break