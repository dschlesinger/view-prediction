import pandas as pd, numpy as np
import os, sys, math, json, shutil

from utils.colors import Colors
from utils.featurizer import featurizer
from utils.settings import Settings

from typing import Tuple, List, Dict, Union, Literal, Callable

import sklearn
import kagglehub
import pydicom
from PIL import Image
import tarfile

from IPython.display import clear_output

class image_processing():

    def __init__(self, unzip_path_inbreast: str = 'INBreast', unzip_path_cbis_ddsm: str = 'CBIS-DDSM') -> None:

        # load id to file for inbreast if exists

        if os.path.exists(unzip_path_inbreast):
            self.id_to_file_inbreast = {f.split('_')[0]: f'{unzip_path_inbreast}/ALL-IMGS/' + f for f in os.listdir(f'{unzip_path_inbreast}/ALL-IMGS') if '.dcm' in f}
        else:
            print(f"{Colors.YELLOW.value}INBreast path not found, did not create id to file key!{Colors.YELLOW.value}")

    def get_inbreast_image(self, filename: str, unzip_path: str = 'INBreast', folder: str = 'ALL-IMGS', compress: bool = False, to_rgb: bool = False, full_filename: bool = False, new_width: int = 224, new_height: int = 224) -> np.array:

        image_path: str = self.id_to_file[filename.__str__()] if not full_filename else f'{unzip_path}/{folder}/' + filename.__str__()

        image = pydicom.dcmread(image_path).pixel_array

        if compress:

            image = image_processing.compress_image_average(image, new_width = new_width, new_height = new_height)
        
        if to_rgb:

            image = image_processing.heatmap_to_rgb(image, scale_to_256=True)
        
        return image
    
    @staticmethod
    def get_cbis_ddsm_image(filename: str, unzip_path: str = 'CBIS-DDSM', folder: str = 'jpeg', compress: bool = False, to_rgb: bool = False, new_width: int = 224, new_height: int = 224) -> np.array:

        image = np.array(Image.open(f'{unzip_path}/{folder}/{filename}'))

        if compress:

            image =  image_processing.compress_image_average(image, new_width = new_width, new_height = new_height)
        
        if to_rgb:

            image = image_processing.heatmap_to_rgb(image, scale_to_256=True)
        
        return image

    @staticmethod
    def compress_image_average(image: np.array, new_width: int = 224, new_height: int = 224) -> np.array:
        
        height = image.__len__()

        width = image[0].__len__()

        if new_height > height or new_width > width:

            raise ArithmeticError(f"{Colors.RED.value}New image with shape ({new_width}, {new_height}) exceeds bounds of old image ({width}, {height}){Colors.RESET.value}")

        kernel_height = height // new_height

        kernel_width = width // new_width

        overhang_height = height % new_height

        overhang_width = width % new_width

        # Generate buffer to add over hang

        buffer_height = np.array([1] * overhang_height + [0] * (new_height - overhang_height))

        np.random.shuffle(buffer_height)

        buffer_width = np.array([1] * overhang_width + [0] * (new_width - overhang_width))

        np.random.shuffle(buffer_width)

        new_image = []

        # Create height, width counters

        counter_height = 0

        for i in range(new_height): # rows

            new_image.append([])

            end_h = counter_height + kernel_height + buffer_height[i]

            counter_width = 0

            for j in range(new_width): # columns

                end_w = counter_width + kernel_width + buffer_width[j]

                area = image[counter_height: end_h, counter_width: end_w]

                new_image[-1].append(int(area.mean()))

                counter_width += kernel_width + buffer_width[j]

            counter_height += kernel_height + buffer_height[i]

        return np.array(new_image)
    
    def inbreast_id_to_file(self, unzip_path: str = 'INBreast') -> Dict:

        self.id_to_file_inbreast = {f.split('_')[0]: f'{unzip_path}/ALL-IMGS/' + f for f in os.listdir(f'{unzip_path}/ALL-IMGS') if '.dcm' in f}

        return self.id_to_file
    
    @staticmethod
    def heatmap_to_rgb(image, scale_to_256: bool = True) -> np.array:

        if scale_to_256:

            max_value: float = np.max(image)

            if max_value > 0:

                image = (image.astype(np.float32) / max_value.astype(np.float32)) * 255

        # Stack to 3 layers

        stacked_image = np.repeat(np.expand_dims(image, axis=-1), repeats=3, axis=2)

        return stacked_image.astype(np.uint8)

class inbreast():

    # Disk size in bytes
    disk_size: int = None

    # Number of Examples + Number usable ('CC' || 'MLO') && image
    num_examples: int = 410
    valid_examples: int = 409

    # Main csv/metadata
    metadata_link: str = 'INbreast.xls'

    @staticmethod
    def download(unzip_path: str = Settings.INBreast.unzip_path) -> None:
        """
        Downloads INBreast dataset from kagglehub into folder system

        Approximate run time: 1-2 min
        """

        # Checks if dataset exists

        if os.path.exists(unzip_path):

            print(f"{Colors.GREEN.value}Path already exists for INBreast!{Colors.RESET.value}")

            return

        # Download latest version

        print(f"{Colors.CYAN.value}Path not found, downloading!{Colors.RESET.value}")

        zip_path = kagglehub.dataset_download("martholi/inbreast") + '/inbreast.tgz'

        # Extract zip

        print(f"{Colors.MAGENTA.value}Extracting!{Colors.RESET.value}")

        with tarfile.open(zip_path) as d:
            d.extractall(path=unzip_path)

        print(f"{Colors.GREEN.value}Done!{Colors.RESET.value}")

        return

    @staticmethod
    def load_dataframe(unzip_path: str = Settings.INBreast.unzip_path, drop_extra_views: bool = True) -> pd.DataFrame:
        """
        Returns a dataframe of the INBreast data
        """

        # Checks if dataset exists
        if not os.path.exists(unzip_path):

            raise FileNotFoundError(f"{Colors.RED.value}Path to INBreast not found{Colors.RESET.value}")
        
        df = pd.read_excel(f'{unzip_path}/{inbreast.metadata_link}', skipfooter=2)

        if drop_extra_views: df = df[(df["View"] == "CC") | (df["View"] == "MLO")]

        return df

    @staticmethod
    def image_batch(num_batches: int = 8, compress: bool = False, new_width: int = Settings.General.def_compress_width, new_height: int = Settings.General.def_compress_width, df = None) -> List[Tuple[Dict, np.array]]:
        """
        Generator to load images in batches, conservers memory
        Note: Final batch takes overhang

        Inputs:
            num_batches (int): INBreast ~8 GB, 8 batches works well
            compress (bool): Compresses loaded images, through averaging
            new_height (int): new height of compressed image
            new_width (int): new width of compressed image
            df (pd.Dataframe): used to get total num examples, loads df if not provided

        Returns:
            images (List[np.array]): 409 / num_batches per batch
        """

        if df is None:

            print(f"{Colors.YELLOW.value}No Dataframe provided, loading!{Colors.RESET.value}")

            clear_output(wait=True)

            df =  inbreast.load_dataframe(drop_extra_views = True)

        len_df: int = df.__len__()

        batch_size: int = math.floor(len_df / num_batches)

        image_tools = image_processing()

        for b in range(num_batches):

            indexs = range(b * batch_size, (b+1) * batch_size if b+1 < num_batches else len_df)

            data = [df.iloc[i] for i in indexs]
            batch = [image_tools.get_inbreast_image(d['File Name'].__str__()) for d in data]

            yield data, batch

    @staticmethod
    def get_features(num_bacthes: int = 8, from_compressed = False, savefile: str = 'inbreast.features.json') -> Dict:
        """
        Calculate features from mammograms
        Average run time: 1-2 min
        """

        if os.path.exists(savefile):

            with open(savefile) as ibf:

                return json.load(ibf)
        
        else:

            batch_loader = inbreast.image_batch(num_batches=num_bacthes, compress=from_compressed)

            features = {}

            for i, d in enumerate(batch_loader):

                print(f"{i + 1}/{num_bacthes}")

                clear_output(wait=True)

                data, batch = d

                pred_left, pred_right = featurizer.getLaterality_parallel(batch)

                pred_views, pred_views_poly =  featurizer.getView_parallel(batch)

                for d, l, r, v, pv in  zip(data, pred_left, pred_right, pred_views, pred_views_poly):

                    features[d['File Name'].__str__()] = {
                        'pred_lat': [float(l), float(r)],
                        'pred_view': [float(v_) for v_ in v],
                        'pred_view_poly': list([float(pv_) for pv_ in pv]),
                        'true_lat': d['Laterality'],
                        'true_view': d['View']
                    }

                with open(savefile, 'w') as ibf:

                    json.dump(features, ibf)

            return features
        
    @staticmethod
    def get_xy(test_split: float = 0.2, validation_split: float = None, random_state: int = 42) -> Tuple[np.array]:
        """
        Returns xt, xtest, xval, yt, ytest, yval if validation_split else no xval, yval
        """

        features = inbreast.get_features()

        x = [d['pred_lat'] + d['pred_view'] + d['pred_view_poly'] for d in features.values()]

        y = [1 if d['true_view'] == 'MLO' else 0 for d in features.values()]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_split, random_state=random_state)

        # Calculate % of train split to take as validation, don't if 0 or None
        if validation_split:
            train_split = (1 - test_split)

            # To normalize for taking from train split
            validation_split = validation_split / train_split

            x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train, y_train, test_size=validation_split, random_state=random_state)

            return list(map(lambda x: np.array(x), [x_train, x_test, x_val, y_train, y_test, y_val]))
        
        return list(map(lambda x: np.array(x), [x_train, x_test, y_train, y_test]))
    
    @staticmethod
    def delete(unzip_path: str = Settings.INBreast.unzip_path) -> None:

        if not os.path.exists(unzip_path):
            print(f'{Colors.YELLOW.value}INBreast data not found!{Colors.RESET.value}')

            return
        
        if input('Are you sure (y/n)?: ').lower() in ['yes', 'y']:

            shutil.rmtree(unzip_path)

        print(f'{Colors.GREEN.value}Sucsessfully removed INBreast!{Colors.RESET.value}')
        
class cbis_ddsm():

    # Disk size in bytes
    disk_size: int = 6,319,366,144

    # Number of Examples + Number usable ('CC' || 'MLO') && image
    num_examples: int = 10,239
    valid_examples: int = 2,857

    # Main csv/metadata
    metadata_link: str = 'csv/dicom_info.csv'

    @staticmethod
    def download(unzip_path: str = 'CBIS-DDSM') -> None:
        """
        Downloads CBIS-DDSM dataset from kagglehub into folder system

        Approximate run time: 
        """

        # Checks if dataset exists

        if os.path.exists(unzip_path):

            print(f"{Colors.GREEN.value}Path already exists for CBIS-DDSM!{Colors.RESET.value}")

            return

        # Download latest version

        print(f"{Colors.CYAN.value}Path not found, downloading!{Colors.RESET.value}")

        zip_path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")

        # Move folder into this repo

        print(f"{Colors.MAGENTA.value}Moving!{Colors.RESET.value}")

        shutil.move(zip_path, os.getcwd())

        # rename to cbis_ddsm

        os.rename(os.getcwd() + '/1', os.getcwd() + f'/{unzip_path}')

        print(f"{Colors.GREEN.value}Done!{Colors.RESET.value}")

        return
    
    @staticmethod
    def load_dataframe(unzip_path: str = 'CBIS-DDSM', drop_extra_views: bool = True) -> pd.DataFrame:
        """
        Returns a dataframe of the CBIS-DDSM data

        No extra views in CBIS-DDSM so drop_extra_views redundant
        """

         # Checks if dataset exists
        if not os.path.exists(unzip_path):

            raise FileNotFoundError(f"{Colors.RED.value}Path to CBIS-DDSM not found{Colors.RESET.value}")

        df = pd.read_csv(f"{unzip_path}/{cbis_ddsm.metadata_link}")

        # Only use full mammograms
        df =  df[df['SeriesDescription'] == 'full mammogram images']

        # Only take image_path, Laterality, and PatientOrientation (View)

        new_df = df[['image_path', 'Laterality', 'PatientOrientation']].copy()
        
        new_df.rename(columns={'PatientOrientation': 'View', 'image_path': 'ImageLink'}, inplace=True)

        new_df['ImageLink'] = new_df['ImageLink'].apply(lambda l: '/'.join(l.split('/')[2:4]))

        # resets range to 1...2,857
        new_df.reset_index(drop=True, inplace=True)
        
        return new_df

    
    @staticmethod
    def image_batch(num_batches: int = 8, compress: bool = False, new_width: int = 224, new_height: int = 224, df = None) -> List[Tuple[Dict, np.array]]:
        """
        Generator to load images in batches, conservers memory
        Note: Final batch takes overhang

        Inputs:
            num_batches (int): CBIS-DDSM ~6 GB, 8 batches works well
            compress (bool): Compresses loaded images, through averaging
            new_height (int): new height of compressed image
            new_width (int): new width of compressed image
            df (pd.Dataframe): used to get total num examples, loads df if not provided

        Returns:
            images (List[np.array]): 3568 / num_batches per batch
        """

        if df is None:

            print(f"{Colors.YELLOW.value}No Dataframe provided, loading!{Colors.RESET.value}")

            clear_output(wait=True)

            df =  cbis_ddsm.load_dataframe()

        len_df: int = df.__len__()

        batch_size: int = math.floor(len_df / num_batches)

        image_tools = image_processing()

        for b in range(num_batches):

            indexs = range(b * batch_size, (b+1) * batch_size if b+1 < num_batches else len_df)

            data = [df.iloc[i] for i in indexs]
            batch = [image_tools.get_cbis_ddsm_image(d['ImageLink']) for d in data]

            yield data, batch

    @staticmethod
    def get_features(num_bacthes: int = 8, from_compressed = False, savefile: str = 'cbis_ddsm.features.json') -> Dict:
        """
        Calculate features from mammograms
        Average run time: 1-2 min
        """

        if os.path.exists(savefile):

            with open(savefile) as ibf:

                return json.load(ibf)
        
        else:

            batch_loader = cbis_ddsm.image_batch(num_batches=num_bacthes, compress=from_compressed)

            features = {}

            for i, d in enumerate(batch_loader):

                print(f"{i + 1}/{num_bacthes}")

                clear_output(wait=True)

                data, batch = d

                pred_left, pred_right = featurizer.getLaterality_parallel(batch)

                pred_views, pred_views_poly =  featurizer.getView_parallel(batch)

                for d, l, r, v, pv in  zip(data, pred_left, pred_right, pred_views, pred_views_poly):

                    features[d['ImageLink'].__str__()] = {
                        'pred_lat': [float(l), float(r)],
                        'pred_view': [float(v_) for v_ in v],
                        'pred_view_poly': list([float(pv_) for pv_ in pv]),
                        'true_lat': d['Laterality'],
                        'true_view': d['View']
                    }

                with open(savefile, 'w') as ibf:

                    json.dump(features, ibf)

            return features

    @staticmethod
    def get_xy(test_split: float = 0.2, validation_split: float = None, random_state: int = 42) -> Tuple[np.array]:
        """
        Returns xt, xtest, xval, yt, ytest, yval if validation_split else no xval, yval
        """

        features = cbis_ddsm.get_features()

        x = [d['pred_lat'] + d['pred_view'] + d['pred_view_poly'] for d in features.values()]

        y = [1 if d['true_view'] == 'MLO' else 0 for d in features.values()]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_split, random_state=random_state)

        # Calculate % of train split to take as validation, don't if 0 or None
        if validation_split:
            train_split = (1 - test_split)

            # To normalize for taking from train split
            validation_split = validation_split / train_split

            x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train, y_train, test_size=validation_split, random_state=random_state)

            return list(map(lambda x: np.array(x), [x_train, x_test, x_val, y_train, y_test, y_val]))
        
        return list(map(lambda x: np.array(x), [x_train, x_test, y_train, y_test]))
    
    @staticmethod
    def get_images(compress: bool = True, new_width: int = 224, new_height: int = 224, df = None, test_split: float = 0.2, validation_split: float = None, random_state: int = 42) -> Tuple[np.array]:
        """
        Loads images and logits for image based analysis

        Warning, setting compress to False could lead to fry RAM, please use with caution

        Average time: 10-15 min
        """

        if not compress: print(f"{Colors.RED.value}Warning compress = False, this will take about {round(cbis_ddsm.disk_size / (1024 ** 3), 2)} GB of RAM!{Colors.RED.value}")

        if df is None:

            print(f"{Colors.YELLOW.value}No Dataframe provided, loading!{Colors.RESET.value}")

            clear_output(wait=True)

            df =  cbis_ddsm.load_dataframe()

        x = []

        y = []

        total_images: int = df.__len__()

        for i, image in df.iterrows():

            print(f"{i + 1}/{total_images}")

            clear_output(wait=True)

            x.append(image_processing.get_cbis_ddsm_image(image['ImageLink'], compress=compress, new_height=new_height, new_width=new_width, to_rgb=True))

            y.append(1 if image['View'] == 'MLO' else 0)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_split, random_state=random_state)

        # Calculate % of train split to take as validation, don't if 0 or None
        if validation_split:
            train_split = (1 - test_split)

            # To normalize for taking from train split
            validation_split = validation_split / train_split

            x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train, y_train, test_size=validation_split, random_state=random_state)

            return list(map(lambda x: np.array(x), [x_train, x_test, x_val, y_train, y_test, y_val]))
        
        return list(map(lambda x: np.array(x), [x_train, x_test, y_train, y_test]))
    
    @staticmethod
    def delete(unzip_path: str = Settings.CBIS_DDSM.unzip_path) -> None:

        if not os.path.exists(unzip_path):
            print(f'{Colors.YELLOW.value}CBIS-DDSM data not found!{Colors.RESET.value}')

            return
        
        if input('Are you sure (y/n)?: ').lower() in ['yes', 'y']:

            shutil.rmtree(unzip_path)

        print(f'{Colors.GREEN.value}Sucsessfully removed CBIS-DDSM!{Colors.RESET.value}')

