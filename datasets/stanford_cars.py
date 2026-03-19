import os
import pickle
from scipy.io import loadmat

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class StanfordCars(DatasetBase):

    dataset_dir = "stanford_cars"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            import sys
            import importlib
            from tqdm import tqdm
            
            # Temporary workaround to bypass local "datasets" folder
            if '' in sys.path: sys.path.remove('')
            if os.getcwd() in sys.path: sys.path.remove(os.getcwd())
            
            hf_datasets = importlib.import_module("datasets")
            load_dataset = hf_datasets.load_dataset
            
            print("Loading HuggingFace tanganke/stanford_cars...")
            hf_ds = load_dataset("tanganke/stanford_cars")
            
            image_dir = os.path.join(self.dataset_dir, "images")
            mkdir_if_missing(image_dir)
            
            def process_split(split_name, hf_split):
                items = []
                features = hf_split.features
                class_names = features['label'].names
                
                print(f"Processing and saving images for {split_name} split...")
                for i, item in enumerate(tqdm(hf_split)):
                    img = item['image']
                    label = item['label']
                    classname = class_names[label]
                    
                    names = classname.split(" ")
                    if names[-1].isdigit() and len(names[-1]) == 4:
                        year = names.pop(-1)
                        names.insert(0, year)
                        classname = " ".join(names)
                    
                    imname = f"{split_name}_{i:06d}.jpg"
                    impath = os.path.join(image_dir, imname)
                    
                    if not os.path.exists(impath):
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(impath)
                    
                    items.append(Datum(impath=impath, label=label, classname=classname))
                return items
            
            trainval = process_split("train", hf_ds["train"])
            test = process_split("test", hf_ds["test"])
            
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

