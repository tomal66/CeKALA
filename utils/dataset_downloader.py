import os
import shutil
import urllib.request
import tarfile
import zipfile
import argparse

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading {url} to {target_path}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            print(f"urllib failed with {e}. Falling back to curl...")
            os.system(f'curl -L -o "{target_path}" "{url}"')
    else:
        print(f"File {target_path} already exists. Skipping download.")

def extract_archive(archive_path, extract_dir):
    print(f"Extracting {archive_path} to {extract_dir}...")
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz') or archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(path=extract_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        print(f"Unknown archive format: {archive_path}")

def gdown_download(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading from Google Drive (ID: {file_id}) to {output_path}...")
        try:
            import gdown
            gdown.download(id=file_id, output=output_path, quiet=False)
        except ImportError:
            print("gdown not installed in python environment. Calling CLI...")
            os.system(f"gdown {file_id} -O {output_path}")
    else:
        print(f"File {output_path} already exists. Skipping download.")

def download_caltech101(data_dir):
    out_dir = os.path.join(data_dir, "caltech-101")
    os.makedirs(out_dir, exist_ok=True)
    
    target_obj_dir = os.path.join(out_dir, "101_ObjectCategories")
    if not os.path.exists(target_obj_dir):
        import torchvision
        print("Downloading Caltech101 using torchvision...")
        torchvision.datasets.Caltech101(root=out_dir, download=True)
        # torchvision extracts to data/caltech-101/caltech101/101_ObjectCategories. We move it to expected path:
        extracted_dir = os.path.join(out_dir, "caltech101", "101_ObjectCategories")
        if os.path.exists(extracted_dir):
            shutil.move(extracted_dir, target_obj_dir)
            shutil.rmtree(os.path.join(out_dir, "caltech101"))
    
    split_path = os.path.join(out_dir, "split_zhou_Caltech101.json")
    gdown_download("1hyarUivQE36mY6jSomru6Fjd-JzwcCzN", split_path)

def download_oxford_pets(data_dir):
    out_dir = os.path.join(data_dir, "oxford_pets")
    os.makedirs(out_dir, exist_ok=True)
    
    images_tar = os.path.join(out_dir, "images.tar.gz")
    if not os.path.exists(os.path.join(out_dir, "images")):
        download_file("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", images_tar)
        extract_archive(images_tar, out_dir)
        os.remove(images_tar)
        
    annotations_tar = os.path.join(out_dir, "annotations.tar.gz")
    if not os.path.exists(os.path.join(out_dir, "annotations")):
        download_file("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", annotations_tar)
        extract_archive(annotations_tar, out_dir)
        os.remove(annotations_tar)
        
    split_path = os.path.join(out_dir, "split_zhou_OxfordPets.json")
    gdown_download("1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs", split_path)

def download_stanford_cars(data_dir):
    out_dir = os.path.join(data_dir, "stanford_cars")
    os.makedirs(out_dir, exist_ok=True)
    
    train_tgz = os.path.join(out_dir, "cars_train.tgz")
    if not os.path.exists(os.path.join(out_dir, "cars_train")):
        download_file("http://ai.stanford.edu/~jkrause/car196/cars_train.tgz", train_tgz)
        extract_archive(train_tgz, out_dir)
        os.remove(train_tgz)
        
    test_tgz = os.path.join(out_dir, "cars_test.tgz")
    if not os.path.exists(os.path.join(out_dir, "cars_test")):
        download_file("http://ai.stanford.edu/~jkrause/car196/cars_test.tgz", test_tgz)
        extract_archive(test_tgz, out_dir)
        os.remove(test_tgz)
        
    devkit_tgz = os.path.join(out_dir, "car_devkit.tgz")
    if not os.path.exists(os.path.join(out_dir, "devkit")):
        download_file("https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz", devkit_tgz)
        extract_archive(devkit_tgz, out_dir)
        os.remove(devkit_tgz)
        
    labels_mat = os.path.join(out_dir, "cars_test_annos_withlabels.mat")
    download_file("http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat", labels_mat)
    
    split_path = os.path.join(out_dir, "split_zhou_StanfordCars.json")
    gdown_download("1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT", split_path)

def download_oxford_flowers(data_dir):
    out_dir = os.path.join(data_dir, "oxford_flowers")
    os.makedirs(out_dir, exist_ok=True)
    
    flowers_tgz = os.path.join(out_dir, "102flowers.tgz")
    if not os.path.exists(os.path.join(out_dir, "jpg")):
        download_file("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", flowers_tgz)
        extract_archive(flowers_tgz, out_dir)
        os.remove(flowers_tgz)
        
    labels_mat = os.path.join(out_dir, "imagelabels.mat")
    download_file("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", labels_mat)
    
    cat_json = os.path.join(out_dir, "cat_to_name.json")
    gdown_download("1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0", cat_json)
    
    split_path = os.path.join(out_dir, "split_zhou_OxfordFlowers.json")
    gdown_download("1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT", split_path)

def download_food101(data_dir):
    out_dir = os.path.join(data_dir, "food-101")
    os.makedirs(out_dir, exist_ok=True)
    
    food_tgz = os.path.join(data_dir, "food-101.tar.gz")
    if not os.path.exists(os.path.join(out_dir, "images")):
        download_file("https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/food-101.tar.gz", food_tgz)
        extract_archive(food_tgz, data_dir)
        os.remove(food_tgz)
        
    split_path = os.path.join(out_dir, "split_zhou_Food101.json")
    gdown_download("1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl", split_path)

def download_fgvc_aircraft(data_dir):
    out_dir = os.path.join(data_dir, "fgvc_aircraft")
    
    aircraft_tgz = os.path.join(data_dir, "fgvc-aircraft-2013b.tar.gz")
    if not os.path.exists(out_dir):
        download_file("https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz", aircraft_tgz)
        extract_archive(aircraft_tgz, data_dir)
        
        extracted_dir = os.path.join(data_dir, "fgvc-aircraft-2013b")
        data_subfolder = os.path.join(extracted_dir, "data")
        if os.path.exists(data_subfolder):
            shutil.move(data_subfolder, out_dir)
        shutil.rmtree(extracted_dir)
        os.remove(aircraft_tgz)

def download_sun397(data_dir):
    out_dir = os.path.join(data_dir, "sun397")
    os.makedirs(out_dir, exist_ok=True)
    
    sun_tgz = os.path.join(out_dir, "SUN397.tar.gz")
    if not os.path.exists(os.path.join(out_dir, "SUN397")):
        download_file("http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz", sun_tgz)
        extract_archive(sun_tgz, out_dir)
        os.remove(sun_tgz)
        
    part_zip = os.path.join(out_dir, "Partitions.zip")
    if not os.path.exists(os.path.join(out_dir, "ClassName.txt")): # checking one of the partition files
        download_file("https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip", part_zip)
        extract_archive(part_zip, out_dir)
        os.remove(part_zip)
        
    split_path = os.path.join(out_dir, "split_zhou_SUN397.json")
    gdown_download("1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq", split_path)

def download_dtd(data_dir):
    out_dir = os.path.join(data_dir, "dtd")
    os.makedirs(out_dir, exist_ok=True)
    
    dtd_tgz = os.path.join(data_dir, "dtd-r1.0.1.tar.gz")
    if not os.path.exists(os.path.join(out_dir, "images")):
        download_file("https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz", dtd_tgz)
        extract_archive(dtd_tgz, data_dir) # extracts to dtd automatically
        os.remove(dtd_tgz)
        
    split_path = os.path.join(out_dir, "split_zhou_DescribableTextures.json")
    gdown_download("1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x", split_path)

def download_eurosat(data_dir):
    out_dir = os.path.join(data_dir, "eurosat")
    os.makedirs(out_dir, exist_ok=True)
    
    eurosat_zip = os.path.join(out_dir, "EuroSAT.zip")
    if not os.path.exists(os.path.join(out_dir, "2750")):
        download_file("http://madm.dfki.de/files/sentinel/EuroSAT.zip", eurosat_zip)
        extract_archive(eurosat_zip, out_dir)
        os.remove(eurosat_zip)
        
    split_path = os.path.join(out_dir, "split_zhou_EuroSAT.json")
    gdown_download("1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o", split_path)

def download_ucf101(data_dir):
    out_dir = os.path.join(data_dir, "ucf101")
    os.makedirs(out_dir, exist_ok=True)
    
    ucf_zip = os.path.join(out_dir, "UCF-101-midframes.zip")
    if not os.path.exists(os.path.join(out_dir, "UCF-101-midframes")):
        gdown_download("10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O", ucf_zip)
        if os.path.exists(ucf_zip):
            extract_archive(ucf_zip, out_dir)
            os.remove(ucf_zip)
        
    split_path = os.path.join(out_dir, "split_zhou_UCF101.json")
    gdown_download("1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y", split_path)

def download_imagenet_family(data_dir, variant=None):
    if variant is None or variant == "imagenet":
        out_dir = os.path.join(data_dir, "imagenet")
        os.makedirs(out_dir, exist_ok=True)
        print("="*60)
        print("ImageNet dataset is too large/requires registration to download automatically.")
        print(f"Please manually download the ILSVRC2012 dataset from https://image-net.org/,")
        print(f"and extract it to {os.path.join(out_dir, 'images')} such that it contains 'train' and 'val' subfolders.")
        print("="*60)
        classnames_txt = os.path.join(out_dir, "classnames.txt")
        gdown_download("1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF", classnames_txt)
        
    elif variant == "imagenetv2":
        out_dir = os.path.join(data_dir, "imagenetv2")
        os.makedirs(out_dir, exist_ok=True)
        print("="*60)
        print("ImageNetV2 requires downloading image contents manually if not automatically available.")
        print("You can get matched-frequency dataset from https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz")
        print(f"Extract it to {out_dir}.")
        print("="*60)
        classnames_txt = os.path.join(out_dir, "classnames.txt")
        gdown_download("1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF", classnames_txt)

    elif variant == "imagenet_sketch":
        out_dir = os.path.join(data_dir, "imagenet-sketch")
        os.makedirs(out_dir, exist_ok=True)
        print("="*60)
        print("ImageNet-Sketch dataset: manually download from https://github.com/HaohanWang/ImageNet-Sketch")
        print(f"and extract to {out_dir}")
        print("="*60)
        classnames_txt = os.path.join(out_dir, "classnames.txt")
        gdown_download("1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF", classnames_txt)

    elif variant == "imagenet_a":
        out_dir = os.path.join(data_dir, "imagenet-adversarial")
        os.makedirs(out_dir, exist_ok=True)
        print("="*60)
        print("ImageNet-A dataset: manually download from https://github.com/hendrycks/natural-adv-examples")
        print(f"and extract to {out_dir}")
        print("="*60)
        classnames_txt = os.path.join(out_dir, "classnames.txt")
        gdown_download("1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF", classnames_txt)

    elif variant == "imagenet_r":
        out_dir = os.path.join(data_dir, "imagenet-rendition")
        os.makedirs(out_dir, exist_ok=True)
        print("="*60)
        print("ImageNet-R dataset: manually download from https://github.com/hendrycks/imagenet-r")
        print(f"and extract to {out_dir}")
        print("="*60)
        classnames_txt = os.path.join(out_dir, "classnames.txt")
        gdown_download("1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF", classnames_txt)

def download_dataset(dataset_name, data_dir="data/"):
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == "caltech101":
        download_caltech101(data_dir)
    elif dataset_name == "oxford_pets":
        download_oxford_pets(data_dir)
    elif dataset_name == "stanford_cars":
        download_stanford_cars(data_dir)
    elif dataset_name == "oxford_flowers":
        download_oxford_flowers(data_dir)
    elif dataset_name == "food101":
        download_food101(data_dir)
    elif dataset_name == "fgvc_aircraft":
        download_fgvc_aircraft(data_dir)
    elif dataset_name == "sun397":
        download_sun397(data_dir)
    elif dataset_name == "dtd":
        download_dtd(data_dir)
    elif dataset_name == "eurosat":
        download_eurosat(data_dir)
    elif dataset_name == "ucf101":
        download_ucf101(data_dir)
    elif dataset_name in ["imagenet", "imagenetv2", "imagenet_sketch", "imagenet_a", "imagenet_r"]:
        download_imagenet_family(data_dir, dataset_name)
    else:
        print(f"Downloader for dataset '{dataset_name}' is not currently configured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and structure datasets for CeKALA")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory to store datasets")
    args = parser.parse_args()
    
    download_dataset(args.dataset, args.data_dir)
