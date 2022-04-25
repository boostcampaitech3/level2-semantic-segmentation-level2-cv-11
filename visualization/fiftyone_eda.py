"""
라이브러리
apt-get install curl
apt-get install libcurl3
apt-get install libcurl4-openssl-dev

pip install fiftyone

실행방법
python fiftyone_eda.py
"""
import fiftyone as fo
import argparse

def main(arg):
    classes = ['Backgroud',
         'General trash',
         'Paper',
         'Paper pack',
         'Metal',
         'Glass',
         'Plastic',
         'Styrofoam',
         'Plastic bag',
         'Battery',
         'Clothing']
    mask_label = {label:name for label,name in enumerate(classes)}
    data_dir = '/opt/ml/input/data/'
    anno_json = '/opt/ml/input/data/' + arg.anno_json
    print(data_dir)
    print(anno_json)
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_dir,
        labels_path=anno_json,
        label_types=["segmentations"],
        classes=classes[1:],
    )
    dataset.default_mask_targets = mask_label
    session = fo.launch_app(dataset, port=arg.port, address="0.0.0.0")
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='/opt/ml/input/data/',
                        help='imageData directory: "trian" or "test". default is "train"')
    parser.add_argument('--anno_json', '-a', type=str, default="val.json", # FIXME
                        help='image json file')
    parser.add_argument('--port', '-p', type=int, default=30001,
                        help='Port Number')
    args = parser.parse_args()
    main(args)
