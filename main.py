from dotenv import load_dotenv

from terrain_segmentation.roboflow import RoboflowClient

load_dotenv()

def main():
    print("Connecting to roboflow...")
    roboflow = RoboflowClient()
    print("Connected to roboflow")
    roboflow.getDataset("dataset_id")

if __name__ == "__main__":
    main()