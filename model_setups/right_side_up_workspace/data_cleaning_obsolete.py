from stuff import OCRPipeline
import os, json, time, csv
import cv2

BASE_DIR = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/dataset"
CLASSIFIED_BASE_DIR = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/classified"
MISCLASSIFIED_BASE_DIR = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/misclassified"

directories = os.listdir(BASE_DIR)
directories.sort()

key = None
with open("/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/ocr_api_key.json") as file:
    key = json.loads(file.read())["key"]


with open("d_labels.csv") as file:
    with open("cleaned_labels.csv","a+") as write_file:
        writer = csv.writer(write_file)
        reader = csv.reader(file)

        for i,line in enumerate(reader):
            file_path = f"data{line[-1]}.png"
            path = os.path.join(BASE_DIR,file_path)
            if not os.path.exists(path):
                print(f"File {file_path} doesn't exist, continuing")
                continue
            img = cv2.imread(path)
            
            pipeline = OCRPipeline(img,key)
            pipeline.ocr_main()
            name = pipeline.extract_card_name()

            if name.strip().lower() == line[0].strip().lower():
                old = os.path.join(BASE_DIR,file_path)
                new = os.path.join(CLASSIFIED_BASE_DIR,file_path)
                os.rename(old,new)
                writer.writerow(line)
            else:
                old = os.path.join(BASE_DIR,file_path)
                new = os.path.join(MISCLASSIFIED_BASE_DIR,file_path)
                os.rename(old,new)  
            time.sleep(.1)

            print(f"{i+1}/{len(directories)} files done")
        


