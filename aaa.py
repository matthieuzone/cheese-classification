import os
os.system("python create_submition_with_ocr_and_sub.py")
os.system('kaggle competitions submit -c inf473v-cheese-classification-challenge -f submissionlast.csv -m "Message"')