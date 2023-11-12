################  test attribute selection model  ################
#### AWA2 ####
python attribute_selection_main.py --testing True --dataset AWA2 --attr_num 85 --data_path /home/c402/data/Dataset/Animals_with_Attributes2/JPEGImages
#### CUB ####
python attribute_selection_main.py --testing True --dataset CUB --attr_num 312 --data_path /home/c402/data/Dataset/CUB_200_2011/images
#### SUN ####
python attribute_selection_main.py --testing True --dataset SUN --attr_num 102 --data_path /home/c402/data/Dataset/SUN/images


# /home/c305/project/DATA_SET/CUB_200_2011/images
# /home/c305/project/DATA_SET/SUN/images
# /home/c305/backup_project/XL/Dataset/Animals_with_Attributes2/JPEGImages

python main.py --training --dataset SUN --attr_num 102 --data_path /home/c305/project/DATA_SET/SUN/images --mat_path /home/c305/project/DATA_SET/xlsa17/data --beta1 0.5 --beta2 0.5
python main.py --training --dataset CUB --attr_num 312 --data_path /home/c305/project/DATA_SET/CUB_200_2011/images --mat_path /home/c305/project/DATA_SET/xlsa17/data --beta1 0.5 --beta2 0.5
python main.py --training --dataset AWA2 --attr_num 85 --data_path /home/c305/backup_project/XL/Dataset/Animals_with_Attributes2/JPEGImages --mat_path /home/c305/project/DATA_SET/xlsa17/data --beta1 0.0 --beta2 1.0
