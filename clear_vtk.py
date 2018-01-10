import os

root_dir = '/opt/multi_task_data_backup/'
cleared_list = ['.vtk','.vvi']

for project_name in os.listdir(root_dir):
    project_dir = root_dir+project_name
    for item_name in os.listdir(project_dir):
        item_dir = project_dir+'/'+item_name
        for file_name in os.listdir(item_dir):
            for cleared_name in cleared_list:
                if cleared_name in file_name:
                    file_dir = item_dir+'/'+file_name
                    print file_dir
                    try:
                        os.remove(file_dir)
                    except Exception,e:
                        continue