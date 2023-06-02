import json,os
import glob,random,shutil

def softLinkImg(p_src, p_dst):
    # 软连接 图片
    #p_src = os.path.join(p_src_dir, df["file_name"])
    #p_dst = os.path.join(p_dst_dir, df["file_name"])
    cmd = f"ln -s {p_src} {p_dst}"
    if os.path.exists(p_dst):
        # print(f"Dst File Exsited. Src is {p_src}")
        pass
    else:
        os.system(cmd)

os.makedirs('./train_images',exist_ok=True)
os.makedirs('./train_gts',exist_ok=True)
os.makedirs('./test_images',exist_ok=True)
os.makedirs('./test_gts',exist_ok=True)
json_files = glob.glob('./TableSplerge_line/pdf2/*.json')
for json_file in json_files:
    img_file = json_file.replace('.json','.jpg')
    basename = os.path.basename(json_file).replace('.json','')
    txt_file = json_file.replace('.json','.txt')
    gt_txt = open(txt_file,'w')
    with open(json_file,'r') as f:
        json_ = json.load(f)
        for shape_ in json_['shapes']:
            if shape_['label'] == 'line_area_hori':
                [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] = shape_['points']
                points = [str(int(x)) for x in [x1,y1,x2,y2,x3,y3,x4,y4]]
                gt_txt.write(','.join(points)+',---\n')
                
            elif shape_['label'] == 'line_area_vert':
                [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] = shape_['points']
                points = [str(int(x)) for x in [x1,y1,x2,y2,x3,y3,x4,y4]]
                gt_txt.write(','.join(points)+',|||\n')
    gt_txt.close()

    

    if random.random()<0.2:
        trainval = 'test'
    else:
        trainval = 'train'

    p_src = img_file
    p_dst = './' + trainval + '_images/' + basename + '.jpg'
    print(p_src,p_dst)
    shutil.copyfile(p_src,p_dst)
    #softLinkImg(p_src, p_dst)   
    p_src = txt_file
    p_dst = './' + trainval + '_gts/' + basename + '.txt'
    #softLinkImg(p_src, p_dst)  
    shutil.copyfile(p_src,p_dst)
    print(p_src,p_dst)  





