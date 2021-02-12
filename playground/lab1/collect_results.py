import os
import re
import ast

save_dir = "/imgs/pt_deep/sigmoid/results"
folders = ["/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.48.54_w=1_d=1_v=2_npl=[2, 3]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000", "/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.49.02_w=1_d=1_v=2_npl=[2, 10, 3]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000", "/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.49.06_w=1_d=1_v=2_npl=[2, 10, 10, 3]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000", "/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.49.12_w=1_d=2_v=2_npl=[2, 2]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000", "/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.49.16_w=1_d=2_v=2_npl=[2, 10, 2]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000", "/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.49.19_w=1_d=2_v=2_npl=[2, 10, 10, 2]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000", "/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.49.28_w=1_d=3_v=2_npl=[2, 2]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000", "/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.49.35_w=1_d=3_v=2_npl=[2, 10, 2]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000", "/mnt/terra/xoding/fer-du/imgs/pt_deep/sigmoid/pt_deep_2021-02-02--00.49.39_w=1_d=3_v=2_npl=[2, 10, 10, 2]_act=sigmoid_eta=0.1_lambda=0.0001_epochs=30000"]

i=1
for path in folders:
    img = path + "/plot_train.png"
    txt = path + "/results.txt"
    os.system(f'cp "{img}" "{save_dir}/{i}.png"')
    os.system(f'echo "{i}" >> "{save_dir}/all_results.txt"')
    # os.system(f'cat "{txt}" >> "{save_dir}/all_results.txt"')
    with open(txt, "r") as f:
        parts = re.compile("{|}").split(f.read())
        params = parts[1]
        results = parts[3]
        with open(f"{save_dir}/all_results.txt", "a") as r:
            r.write(results)
            r.write("\n")
    i+=1
    # print()
os.system(f'montage -density 300 -tile 3x0 -geometry +0+0 -border 3 "{save_dir}/*.png" "{save_dir}/grid.png"')
os.system(f'code "{save_dir}/all_results.txt"')
