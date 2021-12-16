import glob


def run_normalization():
    annot_list = sorted(glob.glob('labels/test/*.txt', recursive=True))

    for annot_name in annot_list:
        print(annot_name)
        image_w = 480
        image_h = 320
        annot_list = get_annotations(annot_name)
        normalized_annot_list = []
        for m in annot_list:
            x, y, w, h = m
            x_center = (x + w / 2) / image_w
            y_center = (y + h / 2) / image_h
            width = w / image_w
            height = h / image_h

            normalized_annot_list.append((x_center, y_center, width, height))

        write_normalized_annotations(annot_name, normalized_annot_list)


def get_annotations(annot_name):
    with open(annot_name, 'r') as f:
        lines = f.readlines()
        annot = []
        for line in lines:
            l_arr = line.split(" ")[1:5]
            l_arr = [int(i) for i in l_arr]
            annot.append(l_arr)
        f.close()
    return annot


def write_normalized_annotations(annot_name, normalized_annot_list):
    with open(annot_name, 'w') as f:
        new_annot = ""
        for normalized_annot in normalized_annot_list:
            x, y, w, h = normalized_annot
            new_annot += f"0 {x} {y} {w} {h}\n"
        f.write(new_annot)
        f.close()


if __name__ == '__main__':
    run_normalization()
